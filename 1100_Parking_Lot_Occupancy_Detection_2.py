import cv2
import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import * # 텍스트 렌더링용 추가
import sys

# ==============================================================================
# [설정] 사용자 정의 상수
# ==============================================================================
class Config:
    IMAGE_PATH = 'parking_lot.png' 
    
    # 정식 주차 칸 설정
    SPOT_W = 40; SPOT_H = 80; GAP_X = 40
    ROWS_Y = [75, 169, 348, 436]
    LEFT_BLOCK_START_X = 300 - (7 * GAP_X) 
    RIGHT_BLOCK_START_X = 429

    # 불법 주차 구역 (사용자 지정 좌표)
    ILLEGAL_ZONES = [
        {'x': 99, 'y': 287, 'w': 80, 'h': 37, 'desc': 'Left White Car'},
        {'x': 481, 'y': 8, 'w': 78, 'h': 34, 'desc': 'Top White Car'},
        {'x': 512, 'y': 162, 'w': 34, 'h': 7, 'desc': 'Middle Lane Car'},
        {'x': 709, 'y': 150, 'w': 37, 'h': 7, 'desc': 'Top Right Bumper'},
        {'x': 711, 'y': 516, 'w': 36, 'h': 10, 'desc': 'Bottom Right Bumper'}
    ]

    CAMERA_DEPTH = 500.0
    COLOR_EMPTY = (0.0, 1.0, 0.0); COLOR_OCCUPIED = (1.0, 0.0, 0.0)
    COLOR_CURSOR = (1.0, 1.0, 0.0); COLOR_ILLEGAL = (1.0, 0.5, 0.0)
    DETECT_THRESHOLD = 20

# ==============================================================================
# [Logic] 주차장 로직
# ==============================================================================
class ParkingManager:
    def __init__(self):
        self.spots = []
        self.illegal_cars = []
        self.current_idx = 23
        self.total_cars = 0 
        self._init_spots()
        
        for zone in Config.ILLEGAL_ZONES:
            self.illegal_cars.append(zone)

    def _init_spots(self):
        self.spots = []
        for r_idx, y in enumerate(Config.ROWS_Y):
            for i in range(8):
                x = Config.LEFT_BLOCK_START_X + (i * Config.GAP_X)
                self.spots.append({'x': x, 'y': y, 'w': Config.SPOT_W, 'h': Config.SPOT_H, 'occupied': False})
            for i in range(8):
                x = Config.RIGHT_BLOCK_START_X + (i * Config.GAP_X)
                self.spots.append({'x': x, 'y': y, 'w': Config.SPOT_W, 'h': Config.SPOT_H, 'occupied': False})

    def update_occupancy(self, gray_img):
        h, w = gray_img.shape
        self.total_cars = 0 
        
        for spot in self.spots:
            if self._check_roi(gray_img, spot):
                spot['occupied'] = True
                self.total_cars += 1 
            else:
                spot['occupied'] = False

    def _check_roi(self, gray, rect, threshold=Config.DETECT_THRESHOLD):
        x, y, sw, sh = int(rect['x']), int(rect['y']), int(rect['w']), int(rect['h'])
        if x + sw < gray.shape[1] and y + sh < gray.shape[0]:
            roi = gray[y:y+sh, x:x+sw]
            return np.std(roi) > threshold
        return False

    def move_cursor(self, d):
        if d=='UP' and self.current_idx>=16: self.current_idx-=16
        elif d=='DOWN' and self.current_idx<48: self.current_idx+=16
        elif d=='LEFT' and self.current_idx>0: self.current_idx-=1
        elif d=='RIGHT' and self.current_idx<63: self.current_idx+=1
    def get_current_spot(self): return self.spots[self.current_idx]

# ==============================================================================
# [Renderer] OpenGL
# ==============================================================================
class GLRenderer:
    def __init__(self, p):
        self.img = cv2.imread(p)
        if self.img is None: self.img = cv2.imread(p.replace('.jpg','.png'))
        if self.img is None: sys.exit("Image Load Error")
        self.h, self.w = self.img.shape[:2]; self.focal_length = self.w
        self.cx, self.cy = self.w/2, self.h/2
        
    def init_gl(self):
        glClearColor(0,0,0,0); glEnable(GL_DEPTH_TEST)
        self.tid = glGenTextures(1); glBindTexture(GL_TEXTURE_2D, self.tid)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        img = cv2.flip(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB), 0)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.w, self.h, 0, GL_RGB, GL_UNSIGNED_BYTE, img)

    def proj(self):
        return np.array([[2*self.focal_length/self.w,0,(self.w-2*self.cx)/self.w,0],
                         [0,2*self.focal_length/self.h,(-self.h+2*self.cy)/self.h,0],
                         [0,0,-1.0002,-0.20002],[0,0,-1,0]])

    def bg(self):
        glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D, self.tid)
        F=Config.CAMERA_DEPTH; m=self.focal_length/F
        l,r,t,b = self.cx/m, (self.w-self.cx)/m, self.cy/m, (self.h-self.cy)/m
        glPushMatrix(); glTranslatef(0,0,-F); glColor3f(1,1,1)
        glBegin(GL_QUADS); glTexCoord2f(0,0); glVertex3f(-l,-b,0); glTexCoord2f(1,0); glVertex3f(r,-b,0)
        glTexCoord2f(1,1); glVertex3f(r,t,0); glTexCoord2f(0,1); glVertex3f(-l,t,0); glEnd(); glPopMatrix(); glDisable(GL_TEXTURE_2D)

    def box(self, px, py, pw, ph, c, raised=False, wire=False):
        F=Config.CAMERA_DEPTH; z=-F
        x,y = (px+pw/2-self.cx)/self.focal_length*(-z), -(py+ph/2-self.cy)/self.focal_length*(-z)
        w,h = (pw/self.focal_length)*(-z), (ph/self.focal_length)*(-z)
        base_h = (Config.SPOT_W / self.focal_length) * (-z)
        d = base_h * 1.5 if raised else 0.1
        glPushMatrix(); glTranslatef(x,y,z+(d/2 if raised else 0)); glScale(w,h,d)
        if not wire:
            glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glColor4f(*c, 0.6); glutSolidCube(1); glDisable(GL_BLEND)
        glLineWidth(2 if not wire else 4); glColor3f(*(c if wire else (1,1,1))); glutWireCube(1.05 if wire else 1.0); glPopMatrix()

    # [추가됨] 2D 텍스트 오버레이 함수
    def draw_text_overlay(self, x, y, text):
        # 1. 2D Orthographic 모드로 전환
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.w, self.h, 0, -1, 1) # 좌상단이 (0,0)이 되도록 설정
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        glDisable(GL_DEPTH_TEST) # 3D 물체 위에 그려지도록 깊이 테스트 끄기
        
        # 2. 텍스트 설정 (노란색)
        glColor3f(1.0, 1.0, 0.0) 
        glRasterPos2i(x, y)
        
        # 3. 글자 렌더링
        for ch in text:
            glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, ord(ch))
            
        # 4. 3D 모드로 복구
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()

# ==============================================================================
# [Main]
# ==============================================================================
rend=None; mgr=None
def disp():
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
    
    # 1. 3D 장면 렌더링
    glMatrixMode(GL_PROJECTION); glLoadTransposeMatrixd(rend.proj())
    glMatrixMode(GL_MODELVIEW); glLoadIdentity()
    rend.bg(); mgr.update_occupancy(cv2.cvtColor(rend.img, cv2.COLOR_BGR2GRAY))
    
    for s in mgr.spots: 
        rend.box(s['x'],s['y'],s['w'],s['h'], Config.COLOR_OCCUPIED if s['occupied'] else Config.COLOR_EMPTY, s['occupied'])
    
    for c in mgr.illegal_cars:
        rend.box(c['x'],c['y'],c['w'],c['h'], Config.COLOR_ILLEGAL, True)
    
    cur=mgr.get_current_spot(); rend.box(cur['x'],cur['y'],cur['w'],cur['h'], Config.COLOR_CURSOR, True, True)
    
    # 2. [추가] 2D 텍스트 오버레이 (좌상단에 노란 글씨)
    info_text = f"Total Cars: {mgr.total_cars} / {len(mgr.spots)}"
    rend.draw_text_overlay(20, 40, info_text) # (x=20, y=40) 위치에 표시

    glutSwapBuffers()

def key(k,x,y):
    if k==GLUT_KEY_UP: mgr.move_cursor('UP')
    elif k==GLUT_KEY_DOWN: mgr.move_cursor('DOWN')
    elif k==GLUT_KEY_LEFT: mgr.move_cursor('LEFT')
    elif k==GLUT_KEY_RIGHT: mgr.move_cursor('RIGHT')
    glutPostRedisplay()

def main():
    global rend, mgr
    rend=GLRenderer(Config.IMAGE_PATH); mgr=ParkingManager()
    glutInit(); glutInitDisplayMode(GLUT_RGBA|GLUT_DOUBLE|GLUT_DEPTH)
    glutInitWindowSize(rend.w, rend.h); glutCreateWindow(b"Smart Parking System")
    rend.init_gl(); glutDisplayFunc(disp); glutSpecialFunc(key); glutMainLoop()

if __name__=="__main__": main()