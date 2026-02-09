import cv2
import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
import sys

# ==============================================================================
# [설정] 사용자 정의 상수 (Configuration)
# ==============================================================================
class Config:
    # 이미지 파일 경로 (jpg 또는 png)
    IMAGE_PATH = 'parking_lot.png' 

    # 주차 칸 좌표 설정 (이전 단계에서 확보한 값)
    SPOT_W = 40
    SPOT_H = 80
    GAP_X = 40
    
    # Y축 줄 위치 (노랑, 주황, 초록, 파랑)
    ROWS_Y = [75, 169, 348, 436]

    # 블록 시작점 계산 (빨간 박스 기준점 300, 709 역산)
    LEFT_BLOCK_START_X = 300 - (7 * GAP_X) 
    RIGHT_BLOCK_START_X = 429

    # 3D 렌더링 설정
    CAMERA_DEPTH = 500.0   # 카메라 거리 (멀수록 작게 보임)
    
    # 색상 정의 (R, G, B)
    COLOR_EMPTY = (0.0, 1.0, 0.0)     # 초록 (빈자리)
    COLOR_OCCUPIED = (1.0, 0.0, 0.0)  # 빨강 (주차됨)
    COLOR_CURSOR = (1.0, 1.0, 0.0)    # 노랑 (커서)

    # 감지 임계값 (표준편차)
    DETECT_THRESHOLD = 20

# ==============================================================================
# [Logic] 주차장 데이터 및 감지 로직 관리 클래스
# ==============================================================================
class ParkingManager:
    def __init__(self):
        self.spots = [] # 주차 칸 리스트
        self.current_idx = 1 * 16 + 7 # 초기 커서 위치 (23번)
        self.total_cars = 0
        self._init_spots()
        
    def _init_spots(self):
        """설정된 좌표를 기반으로 주차 칸 리스트 생성"""
        self.spots = []
        for r_idx, y in enumerate(Config.ROWS_Y):
            # 왼쪽 블록 (8칸)
            for i in range(8):
                x = Config.LEFT_BLOCK_START_X + (i * Config.GAP_X)
                self.spots.append({'x': x, 'y': y, 'w': Config.SPOT_W, 'h': Config.SPOT_H, 'occupied': False})
            # 오른쪽 블록 (8칸)
            for i in range(8):
                x = Config.RIGHT_BLOCK_START_X + (i * Config.GAP_X)
                self.spots.append({'x': x, 'y': y, 'w': Config.SPOT_W, 'h': Config.SPOT_H, 'occupied': False})

    def update_occupancy(self, gray_img):
        """이미지(Gray)를 분석하여 차량 유무 업데이트"""
        h, w = gray_img.shape
        self.total_cars = 0
        
        for spot in self.spots:
            x, y, sw, sh = int(spot['x']), int(spot['y']), int(spot['w']), int(spot['h'])
            
            # 이미지 범위 내인지 확인
            if x + sw < w and y + sh < h:
                roi = gray_img[y:y+sh, x:x+sw]
                std_dev = np.std(roi)
                
                if std_dev > Config.DETECT_THRESHOLD:
                    spot['occupied'] = True
                    self.total_cars += 1
                else:
                    spot['occupied'] = False
        
        return self.total_cars

    def move_cursor(self, direction):
        """화살표 키 입력에 따른 커서 이동"""
        if direction == 'UP' and self.current_idx >= 16:
            self.current_idx -= 16
        elif direction == 'DOWN' and self.current_idx < 48: # 64 - 16
            self.current_idx += 16
        elif direction == 'LEFT' and self.current_idx > 0:
            self.current_idx -= 1
        elif direction == 'RIGHT' and self.current_idx < 63:
            self.current_idx += 1
            
    def get_current_spot(self):
        return self.spots[self.current_idx]

# ==============================================================================
# [Renderer] OpenGL 3D 시각화 클래스
# ==============================================================================
class GLRenderer:
    def __init__(self, img_path):
        # OpenCV 이미지 로드
        self.img = cv2.imread(img_path)
        if self.img is None:
            # jpg가 없으면 png 시도
            self.img = cv2.imread(img_path.replace('.jpg', '.png'))
            if self.img is None:
                print(f"[Error] 이미지를 찾을 수 없습니다: {img_path}")
                sys.exit()
            
        self.h, self.w = self.img.shape[:2]
        self.texture_id = None
        
        # 가상 카메라 파라미터 계산 (투영 행렬용)
        self.focal_length = self.w  # 초점거리
        self.cx, self.cy = self.w / 2, self.h / 2
        
    def init_gl(self):
        """OpenGL 초기 설정 및 텍스처 로드"""
        glClearColor(0.0, 0.0, 0.0, 0.0) # 배경색 검정
        glEnable(GL_DEPTH_TEST)          # 깊이 테스트 활성화 (3D 필수)
        
        # 텍스처 생성 및 바인딩
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        
        # 텍스처 파라미터 (반복, 필터링)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        # [중요] 이미지 데이터 정렬 (사선 깨짐 방지)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        
        # OpenCV(BGR) -> OpenGL(RGB) 변환 및 상하 반전
        img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.flip(img_rgb, 0) # OpenGL 좌표계에 맞춰 반전
        
        # 텍스처 업로드
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.w, self.h, 0, GL_RGB, GL_UNSIGNED_BYTE, img_rgb)

    def get_projection_matrix(self):
        """OpenCV 카메라 모델을 OpenGL 투영 행렬로 변환"""
        near, far = 0.1, 2000.0
        return np.array([
            [2*self.focal_length/self.w, 0.0, (self.w - 2*self.cx)/self.w, 0.0],
            [0.0, 2*self.focal_length/self.h, (-self.h + 2*self.cy)/self.h, 0.0],
            [0.0, 0.0, (-far - near)/(far - near), -2.0*far*near/(far-near)],
            [0.0, 0.0, -1.0, 0.0]
        ])

    def draw_background(self):
        """배경(주차장 바닥) 이미지 그리기"""
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        
        F = Config.CAMERA_DEPTH
        mx = self.focal_length / F
        my = self.focal_length / F
        
        # 텍스처 좌표 매핑을 위한 경계값 계산
        left = self.cx / mx
        right = (self.w - self.cx) / mx
        top = self.cy / my
        bottom = (self.h - self.cy) / my

        glPushMatrix()
        glTranslatef(0.0, 0.0, -F) # 카메라 깊이만큼 뒤로 이동
        glColor3f(1.0, 1.0, 1.0)
        
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0); glVertex3f(-left, -bottom, 0.0)
        glTexCoord2f(1.0, 0.0); glVertex3f( right, -bottom, 0.0)
        glTexCoord2f(1.0, 1.0); glVertex3f( right, top, 0.0)
        glTexCoord2f(0.0, 1.0); glVertex3f(-left,  top, 0.0)
        glEnd()
        glPopMatrix()
        glDisable(GL_TEXTURE_2D)

    def draw_box(self, px, py, pw, ph, color, is_raised=False, is_wireframe=False):
        """
        2D 픽셀 좌표를 3D 박스로 변환하여 그리기
        is_raised: True면 입체적으로 튀어나오게 그림 (주차된 차)
        """
        F = Config.CAMERA_DEPTH
        z = -F
        
        # 픽셀 좌표 -> 3D 월드 좌표 변환
        x_3d = (px + pw/2 - self.cx) / self.focal_length * (-z)
        y_3d = -(py + ph/2 - self.cy) / self.focal_length * (-z) # Y축 반전
        
        # 3D 크기 계산
        w_3d = (pw / self.focal_length) * (-z)
        h_3d = (ph / self.focal_length) * (-z)
        
        # 깊이(두께) 설정
        if is_raised:
            d_3d = w_3d * 1.5 # 주차된 차는 높게 (폭의 1.5배)
            alpha = 0.6       # 반투명
            offset_z = d_3d / 2 # 바닥 위로 올라오게 중심 이동
        else:
            d_3d = 0.1        # 빈자리는 납작하게
            alpha = 0.3
            offset_z = 0

        glPushMatrix()
        glTranslatef(x_3d, y_3d, z + offset_z) 
        glScale(w_3d, h_3d, d_3d)
        
        # 1. 색상 채우기 (Solid)
        if not is_wireframe:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glColor4f(color[0], color[1], color[2], alpha)
            glutSolidCube(1.0)
            glDisable(GL_BLEND)
        
        # 2. 테두리 그리기 (Wireframe)
        glLineWidth(2.0)
        if is_wireframe: # 커서
            glLineWidth(4.0)
            glColor3f(*color)
            glScale(1.05, 1.05, 1.05) # 살짝 크게
        else:
            glColor3f(1.0, 1.0, 1.0) # 흰색 테두리
            
        glutWireCube(1.0)
        glPopMatrix()

# ==============================================================================
# [Main] 프로그램 실행 및 루프
# ==============================================================================
# 전역 인스턴스
renderer = None
manager = None

def display():
    """OpenGL 렌더링 루프"""
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    # 1. 투영 행렬 로드
    glMatrixMode(GL_PROJECTION)
    glLoadTransposeMatrixd(renderer.get_projection_matrix())
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
    # 2. 배경 그리기
    renderer.draw_background()
    
    # 3. 주차 상태 업데이트 (이미지 분석)
    gray = cv2.cvtColor(renderer.img, cv2.COLOR_BGR2GRAY)
    manager.update_occupancy(gray)
    
    # 4. 각 주차 칸 3D 렌더링
    for spot in manager.spots:
        # 주차된 차: 튀어나온 빨간 박스 (is_raised=True)
        # 빈 자리: 납작한 초록 박스 (is_raised=False)
        if spot['occupied']:
            renderer.draw_box(spot['x'], spot['y'], spot['w'], spot['h'], 
                              Config.COLOR_OCCUPIED, is_raised=True)
        else:
            renderer.draw_box(spot['x'], spot['y'], spot['w'], spot['h'], 
                              Config.COLOR_EMPTY, is_raised=False)

    # 5. 커서 그리기 (노란색 와이어프레임)
    cur_spot = manager.get_current_spot()
    renderer.draw_box(cur_spot['x'], cur_spot['y'], cur_spot['w'], cur_spot['h'],
                      Config.COLOR_CURSOR, is_raised=True, is_wireframe=True)

    glutSwapBuffers()
    
    # 윈도우 제목 업데이트
    status = "Occupied" if cur_spot['occupied'] else "Empty"
    title = f"3D Parking AI - Spot:{manager.current_idx} | Status:{status} | Cars:{manager.total_cars}/64"
    glutSetWindowTitle(title.encode("ascii"))

def special_keys(key, x, y):
    """키보드 입력 처리"""
    if key == GLUT_KEY_UP:    manager.move_cursor('UP')
    elif key == GLUT_KEY_DOWN:  manager.move_cursor('DOWN')
    elif key == GLUT_KEY_LEFT:  manager.move_cursor('LEFT')
    elif key == GLUT_KEY_RIGHT: manager.move_cursor('RIGHT')
    glutPostRedisplay()

def main():
    global renderer, manager
    
    # 인스턴스 생성
    renderer = GLRenderer(Config.IMAGE_PATH)
    manager = ParkingManager()
    
    # GLUT 초기화
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(renderer.w, renderer.h)
    glutCreateWindow(b"3D Parking System")
    
    # OpenGL 초기화 (창 생성 후 호출해야 함)
    renderer.init_gl()
    
    # 콜백 함수 등록
    glutDisplayFunc(display)
    glutSpecialFunc(special_keys)
    
    print("=========================================")
    print(" [ 3D Parking System Started ]")
    print(" - 방향키: 노란색 커서 이동")
    print(" - 빨간 박스 (튀어나옴): 주차된 차량")
    print(" - 초록 박스 (납작함): 빈 자리")
    print("=========================================")
    
    glutMainLoop()

if __name__ == "__main__":
    main()