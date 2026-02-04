'''
[프로젝트: 지능형 적응형 VO 시스템]
최종 통합 버전: VO + 2D Constraint + Lane Tracking

기능 요약:
1. Visual Odometry: ORB 기반 특징점 추적 및 포즈 추정
2. 2D Constraint: Y축 이동 및 Roll/Pitch 회전 제거로 지도 평탄화
3. Lane Tracking: Canny Edge & 2차 함수 피팅을 이용한 도로/벽면 경계 인식 (Memory 기능 포함)
4. Visualization: 주행 경로 및 인식된 라인 실시간 시각화
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import deque

# ==========================================
# 설정 (Configuration)
# ==========================================
VIDEO_PATH = "F1_Monaco.mp4"
GT_MAP_PATH = "Circuit_de_Monaco.webp"

# 파일명 설정
OUTPUT_VIDEO = "F1_Result_Lane_Tracking.mp4" 
OUTPUT_RESULT_IMG = "F1_Comparison_Lane_Tracking.png"

RESIZE_W = 640
RESIZE_H = 360
FOCAL = 700.0 
PP = (RESIZE_W / 2, RESIZE_H / 2)

class LaneTracker:
    def __init__(self):
        # 이전 프레임의 곡선 계수(a, b, c)를 저장할 변수
        self.prev_left_fit = None
        self.prev_right_fit = None
        
        # 부드러운 출력을 위한 가중치 (과거 값 비중 70%)
        self.alpha = 0.7 

    def fit_poly(self, shape, leftx, lefty, rightx, righty):
        """좌표점들을 2차 함수(ax^2 + bx + c)로 피팅"""
        left_fit_curve = None
        right_fit_curve = None
        
        # y좌표 생성 (화면 세로 픽셀)
        ploty = np.linspace(0, shape[0]-1, shape[0])

        # [왼쪽 라인 처리]
        if len(lefty) > 50: # 점이 충분히 많으면 새로 피팅
            try:
                left_fit = np.polyfit(lefty, leftx, 2)
                # 메모리 기능: 이전 값과 가중 평균
                if self.prev_left_fit is not None:
                    left_fit = self.alpha * self.prev_left_fit + (1 - self.alpha) * left_fit
                self.prev_left_fit = left_fit
            except TypeError:
                left_fit = self.prev_left_fit
        else: # 점이 부족하면 이전 값 사용
            left_fit = self.prev_left_fit

        # 곡선 좌표 계산
        if left_fit is not None:
            try:
                left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
                # 화면 밖으로 나가는 점들 정리
                pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
                left_fit_curve = pts_left.astype(np.int32)
            except: pass

        # [오른쪽 라인 처리] (로직 동일)
        if len(righty) > 50:
            try:
                right_fit = np.polyfit(righty, rightx, 2)
                if self.prev_right_fit is not None:
                    right_fit = self.alpha * self.prev_right_fit + (1 - self.alpha) * right_fit
                self.prev_right_fit = right_fit
            except TypeError:
                right_fit = self.prev_right_fit
        else:
            right_fit = self.prev_right_fit

        if right_fit is not None:
            try:
                right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
                pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
                right_fit_curve = pts_right.astype(np.int32)
            except: pass

        return left_fit_curve, right_fit_curve

    def detect_and_draw(self, img, mask_car):
        """이미지에서 엣지를 검출하고 라인을 그려주는 메인 함수"""
        # 1. Edge Detection (Canny)
        # 노이즈 제거를 위해 블러링
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        
        # 2. ROI 적용 (자동차 부분 제외 & 도로 영역 한정)
        # 사용자 마스크(차체) 적용 (차체 부분은 0으로 지움)
        edges = cv2.bitwise_and(edges, edges, mask=mask_car)
        
        # 추가 ROI: 화면의 상단 절반(하늘/건물)은 무시하고 하단만 봄
        roi_mask = np.zeros_like(edges)
        height, width = edges.shape
        # 사다리꼴 형태의 ROI
        polygons = np.array([
            [(0, height), (width, height), (width, int(height*0.4)), (0, int(height*0.4))]
        ])
        cv2.fillPoly(roi_mask, polygons, 255)
        masked_edges = cv2.bitwise_and(edges, roi_mask)

        # 3. 픽셀 추출 (왼쪽/오른쪽 분리)
        # 화면 중앙을 기준으로 좌우 분할
        midpoint = width // 2
        nonzero = masked_edges.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # 왼쪽 영역 픽셀들
        left_lane_inds = nonzerox < midpoint
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]

        # 오른쪽 영역 픽셀들
        right_lane_inds = nonzerox >= midpoint
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # 4. 피팅 및 그리기
        left_curve, right_curve = self.fit_poly(img.shape, leftx, lefty, rightx, righty)
        
        # 시각화용 빈 이미지
        line_img = np.zeros((height, width, 3), dtype=np.uint8)
        
        if left_curve is not None:
            cv2.polylines(line_img, [left_curve], False, (0, 255, 0), 5) # 초록색 굵은 선
        if right_curve is not None:
            cv2.polylines(line_img, [right_curve], False, (0, 255, 0), 5)

        return line_img

class MonocularVO:
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=3000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev_gray = None
        self.prev_kp = None
        self.prev_des = None
        self.cur_t = np.array([[0], [0], [0]], dtype=np.float64)
        self.cur_R = np.eye(3, dtype=np.float64)
        self.trajectory = []
        
        # Lane Tracker 인스턴스 생성
        self.lane_tracker = LaneTracker()

    def get_mask(self, height, width):
        """V4 대칭형 정밀 마스크 (차체 및 UI 제거)"""
        mask = np.full((height, width), 255, dtype=np.uint8)
        
        # 차량 실루엣
        pts = np.array([
            [int(width * 0.12), height],              
            [int(width * 0.12), int(height * 0.40)],  
            [int(width * 0.32), int(height * 0.40)],  
            [int(width * 0.35), int(height * 0.50)],  
            [int(width * 0.35), int(height * 0.48)],  
            [int(width * 0.65), int(height * 0.48)],  
            [int(width * 0.65), int(height * 0.50)],  
            [int(width * 0.68), int(height * 0.40)],  
            [int(width * 0.88), int(height * 0.40)],  
            [int(width * 0.88), height]               
        ], np.int32)
        cv2.fillPoly(mask, [pts], 0)
        
        # UI 영역
        ui_h = int(height * 0.35) 
        cv2.rectangle(mask, (0, height - ui_h), (int(width * 0.12), height), 0, -1)
        cv2.rectangle(mask, (int(width * 0.88), height - ui_h), (width, height), 0, -1)

        return mask

    def process_frame(self, frame):
        frame_resized = cv2.resize(frame, (RESIZE_W, RESIZE_H))
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        
        # 마스크 생성 (VO용 & Lane Tracking용 공통 사용)
        mask_car = self.get_mask(RESIZE_H, RESIZE_W)
        
        # [1] Lane Tracking 실행 및 그리기
        # 차량 마스크를 넘겨줘서 차체 엣지가 라인으로 오인되지 않게 함
        lane_vis = self.lane_tracker.detect_and_draw(gray, mask_car)
        
        # 원본 영상에 라인 합성 (투명도 적용)
        img_out = cv2.addWeighted(frame_resized, 1.0, lane_vis, 0.6, 0)

        # [2] Visual Odometry 실행 (ORB)
        kp, des = self.orb.detectAndCompute(gray, mask_car)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_kp = kp
            self.prev_des = des
            return img_out, self.get_trajectory_map()

        if des is not None and self.prev_des is not None:
            matches = self.bf.match(self.prev_des, des)
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = matches[:int(len(matches)*0.2)]
            
            if len(good_matches) >= 8:
                pts1 = np.float32([self.prev_kp[m.queryIdx].pt for m in good_matches])
                pts2 = np.float32([kp[m.trainIdx].pt for m in good_matches])
                
                E, mask_ransac = cv2.findEssentialMat(pts1, pts2, focal=FOCAL, pp=PP, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                
                if E is not None:
                    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, focal=FOCAL, pp=PP)
                    
                    # [2D 평면 제약 적용]
                    if t[2] > 0.1: 
                        yaw = np.arctan2(R[0, 2], R[2, 2])
                        R_planar = np.array([
                            [np.cos(yaw), 0, np.sin(yaw)],
                            [0,           1, 0          ],
                            [-np.sin(yaw), 0, np.cos(yaw)]
                        ])
                        t_planar = np.array([t[0], [0], t[2]]) 
                        
                        scale = 1.0 
                        self.cur_t = self.cur_t + scale * self.cur_R.dot(t_planar)
                        self.cur_R = self.cur_R.dot(R_planar)
                
                # 매칭 점 그리기
                for i, m in enumerate(good_matches):
                    if i < 50:
                        pt1 = tuple(map(int, pts1[i]))
                        pt2 = tuple(map(int, pts2[i]))
                        cv2.line(img_out, pt1, pt2, (0, 0, 255), 1) # 빨간색 선
                        cv2.circle(img_out, pt2, 3, (0, 0, 255), -1)

        self.trajectory.append((float(self.cur_t[0]), float(self.cur_t[2])))
        self.prev_gray = gray
        self.prev_kp = kp
        self.prev_des = des
        return img_out, self.get_trajectory_map()

    def get_trajectory_map(self):
        map_img = np.zeros((RESIZE_H, RESIZE_H, 3), dtype=np.uint8)
        if len(self.trajectory) < 2: return map_img
        
        path = np.array(self.trajectory)
        min_x, max_x = np.min(path[:,0]), np.max(path[:,0])
        min_z, max_z = np.min(path[:,1]), np.max(path[:,1])
        
        range_max = max(max_x - min_x, max_z - min_z)
        scale_draw = (RESIZE_H * 0.8) / range_max if range_max > 0 else 1.0
        
        mid_x = (min_x + max_x) / 2
        mid_z = (min_z + max_z) / 2
        center_screen = RESIZE_H // 2
        
        for i in range(1, len(path)):
            x1, z1 = path[i-1]
            x2, z2 = path[i]
            draw_x1 = int((x1 - mid_x) * scale_draw) + center_screen
            draw_y1 = int(-(z1 - mid_z) * scale_draw) + center_screen
            draw_x2 = int((x2 - mid_x) * scale_draw) + center_screen
            draw_y2 = int(-(z2 - mid_z) * scale_draw) + center_screen
            cv2.line(map_img, (draw_x1, draw_y1), (draw_x2, draw_y2), (0, 255, 255), 1)
            
        cur_x, cur_z = path[-1]
        cx = int((cur_x - mid_x) * scale_draw) + center_screen
        cy = int(-(cur_z - mid_z) * scale_draw) + center_screen
        cv2.circle(map_img, (cx, cy), 3, (0, 0, 255), -1)
            
        return map_img

def save_comparison_result(trajectory_points):
    if os.path.exists(GT_MAP_PATH):
        gt_img = cv2.imread(GT_MAP_PATH)
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
    else:
        gt_img = np.zeros((500, 500, 3), dtype=np.uint8)

    path = np.array(trajectory_points)
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(gt_img)
    plt.title("Ground Truth")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    if len(path) > 0:
        plt.plot(path[:, 0], -path[:, 1], color='blue', linewidth=2, label='VO Path')
        plt.scatter(path[0, 0], -path[0, 1], c='green', label='Start')
        plt.scatter(path[-1, 0], -path[-1, 1], c='red', marker='x', label='End')
        plt.title("Generated Trajectory (Lane + 2D)")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
    
    plt.savefig(OUTPUT_RESULT_IMG)
    plt.show()

def main():
    if not os.path.exists(VIDEO_PATH):
        print("영상 파일 없음.")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened(): return

    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        print("▶ H.264(avc1) 코덱 사용")
    except:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 30.0, (RESIZE_W + RESIZE_H, RESIZE_H))
    vo = MonocularVO()
    
    print("▶ 궤적 생성 시작 (Lane Tracking & 2D Constraint)...")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_vis, map_vis = vo.process_frame(frame)
        combined = np.hstack((frame_vis, map_vis))
        cv2.imshow("VO System - Lane Tracking", combined)
        out.write(combined)
        if cv2.waitKey(1) == ord('q'): break
            
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    save_comparison_result(vo.trajectory)
    print("✅ 완료")

if __name__ == "__main__":
    main()