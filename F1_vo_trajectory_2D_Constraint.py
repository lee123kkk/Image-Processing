'''
[프로젝트: 지능형 적응형 VO 시스템]
버전: V4 기반 + 2D 평면 제약(Ground Plane Constraint) 추가
파일형식: ORB Feature Matching 방식 (Optical Flow 아님)

기능:
1. 기존의 안정적인 ORB 특징점 매칭 사용
2. 2D 평면 제약 적용 (Y축 이동 제거, Roll/Pitch 회전 제거) -> 지도 평탄화
3. 정밀 마스킹(Symmetric Mask V4) 유지
4. 파일명 충돌 방지 처리
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 설정 (Configuration)
# ==========================================
VIDEO_PATH = "F1_Monaco.mp4"
GT_MAP_PATH = "Circuit_de_Monaco.webp"

# [파일명 변경] 기존 파일과 겹치지 않게 명확한 이름 부여
OUTPUT_VIDEO = "F1_Result_2D_Constraint.mp4" 
OUTPUT_RESULT_IMG = "F1_Comparison_2D_Constraint.png"

RESIZE_W = 640
RESIZE_H = 360
FOCAL = 700.0 
PP = (RESIZE_W / 2, RESIZE_H / 2)

class MonocularVO:
    def __init__(self):
        # 기존 방식대로 ORB 사용
        self.orb = cv2.ORB_create(nfeatures=3000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        self.prev_gray = None
        self.prev_kp = None
        self.prev_des = None
        
        self.cur_t = np.array([[0], [0], [0]], dtype=np.float64)
        self.cur_R = np.eye(3, dtype=np.float64)
        self.trajectory = []

    def get_mask(self, height, width):
        """V4에서 검증된 대칭형 정밀 마스크"""
        mask = np.full((height, width), 255, dtype=np.uint8)
        
        # 1. 차량 실루엣 (타이어+노즈)
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
        
        # 2. 양쪽 하단 UI 영역 (높이 35%)
        ui_h = int(height * 0.35) 
        cv2.rectangle(mask, (0, height - ui_h), (int(width * 0.12), height), 0, -1)
        cv2.rectangle(mask, (int(width * 0.88), height - ui_h), (width, height), 0, -1)

        return mask

    def process_frame(self, frame):
        frame_resized = cv2.resize(frame, (RESIZE_W, RESIZE_H))
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        
        mask = self.get_mask(RESIZE_H, RESIZE_W)
        kp, des = self.orb.detectAndCompute(gray, mask)
        
        img_out = frame_resized.copy()
        
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
                    
                    # [핵심 추가] 2D 평면 제약 적용
                    if t[2] > 0.1: # 전진 방향일 때만 업데이트
                        
                        # 1. 회전 행렬에서 Yaw(좌우 회전) 성분만 추출
                        # (차량은 평지에서 좌우로만 회전한다고 가정)
                        yaw = np.arctan2(R[0, 2], R[2, 2])
                        
                        # 2. Roll, Pitch를 제거한 깨끗한 2D 회전 행렬 생성
                        R_planar = np.array([
                            [np.cos(yaw), 0, np.sin(yaw)],
                            [0,           1, 0          ],
                            [-np.sin(yaw), 0, np.cos(yaw)]
                        ])
                        
                        # 3. 이동 벡터에서 Y축(위아래) 성분 제거
                        t_planar = np.array([t[0], [0], t[2]]) 
                        
                        scale = 1.0 
                        
                        # 4. 업데이트 (평면 제약이 적용된 R_planar, t_planar 사용)
                        self.cur_t = self.cur_t + scale * self.cur_R.dot(t_planar)
                        self.cur_R = self.cur_R.dot(R_planar)
                
                # 시각화 (기존 유지)
                for i, m in enumerate(good_matches):
                    if i < 50:
                        pt1 = tuple(map(int, pts1[i]))
                        pt2 = tuple(map(int, pts2[i]))
                        cv2.line(img_out, pt1, pt2, (0, 255, 0), 1)
                        cv2.circle(img_out, pt2, 3, (0, 0, 255), -1)

        # 마스크 디버깅 시각화
        mask_overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        img_masked_debug = cv2.bitwise_and(img_out, mask_overlay)
        img_out = cv2.addWeighted(img_out, 0.3, img_masked_debug, 0.7, 0)

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
        plt.title("Generated Trajectory (2D Constraint)")
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
    
    print("▶ 궤적 생성 시작 (ORB + 2D Constraint)...")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_vis, map_vis = vo.process_frame(frame)
        combined = np.hstack((frame_vis, map_vis))
        cv2.imshow("VO System - 2D Constraint", combined)
        out.write(combined)
        if cv2.waitKey(1) == ord('q'): break
            
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    save_comparison_result(vo.trajectory)
    print("✅ 완료")

if __name__ == "__main__":
    main()