'''
[프로젝트: 지능형 적응형 VO 시스템]
주행 궤적(Trajectory) 생성 및 정밀 마스킹(Car Profile Masking) 적용 코드 V2

개선 사항:
1. 마스킹 영역 재조정 (User Feedback 반영)
   - 화면 양쪽 끝(타이어 바깥쪽) 영역은 마스킹 해제 -> 배경 특징점 확보
   - 차체 중앙(노즈) 마스킹 높이 상향 -> 로고/글씨 영역 추적 방지
2. 차량 실루엣을 정교하게 따르는 다각형 좌표 적용
'''

'''
[프로젝트: 지능형 적응형 VO 시스템]
주행 궤적(Trajectory) 생성 및 정밀 마스킹(Car Profile Masking) 적용 코드 V3

개선 사항:
1. User Feedback 반영: 우하단(오른쪽 아래) 영역도 좌하단과 동일하게 마스킹 처리
   - 좌우 대칭형 UI 마스킹 적용 (하단 25%, 양쪽 12% 폭)
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
OUTPUT_VIDEO = "F1_Trajectory_Result_SymmetricMask.mp4"
OUTPUT_RESULT_IMG = "F1_Trajectory_Comparison_SymmetricMask.png"

RESIZE_W = 640
RESIZE_H = 360
FOCAL = 700.0 
PP = (RESIZE_W / 2, RESIZE_H / 2)

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

    def get_mask(self, height, width):
        """
        차체(타이어+노즈) 및 좌우 하단 구석(UI 영역)을 모두 가리는 마스크
        """
        # 1. 전체 영역을 '추적 가능(255)'으로 초기화
        mask = np.full((height, width), 255, dtype=np.uint8)
        
        # 2. 차량의 실루엣을 따르는 다각형 (차체 가리기)
        pts = np.array([
            # [1] 왼쪽 타이어 영역
            [int(width * 0.12), height],              
            [int(width * 0.12), int(height * 0.40)],  
            [int(width * 0.32), int(height * 0.40)],  

            # [2] 차체 중앙(노즈) 영역
            [int(width * 0.35), int(height * 0.50)],  
            [int(width * 0.35), int(height * 0.48)],  # 노즈 윗면 (로고 가림)
            [int(width * 0.65), int(height * 0.48)],  
            [int(width * 0.65), int(height * 0.50)],  

            # [3] 오른쪽 타이어 영역
            [int(width * 0.68), int(height * 0.40)],  
            [int(width * 0.88), int(height * 0.40)],  
            [int(width * 0.88), height]               
        ], np.int32)

        # 다각형 내부를 검은색(0)으로 채움
        cv2.fillPoly(mask, [pts], 0)
        
        # 3. [추가 수정] 좌우 하단 구석 마스킹 (UI/워터마크 제거)
        ui_h = int(height * 0.25) # 밑에서부터 25% 높이
        
        # (A) 좌측 하단 (기존)
        # 너비: 왼쪽 끝(0) ~ 12% 지점
        cv2.rectangle(mask, (0, height - ui_h), (int(width * 0.12), height), 0, -1)
        
        # (B) 우측 하단 (신규 추가 - 요청사항 반영)
        # 너비: 오른쪽 12% 지점(88%) ~ 오른쪽 끝(100%)
        cv2.rectangle(mask, (int(width * 0.88), height - ui_h), (width, height), 0, -1)

        return mask

    def process_frame(self, frame):
        frame_resized = cv2.resize(frame, (RESIZE_W, RESIZE_H))
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        
        # 마스크 적용
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
                    
                    if t[2] > 0.1: 
                        scale = 1.0 
                        self.cur_t = self.cur_t + scale * self.cur_R.dot(t)
                        self.cur_R = self.cur_R.dot(R)
                
                for i, m in enumerate(good_matches):
                    if i < 50:
                        pt1 = tuple(map(int, pts1[i]))
                        pt2 = tuple(map(int, pts2[i]))
                        cv2.line(img_out, pt1, pt2, (0, 255, 0), 1)
                        cv2.circle(img_out, pt2, 3, (0, 0, 255), -1)

        # 디버깅: 마스크 영역 시각화
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
        
        scale_draw = 1.0
        center_x, center_y = RESIZE_H // 2, RESIZE_H // 2
        path = np.array(self.trajectory)
        
        if len(path) > 10:
            min_x, max_x = np.min(path[:,0]), np.max(path[:,0])
            min_z, max_z = np.min(path[:,1]), np.max(path[:,1])
            range_max = max(max_x - min_x, max_z - min_z)
            if range_max > 0: scale_draw = (RESIZE_H * 0.8) / range_max
        
        for i in range(1, len(path)):
            x1, z1 = path[i-1]
            x2, z2 = path[i]
            draw_x1 = int((x1 - path[0][0]) * scale_draw) + center_x + 100
            draw_y1 = int(-(z1 - path[0][1]) * scale_draw) + center_y + 100
            draw_x2 = int((x2 - path[0][0]) * scale_draw) + center_x + 100
            draw_y2 = int(-(z2 - path[0][1]) * scale_draw) + center_y + 100
            cv2.line(map_img, (draw_x1, draw_y1), (draw_x2, draw_y2), (0, 255, 255), 1)
            
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
        plt.title("Generated Trajectory (Final Symmetric)")
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
    
    print("▶ 궤적 생성 시작 (우하단 마스킹 추가)...")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_vis, map_vis = vo.process_frame(frame)
        combined = np.hstack((frame_vis, map_vis))
        cv2.imshow("VO System - Symmetric Masking", combined)
        out.write(combined)
        if cv2.waitKey(1) == ord('q'): break
            
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    save_comparison_result(vo.trajectory)
    print("✅ 완료")

if __name__ == "__main__":
    main()
    