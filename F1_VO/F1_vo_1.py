'''
ORB를 메인으로 하되, 
상황에 따라 SIFT로 전환하거나 IMU 센서 모드로 대응하는 지능형 VO 시스템

상태 머신(State Machine) 기반으로 동작
'''

'''
구현된 핵심 로직 (State Machine)

기본 주행 (Normal Mode):

빠른 ORB를 사용하여 실시간 트래킹을 수행합니다.
Inlier가 충분하면(400개 이상) 계속 ORB를 유지합니다.

정밀 주행 (Precision Mode - Exception Handling):

ORB의 Inlier가 급감(400개 미만)하면, 
즉시 SIFT를 호출하여 동일 프레임에서 다시 시도합니다. 
(헤어핀 커브 등 회전/블러 구간 대응)
SIFT가 더 나은 결과를 내면 해당 데이터를 채택합니다.

센서 융합 (Blind/IMU Mode):

두 알고리즘 모두 Inlier가 너무 적으면(50개 미만, 터널 탈출 등), 
카메라 입력을 차단합니다.
화면에 "SENSOR FUSION ACTIVATED" 경고를 띄우고, 
시각 정보 대신 관성 센서(IMU)에 의존한다는 것을 시뮬레이션합니다.
'''


import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime

# ==========================================
# 설정 (Configuration)
# ==========================================
VIDEO_PATH = "F1_Monaco.mp4"
OUTPUT_VIDEO_NAME = "F1_Advanced_VO.mp4"
RESIZE_WIDTH = 640
MAX_FRAMES = 10000  # 전체 영상

# 임계값 설정
THRESH_UNSTABLE = 350  # ORB -> SIFT 전환 기준
THRESH_CRITICAL = 50   # SIFT -> IMU 전환 기준

# 상태 상수
MODE_NORMAL = 0   # ORB
MODE_RECOVERY = 1 # SIFT
MODE_BLIND = 2    # IMU

MODE_NAMES = {
    MODE_NORMAL: "NORMAL (ORB)",
    MODE_RECOVERY: "PRECISION (SIFT)",
    MODE_BLIND: "BLIND (IMU FUSION)"
}

MODE_COLORS = {
    MODE_NORMAL: (0, 255, 0),    # Green
    MODE_RECOVERY: (0, 215, 255),# Gold/Yellow
    MODE_BLIND: (0, 0, 255)      # Red
}

class IntelligentVO:
    def __init__(self):
        # 1. ORB (Main)
        self.orb = cv2.ORB_create(nfeatures=2000)
        self.bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # 2. SIFT (Backup)
        self.sift = cv2.SIFT_create()
        self.bf_sift = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        self.prev_gray = None
        self.prev_kp_orb = None
        self.prev_des_orb = None
        self.prev_kp_sift = None
        self.prev_des_sift = None
        
        self.imu_drift_x = 0
        
    def process_frame(self, frame_curr):
        h, w = frame_curr.shape[:2]
        scale = RESIZE_WIDTH / w
        frame_resized = cv2.resize(frame_curr, (int(w * scale), int(h * scale)))
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        
        current_mode = MODE_NORMAL
        inlier_count = 0
        final_matches_vis = frame_resized.copy()
        
        # Step 1: Default - Try ORB
        kp_orb, des_orb = self.orb.detectAndCompute(gray, None)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_kp_orb, self.prev_des_orb = kp_orb, des_orb
            self.prev_kp_sift, self.prev_des_sift = None, None
            return frame_resized, MODE_NORMAL, 0

        matches_orb, inliers_orb = self._match_and_filter(
            self.prev_kp_orb, self.prev_des_orb, kp_orb, des_orb, self.bf_orb, 0.75
        )
        count_orb = len(inliers_orb)
        
        # Step 2: Check & Exception Handling
        if count_orb >= THRESH_UNSTABLE:
            current_mode = MODE_NORMAL
            inlier_count = count_orb
            final_matches_vis = self._draw_result(frame_resized, self.prev_kp_orb, kp_orb, matches_orb, inliers_orb, (0, 255, 0))
            
            self.prev_kp_orb, self.prev_des_orb = kp_orb, des_orb
            self.prev_kp_sift, self.prev_des_sift = None, None 

        else:
            # ORB 불안정 -> SIFT 시도
            kp_sift, des_sift = self.sift.detectAndCompute(gray, None)
            
            if self.prev_des_sift is None:
                self.prev_kp_sift, self.prev_des_sift = self.sift.detectAndCompute(self.prev_gray, None)
            
            matches_sift, inliers_sift = self._match_and_filter(
                self.prev_kp_sift, self.prev_des_sift, kp_sift, des_sift, self.bf_sift, 0.75
            )
            count_sift = len(inliers_sift)
            
            if count_sift >= THRESH_CRITICAL:
                current_mode = MODE_RECOVERY
                inlier_count = count_sift
                final_matches_vis = self._draw_result(frame_resized, self.prev_kp_sift, kp_sift, matches_sift, inliers_sift, (0, 215, 255))
                
                self.prev_kp_sift, self.prev_des_sift = kp_sift, des_sift
                self.prev_kp_orb, self.prev_des_orb = kp_orb, des_orb
                
            else:
                # 둘 다 실패 -> IMU 모드
                current_mode = MODE_BLIND
                inlier_count = 0
                
                overlay = final_matches_vis.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, final_matches_vis, 0.3, 0, final_matches_vis)
                
                self.imu_drift_x = (self.imu_drift_x + 5) % w
                cv2.line(final_matches_vis, (w//2, h//2), (w//2, h//2 - 50), (0,0,255), 3)
                cv2.putText(final_matches_vis, "IMU DATA PROCESSING...", (w//2 - 100, h//2 + 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                
        if current_mode != MODE_BLIND:
            self.prev_gray = gray
            
        return final_matches_vis, current_mode, inlier_count

    def _match_and_filter(self, kp1, des1, kp2, des2, matcher, ratio):
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return [], []
        
        knn_matches = matcher.knnMatch(des1, des2, k=2)
        good = []
        for m_n in knn_matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < ratio * n.distance:
                    good.append(m)
        
        inliers = []
        if len(good) >= 4:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if mask is not None:
                inliers = [m for m, val in zip(good, mask.ravel()) if val == 1]
                
        return good, inliers

    def _draw_result(self, img, kp1, kp2, matches, inliers, color):
        return cv2.drawMatches(img, kp1, img, kp2, inliers, None,
                               matchColor=color,
                               singlePointColor=None,
                               flags=2)

# ==========================================
# 그래프 그리기 함수 (이 부분이 누락되었었습니다)
# ==========================================
def plot_advanced_graph(frames, inliers, modes):
    plt.figure(figsize=(14, 6))
    
    frames = np.array(frames)
    inliers = np.array(inliers)
    modes = np.array(modes)
    
    # 전체 흐름선
    plt.plot(frames, inliers, color='gray', alpha=0.3, label='Inlier Count')
    
    # 상태별 산점도 (Scatter)
    mask_normal = (modes == MODE_NORMAL)
    mask_recovery = (modes == MODE_RECOVERY)
    mask_blind = (modes == MODE_BLIND)
    
    if np.any(mask_normal):
        plt.scatter(frames[mask_normal], inliers[mask_normal], c='green', s=2, label='Normal (ORB)', alpha=0.6)
    if np.any(mask_recovery):
        plt.scatter(frames[mask_recovery], inliers[mask_recovery], c='orange', s=15, label='Precision (SIFT)', zorder=5)
    if np.any(mask_blind):
        plt.scatter(frames[mask_blind], inliers[mask_blind], c='red', s=20, label='Blind (IMU)', zorder=10)
    
    plt.title("Intelligent VO: Adaptive Algorithm Switching System")
    plt.xlabel("Frame Number")
    plt.ylabel("Inlier Count (Feature Matches)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    filename = f"F1_Advanced_Result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename, dpi=300)
    print(f"✅ 그래프 저장 완료: {filename}")
    plt.show()

# ==========================================
# 메인 실행 함수 (동영상 저장 버그 수정됨)
# ==========================================
def main():
    vo = IntelligentVO()
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # [수정] 첫 프레임으로 정확한 해상도 측정
    ret, first_frame = cap.read()
    if not ret: return
        
    h_orig, w_orig = first_frame.shape[:2]
    scale = RESIZE_WIDTH / w_orig
    first_resized = cv2.resize(first_frame, (int(w_orig * scale), int(h_orig * scale)))
    
    h, w = first_resized.shape[:2] # 이 크기가 VideoWriter와 정확히 일치해야 함
    
    # [수정] 코덱 설정 (avc1 권장)
    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1') 
    except:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(OUTPUT_VIDEO_NAME, fourcc, 30.0, (w, h))

    history_frames = []
    history_inliers = []
    history_modes = []

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # 다시 처음으로
    frame_idx = 0
    
    print(f"▶ 지능형 VO 시스템 시작... (해상도: {w}x{h})")
    
    while frame_idx < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
            
        vis_frame, mode, inliers = vo.process_frame(frame)
        
        # [안전장치] 크기 강제 맞춤
        if vis_frame.shape[:2] != (h, w):
            vis_frame = cv2.resize(vis_frame, (w, h))
        
        color = MODE_COLORS[mode]
        text = f"[{MODE_NAMES[mode]}] Inliers: {inliers}"
        
        cv2.rectangle(vis_frame, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.putText(vis_frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        out.write(vis_frame)
        cv2.imshow("Intelligent VO System", vis_frame)
        
        history_frames.append(frame_idx)
        history_inliers.append(inliers)
        history_modes.append(mode)
        
        frame_idx += 1
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # 저장 확인
    if os.path.exists(OUTPUT_VIDEO_NAME):
        size_mb = os.path.getsize(OUTPUT_VIDEO_NAME) / (1024 * 1024)
        print(f"✅ 동영상 저장 완료: {OUTPUT_VIDEO_NAME} ({size_mb:.2f} MB)")
    
    print("그래프 작성 중...")
    plot_advanced_graph(history_frames, history_inliers, history_modes)

if __name__ == "__main__":
    main()
