'''
[프로젝트: 지능형 적응형 VO 시스템 V5]
- 기능: Adaptive Flood Fill 기반 도로 인식 및 모듈화 리팩토링
- 개선: 
  1. CLAHE 적용으로 터널/조명 변화 대응
  2. Flood Fill을 이용한 동적 도로 마스킹 (Dynamic ROI)
  3. 코드 모듈화 (Segmentor, Visualizer, VO 분리)
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 설정 및 상수 (Configuration)
# ==========================================
class Config:
    VIDEO_PATH = "F1_Monaco.mp4"
    GT_MAP_PATH = "Circuit_de_Monaco.webp"
    OUTPUT_VIDEO = "F1_VO_FloodFill_Result.mp4"
    OUTPUT_RESULT_IMG = "F1_VO_Result_Plot.png"
    
    RESIZE_W = 640
    RESIZE_H = 360
    FOCAL = 700.0 
    PP = (RESIZE_W / 2, RESIZE_H / 2)
    
    # Flood Fill 설정
    FLOOD_LO_DIFF = (20, 20, 20) # 색상 하한 임계값
    FLOOD_UP_DIFF = (20, 20, 20) # 색상 상한 임계값

# ==========================================
# 2. 도로 인식 모듈 (Road Segmentation)
# ==========================================
class RoadSegmentor:
    def __init__(self, width, height):
        self.w = width
        self.h = height
        # 차량을 가리는 고정 마스크 (Static Mask) 미리 생성
        self.static_mask = self._create_car_mask()

    def _create_car_mask(self):
        """차량의 차체와 UI를 가리는 고정 마스크 생성 (0: 가림, 255: 보임)"""
        mask = np.full((self.h, self.w), 255, dtype=np.uint8)
        
        # 1. 차량 실루엣 (타이어+노즈)
        pts = np.array([
            [int(self.w * 0.12), self.h],             
            [int(self.w * 0.12), int(self.h * 0.40)],  
            [int(self.w * 0.32), int(self.h * 0.40)],  
            [int(self.w * 0.35), int(self.h * 0.50)], # 노즈 부분
            [int(self.w * 0.35), int(self.h * 0.48)],  
            [int(self.w * 0.65), int(self.h * 0.48)],  
            [int(self.w * 0.65), int(self.h * 0.50)],  
            [int(self.w * 0.68), int(self.h * 0.40)],  
            [int(self.w * 0.88), int(self.h * 0.40)],  
            [int(self.w * 0.88), self.h]              
        ], np.int32)
        cv2.fillPoly(mask, [pts], 0)
        
        # 2. 하단 UI 영역 마스킹 (높이 상향 조정됨)
        ui_h = int(self.h * 0.35) 
        cv2.rectangle(mask, (0, self.h - ui_h), (int(self.w * 0.12), self.h), 0, -1)
        cv2.rectangle(mask, (int(self.w * 0.88), self.h - ui_h), (self.w, self.h), 0, -1)
        
        return mask

    def apply_clahe(self, frame):
        """조명 변화 대응을 위한 전처리"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        # Contrast Limited Adaptive Histogram Equalization
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def get_adaptive_road_mask(self, frame):
        """Adaptive Flood Fill을 이용해 도로 영역 추출"""
        # 1. 조명 보정 (CLAHE)
        enhanced_frame = self.apply_clahe(frame)
        
        # 2. Flood Fill 준비
        # Seed Point: 차량 노즈 바로 위쪽 (도로가 확실한 영역)
        # 화면 중앙(0.5), 높이는 노즈 위쪽(약 0.45~0.48 지점)
        seed_point = (int(self.w * 0.5), int(self.h * 0.45))
        
        # FloodFill을 위한 마스크 (원본보다 h+2, w+2 커야 함)
        h, w = frame.shape[:2]
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        
        # 3. Flood Fill 실행
        # loDiff, upDiff: 색상 허용 범위 (아스팔트의 미세한 변화 수용)
        cv2.floodFill(enhanced_frame, flood_mask, seed_point, (255, 0, 0), 
                      Config.FLOOD_LO_DIFF, Config.FLOOD_UP_DIFF, 
                      flags=4 | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE)
        
        # 4. Flood Mask 크기 원복 및 고정 마스크와 결합
        road_mask = flood_mask[1:h+1, 1:w+1]
        
        # 최종 마스크 = (FloodFill로 찾은 도로) AND (차량이 아닌 영역)
        final_mask = cv2.bitwise_and(road_mask, self.static_mask)
        
        # 시각화용 디버그 이미지 (씨앗 점 표시)
        debug_img = enhanced_frame.copy()
        cv2.circle(debug_img, seed_point, 5, (0, 255, 0), -1) # 초록색 점이 씨앗 위치
        
        return final_mask, debug_img

# ==========================================
# 3. 시각화 모듈 (Visualization)
# ==========================================
class TrajectoryVisualizer:
    def __init__(self, width, height):
        self.w = width
        self.h = height
        
    def draw_trajectory(self, trajectory):
        map_img = np.zeros((self.h, self.h, 3), dtype=np.uint8)
        if len(trajectory) < 2: return map_img
        
        path = np.array(trajectory)
        
        # Bounding Box 계산 및 정규화
        min_x, max_x = np.min(path[:,0]), np.max(path[:,0])
        min_z, max_z = np.min(path[:,1]), np.max(path[:,1])
        
        range_max = max(max_x - min_x, max_z - min_z)
        scale_draw = (self.h * 0.8) / range_max if range_max > 0 else 1.0
        
        mid_x = (min_x + max_x) / 2
        mid_z = (min_z + max_z) / 2
        center_screen = self.h // 2
        
        # 그리기
        for i in range(1, len(path)):
            x1, z1 = path[i-1]
            x2, z2 = path[i]
            
            draw_x1 = int((x1 - mid_x) * scale_draw) + center_screen
            draw_y1 = int(-(z1 - mid_z) * scale_draw) + center_screen
            draw_x2 = int((x2 - mid_x) * scale_draw) + center_screen
            draw_y2 = int(-(z2 - mid_z) * scale_draw) + center_screen
            
            cv2.line(map_img, (draw_x1, draw_y1), (draw_x2, draw_y2), (0, 255, 255), 1)
        
        # 현재 위치
        cur_x, cur_z = path[-1]
        cx = int((cur_x - mid_x) * scale_draw) + center_screen
        cy = int(-(cur_z - mid_z) * scale_draw) + center_screen
        cv2.circle(map_img, (cx, cy), 3, (0, 0, 255), -1)
            
        return map_img

    def save_comparison(self, trajectory):
        if os.path.exists(Config.GT_MAP_PATH):
            gt_img = cv2.imread(Config.GT_MAP_PATH)
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
        else:
            gt_img = np.zeros((500, 500, 3), dtype=np.uint8)

        path = np.array(trajectory)
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
            plt.title("Generated Trajectory (Flood Fill)")
            plt.legend()
            plt.grid(True)
            plt.axis('equal')
        
        plt.savefig(Config.OUTPUT_RESULT_IMG)
        print(f"▶ 결과 이미지 저장됨: {Config.OUTPUT_RESULT_IMG}")

# ==========================================
# 4. VO 시스템 코어 (VO Logic)
# ==========================================
class MonocularVO:
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=3000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.segmentor = RoadSegmentor(Config.RESIZE_W, Config.RESIZE_H)
        self.visualizer = TrajectoryVisualizer(Config.RESIZE_W, Config.RESIZE_H)
        
        self.prev_gray = None
        self.prev_kp = None
        self.prev_des = None
        self.cur_t = np.array([[0], [0], [0]], dtype=np.float64)
        self.cur_R = np.eye(3, dtype=np.float64)
        self.trajectory = []

    def process_frame(self, frame):
        frame_resized = cv2.resize(frame, (Config.RESIZE_W, Config.RESIZE_H))
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        
        # [핵심] Adaptive Flood Fill로 동적 마스크 생성
        # road_mask: 도로 영역만 255, 나머지 0
        road_mask, debug_img = self.segmentor.get_adaptive_road_mask(frame_resized)
        
        # 마스크를 적용하여 특징점 추출 (도로 위 특징점만 사용)
        kp, des = self.orb.detectAndCompute(gray, road_mask)
        
        img_out = frame_resized.copy()
        
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_kp = kp
            self.prev_des = des
            return img_out, self.visualizer.draw_trajectory(self.trajectory)

        if des is not None and self.prev_des is not None:
            matches = self.bf.match(self.prev_des, des)
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = matches[:int(len(matches)*0.2)]
            
            if len(good_matches) >= 8:
                pts1 = np.float32([self.prev_kp[m.queryIdx].pt for m in good_matches])
                pts2 = np.float32([kp[m.trainIdx].pt for m in good_matches])
                
                E, mask_ransac = cv2.findEssentialMat(pts1, pts2, focal=Config.FOCAL, pp=Config.PP, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                
                if E is not None:
                    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, focal=Config.FOCAL, pp=Config.PP)
                    
                    if t[2] > 0.1: # 전진하는 경우만 업데이트
                        scale = 1.0 
                        self.cur_t = self.cur_t + scale * self.cur_R.dot(t)
                        self.cur_R = self.cur_R.dot(R)
                
                # 매칭 결과 시각화
                for i, m in enumerate(good_matches):
                    if i < 30:
                        pt1 = tuple(map(int, pts1[i]))
                        pt2 = tuple(map(int, pts2[i]))
                        cv2.line(img_out, pt1, pt2, (0, 255, 0), 1)

        # 디버깅: 도로 마스크 영역을 붉은색 틴트로 표시
        mask_overlay = cv2.cvtColor(road_mask, cv2.COLOR_GRAY2BGR)
        mask_overlay[:, :, 0] = 0 # Blue channel off
        mask_overlay[:, :, 1] = 0 # Green channel off
        # Red channel remains (Masked area will be red)
        
        # 도로 인식 영역 시각화
        img_out = cv2.addWeighted(img_out, 0.8, mask_overlay, 0.4, 0)
        
        # Seed Point 확인용 (녹색 점)
        cv2.circle(img_out, (int(Config.RESIZE_W * 0.5), int(Config.RESIZE_H * 0.45)), 3, (0, 255, 0), -1)

        self.trajectory.append((float(self.cur_t[0]), float(self.cur_t[2])))
        self.prev_gray = gray
        self.prev_kp = kp
        self.prev_des = des
        
        return img_out, self.visualizer.draw_trajectory(self.trajectory)

# ==========================================
# 5. 메인 실행 (Main Execution)
# ==========================================
def main():
    if not os.path.exists(Config.VIDEO_PATH):
        print(f"영상 파일 없음: {Config.VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(Config.VIDEO_PATH)
    if not cap.isOpened(): return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(Config.OUTPUT_VIDEO, fourcc, 30.0, (Config.RESIZE_W + Config.RESIZE_H, Config.RESIZE_H))
    
    vo = MonocularVO()
    
    print("▶ 궤적 생성 시작 (Adaptive Flood Fill 적용)...")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame_vis, map_vis = vo.process_frame(frame)
        combined = np.hstack((frame_vis, map_vis))
        
        cv2.imshow("VO System - Road Segmentation", combined)
        out.write(combined)
        
        if cv2.waitKey(1) == ord('q'): break
            
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    vo.visualizer.save_comparison(vo.trajectory)
    print("✅ 완료")

if __name__ == "__main__":
    main()