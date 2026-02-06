'''
[프로젝트: 지능형 적응형 VO 시스템 V6]
- 기능: Adaptive Flood Fill 기반 도로 인식 + VO 융합 + 방향 시각화
- 개선: 
  1. 차량 진행 방향(Heading) 화살표 시각화 복구
  2. 도로 영역의 무게중심(Centroid)을 이용한 이동 벡터 보정 (Drift Correction)
  3. 도로 인식 가중치 강화 (Road Guidance)
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
    OUTPUT_VIDEO = "F1_VO_Fusion_Result.mp4"
    OUTPUT_RESULT_IMG = "F1_VO_Fusion_Plot.png"
    
    RESIZE_W = 640
    RESIZE_H = 360
    FOCAL = 700.0 
    PP = (RESIZE_W / 2, RESIZE_H / 2)
    
    # Flood Fill 설정
    FLOOD_LO_DIFF = (20, 20, 20)
    FLOOD_UP_DIFF = (20, 20, 20)
    
    # 융합 가중치 (VO vs Road Guidance)
    WEIGHT_VO = 0.4
    WEIGHT_ROAD = 0.6  # 도로 인식을 더 신뢰

# ==========================================
# 2. 도로 인식 모듈 (Road Segmentor)
# ==========================================
class RoadSegmentor:
    def __init__(self, width, height):
        self.w = width
        self.h = height
        self.static_mask = self._create_car_mask()

    def _create_car_mask(self):
        mask = np.full((self.h, self.w), 255, dtype=np.uint8)
        pts = np.array([
            [int(self.w * 0.12), self.h],             
            [int(self.w * 0.12), int(self.h * 0.40)],  
            [int(self.w * 0.32), int(self.h * 0.40)],  
            [int(self.w * 0.35), int(self.h * 0.50)], 
            [int(self.w * 0.35), int(self.h * 0.48)],  
            [int(self.w * 0.65), int(self.h * 0.48)],  
            [int(self.w * 0.65), int(self.h * 0.50)],  
            [int(self.w * 0.68), int(self.h * 0.40)],  
            [int(self.w * 0.88), int(self.h * 0.40)],  
            [int(self.w * 0.88), self.h]              
        ], np.int32)
        cv2.fillPoly(mask, [pts], 0)
        
        ui_h = int(self.h * 0.35) 
        cv2.rectangle(mask, (0, self.h - ui_h), (int(self.w * 0.12), self.h), 0, -1)
        cv2.rectangle(mask, (int(self.w * 0.88), self.h - ui_h), (self.w, self.h), 0, -1)
        return mask

    def apply_clahe(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def get_adaptive_road_mask(self, frame):
        enhanced_frame = self.apply_clahe(frame)
        seed_point = (int(self.w * 0.5), int(self.h * 0.45))
        
        h, w = frame.shape[:2]
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        
        cv2.floodFill(enhanced_frame, flood_mask, seed_point, (255, 0, 0), 
                      Config.FLOOD_LO_DIFF, Config.FLOOD_UP_DIFF, 
                      flags=4 | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE)
        
        road_mask = flood_mask[1:h+1, 1:w+1]
        final_mask = cv2.bitwise_and(road_mask, self.static_mask)
        
        return final_mask, enhanced_frame

    def get_road_centroid(self, mask):
        """도로 영역의 무게중심(Centroid) 계산"""
        moments = cv2.moments(mask, binaryImage=True)
        if moments["m00"] > 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            return (cx, cy)
        return None

# ==========================================
# 3. 시각화 모듈 (Visualizer)
# ==========================================
class TrajectoryVisualizer:
    def __init__(self, width, height):
        self.w = width
        self.h = height
        
    def draw_trajectory(self, trajectory, current_R):
        map_img = np.zeros((self.h, self.h, 3), dtype=np.uint8)
        if len(trajectory) < 2: return map_img
        
        path = np.array(trajectory)
        min_x, max_x = np.min(path[:,0]), np.max(path[:,0])
        min_z, max_z = np.min(path[:,1]), np.max(path[:,1])
        
        range_max = max(max_x - min_x, max_z - min_z)
        scale_draw = (self.h * 0.8) / range_max if range_max > 0 else 1.0
        
        mid_x = (min_x + max_x) / 2
        mid_z = (min_z + max_z) / 2
        center_screen = self.h // 2
        
        # 1. 궤적 선 그리기
        for i in range(1, len(path)):
            x1, z1 = path[i-1]
            x2, z2 = path[i]
            
            draw_x1 = int((x1 - mid_x) * scale_draw) + center_screen
            draw_y1 = int(-(z1 - mid_z) * scale_draw) + center_screen
            draw_x2 = int((x2 - mid_x) * scale_draw) + center_screen
            draw_y2 = int(-(z2 - mid_z) * scale_draw) + center_screen
            
            cv2.line(map_img, (draw_x1, draw_y1), (draw_x2, draw_y2), (0, 255, 255), 1)
        
        # 2. 현재 위치 및 방향(Heading) 그리기
        cur_x, cur_z = path[-1]
        cx = int((cur_x - mid_x) * scale_draw) + center_screen
        cy = int(-(cur_z - mid_z) * scale_draw) + center_screen
        
        # 회전 행렬에서 방향 벡터 추출 (Z축 방향)
        # R * [0, 0, 1]^T -> [r13, r23, r33]^T
        dir_x = current_R[0, 2]
        dir_z = current_R[2, 2]
        
        # 지도상에서의 방향 (Z축이 위쪽이므로 부호 주의)
        arrow_len = 20
        end_x = int(cx + dir_x * arrow_len)
        end_y = int(cy - dir_z * arrow_len) # 화면 좌표계는 y가 아래로 증가하므로 -
        
        cv2.arrowedLine(map_img, (cx, cy), (end_x, end_y), (0, 0, 255), 2, tipLength=0.3)
        cv2.circle(map_img, (cx, cy), 3, (0, 255, 0), -1)
            
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
            plt.title("Fusion Trajectory (VO + Road)")
            plt.grid(True)
            plt.axis('equal')
        
        plt.savefig(Config.OUTPUT_RESULT_IMG)
        print(f"▶ 결과 이미지 저장됨: {Config.OUTPUT_RESULT_IMG}")

# ==========================================
# 4. VO 시스템 코어 (Fusion Logic)
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
        
        # 1. 도로 인식
        road_mask, enhanced_img = self.segmentor.get_adaptive_road_mask(frame_resized)
        
        # 2. 특징점 추출 (도로 마스크 적용)
        kp, des = self.orb.detectAndCompute(gray, road_mask)
        img_out = frame_resized.copy()
        
        # 초기화 단계
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_kp = kp
            self.prev_des = des
            return img_out, self.visualizer.draw_trajectory(self.trajectory, self.cur_R)

        # 3. VO 계산 및 융합
        if des is not None and self.prev_des is not None:
            matches = self.bf.match(self.prev_des, des)
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = matches[:int(len(matches)*0.2)]
            
            if len(good_matches) >= 8:
                pts1 = np.float32([self.prev_kp[m.queryIdx].pt for m in good_matches])
                pts2 = np.float32([kp[m.trainIdx].pt for m in good_matches])
                
                E, _ = cv2.findEssentialMat(pts1, pts2, focal=Config.FOCAL, pp=Config.PP, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                
                if E is not None:
                    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, focal=Config.FOCAL, pp=Config.PP)
                    
                    if t[2] > 0.1: # 전진 중일 때만 업데이트
                        # [Fusion Logic] 도로 중심 정보를 이용한 회전 보정
                        centroid = self.segmentor.get_road_centroid(road_mask)
                        road_steering_val = 0.0
                        
                        if centroid is not None:
                            # 화면 중앙(cx)과 도로 중심(centroid_x)의 차이 계산
                            # 차이가 크면 도로가 휘어지고 있다는 뜻
                            image_center_x = Config.RESIZE_W / 2
                            deviation = (centroid[0] - image_center_x) / (Config.RESIZE_W / 2) # -1.0 ~ 1.0 정규화
                            
                            # 도로 곡률에 따른 회전 보정값 (단순화된 모델)
                            # 도로 중심이 오른쪽(>0)이면 우회전 필요
                            road_steering_val = deviation * 0.05 
                        
                        # VO에서 계산된 회전각 (Y축 회전, Yaw) 추출 근사
                        # R[0,2] 값은 대략적인 Sin(Yaw) 값과 유사
                        vo_yaw = t[0] # 평행이동 벡터의 x성분도 회전 성향을 반영함
                        
                        # [핵심] VO 값과 도로 인식 값의 가중 합
                        # t[0] (횡이동)를 보정하여 궤적을 부드럽게 만듦
                        # 순수 VO는 노이즈가 많으므로 도로 인식 결과로 damping
                        t[0] = (t[0] * Config.WEIGHT_VO) + (road_steering_val * Config.WEIGHT_ROAD)
                        
                        scale = 1.0 
                        self.cur_t = self.cur_t + scale * self.cur_R.dot(t)
                        self.cur_R = self.cur_R.dot(R)
                
                # 매칭 시각화
                for i, m in enumerate(good_matches):
                    if i < 30:
                        pt1 = tuple(map(int, pts1[i]))
                        pt2 = tuple(map(int, pts2[i]))
                        cv2.line(img_out, pt1, pt2, (0, 255, 0), 1)

        # 디버깅: 도로 마스크 및 중심점 시각화
        mask_overlay = cv2.cvtColor(road_mask, cv2.COLOR_GRAY2BGR)
        mask_overlay[:, :, 0] = 0 
        mask_overlay[:, :, 1] = 0 
        img_out = cv2.addWeighted(img_out, 0.8, mask_overlay, 0.4, 0)
        
        centroid = self.segmentor.get_road_centroid(road_mask)
        if centroid:
            cv2.circle(img_out, centroid, 8, (0, 255, 255), -1) # 노란색 점: 도로 중심
            cv2.line(img_out, (int(Config.RESIZE_W/2), int(Config.RESIZE_H)), centroid, (255, 0, 0), 2) # 파란선: 차량->도로중심

        self.trajectory.append((float(self.cur_t[0]), float(self.cur_t[2])))
        self.prev_gray = gray
        self.prev_kp = kp
        self.prev_des = des
        
        return img_out, self.visualizer.draw_trajectory(self.trajectory, self.cur_R)

# ==========================================
# 5. 메인 실행
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
    
    print("▶ 궤적 생성 시작 (VO + Road Fusion)...")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame_vis, map_vis = vo.process_frame(frame)
        combined = np.hstack((frame_vis, map_vis))
        
        cv2.imshow("VO System - Fusion", combined)
        out.write(combined)
        
        if cv2.waitKey(1) == ord('q'): break
            
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    vo.visualizer.save_comparison(vo.trajectory)
    print("✅ 완료")

if __name__ == "__main__":
    main()