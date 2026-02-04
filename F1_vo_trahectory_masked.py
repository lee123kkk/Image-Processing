'''
[프로젝트: 지능형 적응형 VO 시스템]
주행 궤적(Trajectory) 생성 및 마스킹(Masking) 적용 코드

개선 사항:
1. Self-Feature Tracking 방지: 자동차 본체 및 자막(UI) 영역 마스킹 적용
2. 전진 필터링: Z축(전진) 이동량이 양수일 때만 궤적 업데이트
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
OUTPUT_VIDEO = "F1_Trajectory_Result_Masked.mp4"
OUTPUT_RESULT_IMG = "F1_Trajectory_Comparison_Masked.png"

RESIZE_W = 640
RESIZE_H = 360
# 초점 거리(Focal Length)는 대략적인 값입니다. 
# 실제 트랙 비율과 맞지 않으면 이 값을 600~800 사이로 조절해 볼 수 있습니다.
FOCAL = 700.0 
PP = (RESIZE_W / 2, RESIZE_H / 2)

class MonocularVO:
    def __init__(self):
        # ORB 특징점 추출기 생성
        self.orb = cv2.ORB_create(nfeatures=3000)
        # Hamming Distance를 사용하는 Brute-Force Matcher
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        self.prev_gray = None
        self.prev_kp = None
        self.prev_des = None
        
        # 초기 위치 (0,0,0) 및 회전 행렬 (Identity)
        self.cur_t = np.array([[0], [0], [0]], dtype=np.float64)
        self.cur_R = np.eye(3, dtype=np.float64)
        self.trajectory = []

    def get_mask(self, height, width):
        """
        특징점 추출에서 제외할 영역(자동차 본체, UI 등)을 설정하는 마스크 생성 함수
        return: 0(제외)과 255(포함)로 구성된 이미지
        """
        # 1. 기본적으로 모든 영역을 추적 대상(White, 255)으로 설정
        mask = np.full((height, width), 255, dtype=np.uint8)
        
        # 2. 마스킹 영역 설정 (이미지 좌표계: 좌상단이 0,0)
        
        # (A) 하단 영역 통째로 날리기 (자동차 본체 + 바퀴 + 서스펜션)
        # 화면의 하단 40% 정도가 차체라고 가정
        car_hood_y = int(height * 0.60) 
        cv2.rectangle(mask, (0, car_hood_y), (width, height), 0, -1)
        
        # (B) 좌측 하단 자막(UI) 영역 제거
        # 속도계/기어 표시 등이 있는 좌측 하단 박스
        ui_h = int(height * 0.25) # 밑에서부터 25% 높이
        ui_w = int(width * 0.35)  # 왼쪽에서 35% 너비
        cv2.rectangle(mask, (0, height - ui_h), (ui_w, height), 0, -1)

        return mask

    def process_frame(self, frame):
        frame_resized = cv2.resize(frame, (RESIZE_W, RESIZE_H))
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        
        # [핵심] 마스크 생성 및 적용
        mask = self.get_mask(RESIZE_H, RESIZE_W)
        
        # mask 파라미터를 사용하여 해당 영역(검은색)에서는 특징점을 찾지 않음
        kp, des = self.orb.detectAndCompute(gray, mask)
        
        img_out = frame_resized.copy()
        
        # 첫 프레임 처리
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_kp = kp
            self.prev_des = des
            return img_out, self.get_trajectory_map()

        # 이전 프레임과 매칭 진행
        if des is not None and self.prev_des is not None:
            matches = self.bf.match(self.prev_des, des)
            matches = sorted(matches, key=lambda x: x.distance)
            
            # 상위 20%의 좋은 매칭점만 사용
            good_matches = matches[:int(len(matches)*0.2)]
            
            if len(good_matches) >= 8:
                pts1 = np.float32([self.prev_kp[m.queryIdx].pt for m in good_matches])
                pts2 = np.float32([kp[m.trainIdx].pt for m in good_matches])
                
                # Essential Matrix 계산 (RANSAC으로 이상치 제거)
                E, mask_ransac = cv2.findEssentialMat(pts1, pts2, focal=FOCAL, pp=PP, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                
                if E is not None:
                    # 포즈 복원 (R, t 추출)
                    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, focal=FOCAL, pp=PP)
                    
                    # [핵심] 전진 방향(Z축) 필터링
                    # 차는 앞으로만 달리므로, Z축 이동량이 양수일 때만 궤적에 반영
                    # 값이 너무 작으면(0.1 미만) 정지해 있거나 노이즈로 간주
                    if t[2] > 0.1: 
                        scale = 1.0 # 현재는 속도 정보가 없으므로 1.0으로 고정 (Scale Ambiguity)
                        self.cur_t = self.cur_t + scale * self.cur_R.dot(t)
                        self.cur_R = self.cur_R.dot(R)
                
                # 시각화: 매칭된 점 그리기 (마스킹 된 영역에는 점이 없어야 함)
                for i, m in enumerate(good_matches):
                    if i < 50: # 화면이 너무 복잡해지지 않게 50개만 그리기
                        pt1 = tuple(map(int, pts1[i]))
                        pt2 = tuple(map(int, pts2[i]))
                        cv2.line(img_out, pt1, pt2, (0, 255, 0), 1)
                        cv2.circle(img_out, pt2, 3, (0, 0, 255), -1)

        # [시각화 디버깅] 마스크 영역을 어둡게 표시하여 제대로 가려졌는지 확인
        # 마스크가 0인 부분(차체, 자막)을 원본 이미지에서 어둡게 만듦
        mask_overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) # 3채널로 변환
        # 마스크가 0인 부분은 검은색이 되고, 255인 부분은 원본 색상 유지
        img_masked_debug = cv2.bitwise_and(img_out, mask_overlay)
        # 시각적으로 "여기는 무시했다"는 느낌을 주기 위해 약간 어둡게 합성
        img_out = cv2.addWeighted(img_out, 0.3, img_masked_debug, 0.7, 0)

        # 궤적 데이터 저장 (XZ 평면)
        self.trajectory.append((float(self.cur_t[0]), float(self.cur_t[2])))
        
        # 현재 데이터를 다음 프레임의 '이전 데이터'로 저장
        self.prev_gray = gray
        self.prev_kp = kp
        self.prev_des = des
        
        return img_out, self.get_trajectory_map()

    def get_trajectory_map(self):
        """계산된 궤적을 2D 이미지로 그려주는 함수"""
        map_img = np.zeros((RESIZE_H, RESIZE_H, 3), dtype=np.uint8)
        if len(self.trajectory) < 2: return map_img
        
        scale_draw = 1.0
        center_x, center_y = RESIZE_H // 2, RESIZE_H // 2
        path = np.array(self.trajectory)
        
        # 궤적의 전체 크기에 맞춰 화면에 꽉 차게 그리기 위한 스케일링
        if len(path) > 10:
            min_x, max_x = np.min(path[:,0]), np.max(path[:,0])
            min_z, max_z = np.min(path[:,1]), np.max(path[:,1])
            range_max = max(max_x - min_x, max_z - min_z)
            if range_max > 0: scale_draw = (RESIZE_H * 0.8) / range_max
        
        # 궤적 그리기
        for i in range(1, len(path)):
            x1, z1 = path[i-1]
            x2, z2 = path[i]
            # 좌표 변환: 시작점을 중심으로 이동시키고 스케일 적용
            draw_x1 = int((x1 - path[0][0]) * scale_draw) + center_x + 100
            draw_y1 = int(-(z1 - path[0][1]) * scale_draw) + center_y + 100 # Z축을 Y축으로 매핑 (부호 반전)
            draw_x2 = int((x2 - path[0][0]) * scale_draw) + center_x + 100
            draw_y2 = int(-(z2 - path[0][1]) * scale_draw) + center_y + 100
            
            cv2.line(map_img, (draw_x1, draw_y1), (draw_x2, draw_y2), (0, 255, 255), 1)
            
        return map_img

def save_comparison_result(trajectory_points):
    """최종 결과 그래프를 저장하고 보여주는 함수"""
    if os.path.exists(GT_MAP_PATH):
        gt_img = cv2.imread(GT_MAP_PATH)
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
    else:
        gt_img = np.zeros((500, 500, 3), dtype=np.uint8)

    path = np.array(trajectory_points)
    plt.figure(figsize=(12, 6))
    
    # 왼쪽: 실제 지도 (Ground Truth)
    plt.subplot(1, 2, 1)
    plt.imshow(gt_img)
    plt.title("Ground Truth (Circuit de Monaco)")
    plt.axis('off')
    
    # 오른쪽: VO로 생성된 궤적
    plt.subplot(1, 2, 2)
    if len(path) > 0:
        # X, Z 좌표를 플로팅 (Z는 위쪽이 진행 방향이 되도록 부호 반전하여 시각화)
        plt.plot(path[:, 0], -path[:, 1], color='blue', linewidth=2, label='VO Path')
        plt.scatter(path[0, 0], -path[0, 1], c='green', label='Start')
        plt.scatter(path[-1, 0], -path[-1, 1], c='red', marker='x', label='End')
        plt.title("Generated Trajectory (Masked)")
        plt.legend()
        plt.grid(True)
        plt.axis('equal') # X, Y 축 비율을 1:1로 고정하여 왜곡 방지
    
    plt.savefig(OUTPUT_RESULT_IMG)
    plt.show()

def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"오류: '{VIDEO_PATH}' 영상 파일을 찾을 수 없습니다.")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened(): 
        print("영상을 열 수 없습니다.")
        return

    # 코덱 설정
    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        print("▶ H.264(avc1) 코덱으로 저장을 시도합니다.")
    except:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        print("▶ avc1 실패. mp4v로 대체합니다.")

    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 30.0, (RESIZE_W + RESIZE_H, RESIZE_H))
    vo = MonocularVO()
    
    print(f"▶ 궤적 생성 시작... (영상 길이: {RESIZE_W}x{RESIZE_H})")
    print("▶ 'q' 키를 누르면 중간에 종료할 수 있습니다.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 프레임 처리 및 궤적 맵 생성
        frame_vis, map_vis = vo.process_frame(frame)
        
        # 영상(왼쪽)과 궤적 맵(오른쪽)을 합쳐서 보여줌
        combined = np.hstack((frame_vis, map_vis))
        
        cv2.imshow("VO System - Masking Applied", combined)
        out.write(combined)
        
        if cv2.waitKey(1) == ord('q'): break
            
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"✅ 처리 완료! 결과 영상: {OUTPUT_VIDEO}")
    print(f"✅ 결과 그래프 저장: {OUTPUT_RESULT_IMG}")
    
    save_comparison_result(vo.trajectory)

if __name__ == "__main__":
    main()