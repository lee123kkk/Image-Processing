
'''
주행 궤적(Trajectory)을 그려 실제 서킷 지도와 비교하는 코드

속도 보정(Scale Correction)을 전혀 하지 않았을 때
단안 카메라(Monocular Camera) VO가 어떤 한계를 가지는지(모양이 어떻게 찌그러지는지)를 
명확하게 보여주기 위해 설계
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
OUTPUT_VIDEO = "F1_Trajectory_Result_H264.mp4" # 파일명 변경
OUTPUT_RESULT_IMG = "F1_Trajectory_Comparison.png"

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

    def process_frame(self, frame):
        frame_resized = cv2.resize(frame, (RESIZE_W, RESIZE_H))
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)
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
                
                E, mask = cv2.findEssentialMat(pts1, pts2, focal=FOCAL, pp=PP, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                
                if E is not None:
                    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, focal=FOCAL, pp=PP)
                    scale = 1.0 
                    if scale > 0.1: 
                        self.cur_t = self.cur_t + scale * self.cur_R.dot(t)
                        self.cur_R = self.cur_R.dot(R)
                
                for i, m in enumerate(good_matches):
                    if i < 50:
                        pt1 = tuple(map(int, pts1[i]))
                        pt2 = tuple(map(int, pts2[i]))
                        cv2.line(img_out, pt1, pt2, (0, 255, 0), 1)
                        cv2.circle(img_out, pt2, 3, (0, 0, 255), -1)

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
        plt.title("Generated Trajectory (H.264)")
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

    # [핵심 수정] 코덱을 avc1 (H.264)으로 변경
    # 만약 avc1이 설치 안 된 PC라면 에러가 날 수 있어 try-except 처리
    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        print("▶ H.264(avc1) 코덱으로 설정을 시도합니다.")
    except:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        print("▶ avc1 실패. mp4v로 대체합니다 (업로드 시 호환성 문제 가능성 있음).")

    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 30.0, (RESIZE_W + RESIZE_H, RESIZE_H))
    vo = MonocularVO()
    
    print("▶ 궤적 생성 시작... ('q'로 종료)")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_vis, map_vis = vo.process_frame(frame)
        combined = np.hstack((frame_vis, map_vis))
        cv2.imshow("VO Test", combined)
        out.write(combined)
        if cv2.waitKey(1) == ord('q'): break
            
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"✅ 동영상 저장 완료: {OUTPUT_VIDEO} (호환성 개선됨)")
    save_comparison_result(vo.trajectory)

if __name__ == "__main__":
    main()