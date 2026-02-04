'''
6가지 알고리즘(ORB, BRISK, AKAZE, KAZE, SIFT, SURF)을 순차적으로 수행하고, 
실행 화면 시각화와 최종적인 성능 비교 그래프(FPS, Inlier 개수)를 출력

'''


import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import os
from datetime import datetime

# ==========================================
# 설정 (Configuration)
# ==========================================
VIDEO_PATH = "F1_Monaco.mp4"
RESIZE_WIDTH = 640  # 리사이징 너비
SKIP_FRAMES = 0     # 초반 건너뛰기
MAX_FRAMES = 10000  # [수정됨] 전체 영상을 돌리기 위해 넉넉하게 설정 (약 10,000 프레임)

# 비교할 알고리즘 목록
ALGOS = ["ORB", "BRISK", "AKAZE", "KAZE", "SIFT", "SURF"]

# 그래프 색상 팔레트
COLORS = {
    "ORB": "red", "BRISK": "orange", "AKAZE": "green",
    "KAZE": "blue", "SIFT": "purple", "SURF": "brown"
}

# ==========================================
# 1. 알고리즘 초기화 팩토리
# ==========================================
def create_detector(name):
    name = name.upper()
    try:
        if name == "ORB":
            return cv2.ORB_create(nfeatures=2000), cv2.NORM_HAMMING
        elif name == "BRISK":
            return cv2.BRISK_create(), cv2.NORM_HAMMING
        elif name == "AKAZE":
            return cv2.AKAZE_create(), cv2.NORM_HAMMING
        elif name == "KAZE":
            return cv2.KAZE_create(), cv2.NORM_L2
        elif name == "SIFT":
            return cv2.SIFT_create(), cv2.NORM_L2
        elif name == "SURF":
            return cv2.xfeatures2d.SURF_create(hessianThreshold=400), cv2.NORM_L2
    except AttributeError:
        print(f"[Warn] {name} 알고리즘을 사용할 수 없습니다 (OpenCV 버전/설치 확인).")
        return None, None
    return None, None

# ==========================================
# 2. 메인 벤치마크 로직
# ==========================================
def run_benchmark():
    # 결과 저장용 딕셔너리
    benchmark_data = {algo: {"fps": [], "inliers": []} for algo in ALGOS}
    
    for algo_name in ALGOS:
        print(f"\n=========================================")
        print(f"▶ [{algo_name}] 벤치마크 시작...")
        
        detector, norm_type = create_detector(algo_name)
        if detector is None:
            continue

        matcher = cv2.BFMatcher(norm_type, crossCheck=False)
        
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            print(f"[Error] 영상을 찾을 수 없습니다: {VIDEO_PATH}")
            return

        cap.set(cv2.CAP_PROP_POS_FRAMES, SKIP_FRAMES)

        prev_gray = None
        prev_kp = None
        prev_des = None
        
        frame_idx = 0
        
        while frame_idx < MAX_FRAMES:
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            scale = RESIZE_WIDTH / w
            frame_resized = cv2.resize(frame, (int(w * scale), int(h * scale)))
            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

            try:
                kp, des = detector.detectAndCompute(gray, None)
            except Exception as e:
                print(f"[Error] {algo_name} 처리 중 오류: {e}")
                break

            inlier_count = 0
            matches_vis = frame_resized.copy()

            # 매칭 로직 (이전 프레임 vs 현재 프레임)
            if prev_des is not None and des is not None and len(prev_des) > 2 and len(des) > 2:
                matches = matcher.knnMatch(prev_des, des, k=2)
                
                good_matches = []
                for m_n in matches:
                    if len(m_n) == 2:
                        m, n = m_n
                        if m.distance < 0.75 * n.distance:
                            good_matches.append(m)

                if len(good_matches) >= 4:
                    src_pts = np.float32([prev_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    if mask is not None:
                        inlier_count = np.sum(mask)
                        matches_mask = mask.ravel().tolist()
                        draw_params = dict(matchColor=(0, 255, 0),
                                           singlePointColor=None,
                                           matchesMask=matches_mask,
                                           flags=2)
                        matches_vis = cv2.drawMatches(prev_gray, prev_kp, gray, kp, good_matches, None, **draw_params)
                    else:
                        matches_vis = frame_resized
                else:
                    pass

            process_time = time.time() - loop_start
            fps = 1.0 / process_time if process_time > 0 else 0

            benchmark_data[algo_name]["fps"].append(fps)
            benchmark_data[algo_name]["inliers"].append(inlier_count)

            info_text = f"Algo: {algo_name} | Frame: {frame_idx} | FPS: {fps:.1f} | Inliers: {inlier_count}"
            cv2.putText(matches_vis, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("Algorithm Benchmark", matches_vis)
            
            prev_gray = gray
            prev_kp = kp
            prev_des = des
            frame_idx += 1

            key = cv2.waitKey(1)
            if key == ord('q'):
                print("사용자 종료")
                cap.release()
                cv2.destroyAllWindows()
                return
            if key == ord('n'):
                print("다음 알고리즘으로 넘어갑니다.")
                break

        cap.release()

    cv2.destroyAllWindows()
    print("\n✅ 모든 벤치마크 완료! 결과 저장 중...")
    plot_and_save_results(benchmark_data)

# ==========================================
# 3. 결과 그래프 출력 및 저장 함수 (수정됨)
# ==========================================
def plot_and_save_results(data):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    max_len = 0
    for algo in data:
        max_len = max(max_len, len(data[algo]["inliers"]))
    
    for algo, metrics in data.items():
        if not metrics["inliers"]: continue
        
        y_inliers = metrics["inliers"]
        y_fps = metrics["fps"]
        curr_len = len(y_inliers)
        
        ax1.plot(range(curr_len), y_inliers, label=algo, color=COLORS.get(algo, "black"), alpha=0.8)
        ax2.plot(range(curr_len), y_fps, label=algo, color=COLORS.get(algo, "black"), alpha=0.8)

    ax1.set_title("Robustness Comparison: Number of Inliers (Higher is Better)")
    ax1.set_ylabel("Inlier Count (RANSAC passed)")
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    ax2.set_title("Speed Comparison: FPS (Higher is Better)")
    ax2.set_ylabel("Frames Per Second")
    ax2.set_xlabel("Frame Number")
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()

    # [수정된 부분] 파일 저장 로직
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"F1_Benchmark_Result_{timestamp}.png"
    plt.savefig(filename, dpi=300) # dpi=300으로 고화질 저장
    print(f"✅ 그래프가 저장되었습니다: {os.path.abspath(filename)}")
    
    plt.show()

if __name__ == "__main__":
    run_benchmark()

'''
[실험 결론: 고속 주행 환경에서의 Visual Odometry 알고리즘 선정]

극한 환경 한계점 도출: 
Frame 1280(터널 탈출) 구간에서 발생한 과노출(Over-exposure)로 인해 
모든 알고리즘의 추적 기능이 소실(Tracking Lost)됨을 확인했습니다. 
특히 BRISK는 특징점 검출 실패로 인해 연산 루프가 비정상 종료되는 현상(FPS 500+ 스파이크)이 관찰되었습니다.

ORB의 재발견: 
ORB는 Inlier의 노이즈(Jitter)는 심했으나, 
Tracking Lost 발생 후 가장 빠른 회복 속도(Re-localization)를 보였으며 
평균 50FPS를 유지했습니다.

최종 제안: 
안정적인 주행을 위해서는 평상시 ORB를 사용하되, 
터널 탈출과 같은 조도 급변 구간에서는 IMU(관성센서) 데이터를 융합하거나, 
Inlier가 100개 미만으로 떨어질 때 'Wait & Catch' 상태로 전환하는 예외처리 로직이 필수적입니다.
'''    