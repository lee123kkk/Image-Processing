'''
Mouse Tracking (마우스 추적):

방금 전 실습했던 코드와 거의 유사합니다. 
OpenCV의 kalman.cpp나 파이썬 포팅 버전들이 주를 이루며, 
"예측(Predict)과 보정(Correct)의 순환 구조"를 시각적으로 가장 잘 보여주는 기초 예제입니다.

Lane Detection (차선 검출 보정):

맥락: 자율주행 차가 달릴 때, 카메라 진동이나 조명 변화로 인해 영상처리(Hough Transform 등)로 
검출된 차선이 매 프레임마다 파들파들 떨리거나(Jittering), 잠시 사라지는 현상이 발생합니다.

역할: 칼만 필터를 사용하여 차선의 기울기(Slope)와 절편(Intercept)을 추적함으로써, 
차선을 부드럽게 유지하고 잠시 감지되지 않아도 이전 경로를 예측해 그려줍니다.
'''

'''
칼만 필터를 이용한 차선 흔들림 보정 (Lane Stabilizer)
이 코드는 영상처리 단계(Canny, Hough)를 거쳐 나온 "노이즈가 섞인 차선 데이터"를 가정하고, 
칼만 필터가 이를 어떻게 부드러운 주행 선으로 바꾸는지 시뮬레이션합니다.
'''


import cv2
import numpy as np
import matplotlib.pyplot as plt

class LaneKalmanFilter:
    def __init__(self):
        # 1. 칼만 필터 생성
        # 상태 변수(4개): [기울기(m), 절편(b), 기울기변화량(dm), 절편변화량(db)]
        # 측정 변수(2개): [기울기(m), 절편(b)] (영상처리로 얻은 값)
        self.kf = cv2.KalmanFilter(4, 2, 0)

        # 2. 전이 행렬 (A) - 등속도 모델
        # m_new = m_old + dm, b_new = b_old + db
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32)

        # 3. 측정 행렬 (H)
        # 우리는 m과 b만 측정함
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], np.float32)

        # 4. 노이즈 공분산 설정
        # Q (Process Noise): 차선이 갑자기 꺾이지 않는다고 가정 (작은 값)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-4
        # R (Measurement Noise): 영상처리는 꽤 불안정하다고 가정 (큰 값)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        
        # 초기값 에러 공분산
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

    def predict(self):
        # 다음 프레임의 차선 위치 예측
        return self.kf.predict()

    def correct(self, m, b):
        # 실제 검출된 차선(m, b)으로 보정
        measurement = np.array([[np.float32(m)], [np.float32(b)]])
        return self.kf.correct(measurement)

# ==========================================
# 시뮬레이션 시작
# ==========================================

# 1. 가상의 차선 데이터 생성 (직진 주행 상황)
frames = 100
true_slope = 0.5  # 실제 차선 기울기
true_intercept = 100 # 실제 차선 위치

# 노이즈가 섞인 측정값 생성 (카메라 진동 등)
# 실제로는 cv2.HoughLinesP() 등의 결과값이 됩니다.
noisy_slopes = true_slope + np.random.normal(0, 0.1, frames)
noisy_intercepts = true_intercept + np.random.normal(0, 5, frames)

# 2. 칼만 필터 적용
lane_kf = LaneKalmanFilter()
filtered_slopes = []
filtered_intercepts = []

# 초기 상태 설정 (첫 프레임 값으로 초기화)
lane_kf.kf.statePost = np.array([[noisy_slopes[0]], [noisy_intercepts[0]], [0], [0]], dtype=np.float32)

for i in range(frames):
    # A. 예측 (Predict)
    lane_kf.predict()
    
    # B. 보정 (Correct) - 노이즈 섞인 측정값 입력
    estimate = lane_kf.correct(noisy_slopes[i], noisy_intercepts[i])
    
    # 결과 저장 (기울기 m, 절편 b)
    filtered_slopes.append(estimate[0, 0])
    filtered_intercepts.append(estimate[1, 0])

# ==========================================
# 결과 시각화
# ==========================================
plt.figure(figsize=(12, 5))

# 기울기(Slope) 비교
plt.subplot(1, 2, 1)
plt.plot(noisy_slopes, 'r-', label='Detected (Noisy)', alpha=0.5)
plt.plot(filtered_slopes, 'b-', label='Kalman Smoothed', linewidth=2)
plt.axhline(true_slope, color='g', linestyle='--', label='Ground Truth')
plt.title("Lane Slope Stabilization")
plt.xlabel("Frame")
plt.ylabel("Slope (m)")
plt.legend()

# 절편(Intercept) 비교
plt.subplot(1, 2, 2)
plt.plot(noisy_intercepts, 'r-', label='Detected (Noisy)', alpha=0.5)
plt.plot(filtered_intercepts, 'b-', label='Kalman Smoothed', linewidth=2)
plt.axhline(true_intercept, color='g', linestyle='--', label='Ground Truth')
plt.title("Lane Position Stabilization")
plt.xlabel("Frame")
plt.ylabel("Intercept (b)")
plt.legend()

plt.tight_layout()
plt.show()

'''
상태 변수 (x): [m, b, dm, db]단순히 차선의 위치(m, b)뿐만 아니라, 
차선이 변해가는 속도(dm, db)까지 추적합니다. 
이는 곡선 도로에 진입할 때 차선이 서서히 휘어지는 것을 자연스럽게 따라가게 합니다.
측정 노이즈 (R): 0.1 (비교적 큼)Hough Transform 등으로 얻은 차선 좌표가 튀는 현상(Jitter)을 "노이즈"로 간주하고 무시하겠다는 의도입니다.
효과:그래프를 보면 빨간색 선(검출값)은 위아래로 심하게 튀지만, 파란색 선(칼만 필터)은 중앙을 부드럽게 가로지릅니다.
실제 자율주행차에서는 이 '파란색 값'을 사용하여 핸들을 조향하므로, 승차감이 훨씬 부드러워집니다.
'''

# 차선검출에서 칼만 필터는 영상 처리의 jitter를 평활화하여 노이즈가 영항을 주지 않도록 하는 안정화장치 역할을 수행한다.
