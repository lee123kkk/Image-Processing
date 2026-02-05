
'''
데모 구현: 자율주행 차량 위치 추정 (GNSS + IMU 퓨전 시뮬레이션)
이 데모는 자율주행 분야의 핵심인 "위치 추정"을 시뮬레이션합니다.

상황: 차량이 2차원 평면을 이동합니다.

센서 1 (IMU 역할): 차량의 속도와 방향을 알지만 시간이 지날수록 오차가 누적됩니다 (Process Model).

센서 2 (GPS 역할): 절대 좌표를 알려주지만 노이즈가 심합니다 (Measurement).

목표: 두 정보를 융합하여 실제 경로를 추적합니다.
'''



import numpy as np
import matplotlib.pyplot as plt
import cv2

# 1. 시뮬레이션 데이터 생성 (차량의 실제 이동 경로)
def generate_data(steps=200):
    # 참값 (Ground Truth): 원형으로 이동하는 차량
    t = np.linspace(0, 2*np.pi, steps)
    true_x = 100 * np.cos(t) + 200 # 중심 (200, 200), 반지름 100
    true_y = 100 * np.sin(t) + 200
    
    # GPS 측정값 (Measurement): 참값에 심한 노이즈 추가
    gps_noise_std = 15.0 # 노이즈가 꽤 큼
    obs_x = true_x + np.random.normal(0, gps_noise_std, steps)
    obs_y = true_y + np.random.normal(0, gps_noise_std, steps)
    
    return true_x, true_y, obs_x, obs_y

# 2. 칼만 필터 설정
# 상태 변수(4개): [x, y, vx, vy] (위치, 속도)
# 측정 변수(2개): [x, y] (GPS 좌표)
KF = cv2.KalmanFilter(4, 2, 0)

# A 행렬 (상태 전이): 등속도 모델
dt = 1.0 # 시간 간격
KF.transitionMatrix = np.array([[1, 0, dt, 0],
                                [0, 1, 0, dt],
                                [0, 0, 1,  0],
                                [0, 0, 0,  1]], np.float32)

# H 행렬 (측정): 위치(x, y)만 측정 가능
KF.measurementMatrix = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0]], np.float32)

# Q 행렬 (프로세스 노이즈): 차량의 급격한 가속/감속 가능성
# 이 값이 작으면 관성을 믿고(부드러움), 크면 측정을 믿음(반응 빠름)
KF.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2

# R 행렬 (측정 노이즈): GPS의 오차 정도
# 이 값이 크면 GPS를 덜 신뢰하고 예측(관성)을 더 믿음
KF.measurementNoiseCov = np.eye(2, dtype=np.float32) * 150.0 

# 초기값 설정
KF.errorCovPost = np.eye(4, dtype=np.float32) # 초기 오차
KF.statePost = np.array([[300], [200], [0], [0]], dtype=np.float32) # 시작점 근처

# 3. 시뮬레이션 실행 및 필터링
true_x, true_y, obs_x, obs_y = generate_data()
est_x = []
est_y = []

for i in range(len(true_x)):
    # 3-1. Predict (IMU/차량 모델에 해당)
    # 이전 속도를 기반으로 다음 위치 예측
    KF.predict()
    
    # 3-2. Correct (GPS 센서에 해당)
    # 노이즈가 섞인 GPS 좌표 입력
    z = np.array([[obs_x[i]], [obs_y[i]]], dtype=np.float32)
    estimate = KF.correct(z)
    
    est_x.append(estimate[0, 0])
    est_y.append(estimate[1, 0])

# 4. 결과 시각화
plt.figure(figsize=(10, 8))
plt.plot(true_x, true_y, 'g-', linewidth=2, label='Ground Truth (Real Path)')
plt.scatter(obs_x, obs_y, c='r', marker='x', s=20, label='GPS Measurement (Noisy)')
plt.plot(est_x, est_y, 'b-', linewidth=3, label='Kalman Filter Estimate')

plt.title("Autonomous Driving Localization (GPS + Model Fusion)")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

#===================================================================
# 데모 결과 분석 및 시사점
# 녹색 선 (Ground Truth): 차량이 실제로 주행한 매끄러운 원형 경로입니다.
# 빨간 X (GPS): 실제 경로 주변으로 심하게 흩뿌려져 있습니다. 
# 만약 이대로 자율주행을 한다면 차가 좌우로 심하게 흔들릴 것입니다.
# 파란 선 (Kalman Filter): 빨간 점들을 입력받았음에도 불구하고, 
# 녹색 선(실제 경로)과 매우 유사하게 부드러운 궤적을 그립니다.


# 자율 주행 및 로보틱스 분야에서 칼만 필터는 신뢰도가 낮고 노이즈가 심한 센서를 
# 물리적 운동 모델과 결합하여 실제 위치에 가장 가까운 최적의 값을 실시간으로 추정하는 핵심 알고리즘이다.
