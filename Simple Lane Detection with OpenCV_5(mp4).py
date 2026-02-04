import os
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# --- [전역 변수] 곡선 계수(a, b, c)를 기억하기 위한 변수 ---
# 2차 함수식: x = ay^2 + by + c
# 이전 프레임의 [a, b, c] 값을 저장합니다.
prev_left_fit = None
prev_right_fit = None

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_curve(img, left_fit, right_fit, color=[0, 255, 0], thickness=8):
    """
    [곡선 그리기 함수]
    2차 함수 계수(fit)를 받아서 실제 화면에 그릴 좌표점들을 생성하고 곡선을 그립니다.
    """
    line_img = np.zeros_like(img)
    
    # y좌표 생성 (화면 전체 높이)
    plot_y = np.linspace(0, img.shape[0]-1, img.shape[0])
    
    # 왼쪽 차선 그리기
    if left_fit is not None:
        # 2차 함수식: x = ay^2 + by + c
        left_fit_x = left_fit[0]*plot_y**2 + left_fit[1]*plot_y + left_fit[2]
        
        # 좌표들을 (N, 1, 2) 형태의 정수형 배열로 변환 (cv2.polylines용)
        pts_left = np.array([np.transpose(np.vstack([left_fit_x, plot_y]))])
        pts_left = pts_left.astype(np.int32)
        
        # 곡선 그리기
        cv2.polylines(line_img, pts_left, False, color, thickness)

    # 오른쪽 차선 그리기
    if right_fit is not None:
        right_fit_x = right_fit[0]*plot_y**2 + right_fit[1]*plot_y + right_fit[2]
        pts_right = np.array([np.transpose(np.vstack([right_fit_x, plot_y]))])
        pts_right = pts_right.astype(np.int32)
        cv2.polylines(line_img, pts_right, False, color, thickness)

    # 원본과 합성
    img_result = cv2.addWeighted(img, 1.0, line_img, 0.8, 0.0) # 선을 좀 더 투명하게(0.8)
    return img_result

def pipeline(image):
    global prev_left_fit, prev_right_fit

    height = image.shape[0]
    width = image.shape[1]
    
    region_of_interest_vertices = [
        (0, height),
        (width / 2, height / 2),
        (width, height),
    ]
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cannyed_image = cv2.Canny(gray_image, 100, 200)
 
    cropped_image = region_of_interest(
        cannyed_image,
        np.array([region_of_interest_vertices], np.int32),
    )
 
    # [조건 유지] 허프 변환 사용
    # 파라미터는 튜닝된 값을 적용했습니다.
    lines = cv2.HoughLinesP(
        cropped_image,
        rho=6,
        theta=np.pi / 60,
        threshold=130,      # 160 -> 130 (그림자 대응)
        lines=np.array([]),
        minLineLength=40,   # 60 -> 40 (곡선 대응)
        maxLineGap=150      # 100 -> 150 (끊김 연결)
    )
 
    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []
 
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y2 - y1) / (x2 - x1)
                
                # 기울기 제한 (0.3으로 완화하여 급커브 감지)
                if math.fabs(slope) < 0.3: 
                    continue
                    
                if slope <= 0:
                    left_line_x.extend([x1, x2])
                    left_line_y.extend([y1, y2])
                else:
                    right_line_x.extend([x1, x2])
                    right_line_y.extend([y1, y2])
 
    # --- [2차 다항식 맞춤 & 스무딩] ---
    
    # 1. 현재 프레임의 2차 함수 계수 계산 (deg=2)
    # x = ay^2 + by + c 형태이므로 (y, x) 순서로 넣어줍니다.
    curr_left_fit = None
    if left_line_x and left_line_y:
        try:
            curr_left_fit = np.polyfit(left_line_y, left_line_x, 2)
        except: pass # 계산 실패 시 무시

    curr_right_fit = None
    if right_line_x and right_line_y:
        try:
            curr_right_fit = np.polyfit(right_line_y, right_line_x, 2)
        except: pass

    # 2. 이동 평균 (계수 자체를 평균냄)
    alpha = 0.1 # 스무딩 강도 (낮을수록 부드러움)

    # 왼쪽 업데이트
    if curr_left_fit is not None:
        if prev_left_fit is None:
            prev_left_fit = curr_left_fit
        else:
            prev_left_fit = prev_left_fit * (1-alpha) + curr_left_fit * alpha
    
    # 오른쪽 업데이트
    if curr_right_fit is not None:
        if prev_right_fit is None:
            prev_right_fit = curr_right_fit
        else:
            prev_right_fit = prev_right_fit * (1-alpha) + curr_right_fit * alpha

    # 3. 곡선 그리기
    # 현재 계산된(스무딩된) 계수를 넘겨줍니다.
    img_out = draw_curve(image, prev_left_fit, prev_right_fit, thickness=8)
    
    return img_out

# --- 메인 실행 코드 ---
files = os.listdir(os.getcwd())
video_files = [f for f in files if f.endswith('.mp4')]

print(f"재생할 비디오 목록: {video_files}")

for filename in video_files:
    print(f"--- 현재 재생 중: {filename} ---")
    
    prev_left_fit = None
    prev_right_fit = None
    
    cap = cv2.VideoCapture(filename)

    if not cap.isOpened():
        print(f"오류: {filename}을 열 수 없습니다.")
        continue

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"{filename} 재생 완료.")
            break

        try:
            result = pipeline(frame)
            cv2.imshow('Curved Lane Detection', result)
        except Exception as e:
            print(f"처리 중 오류: {e}")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("다음 비디오로 넘어갑니다.")
            break 

    cap.release()

cv2.destroyAllWindows()
print("모든 비디오 재생이 끝났습니다.")

#======================<기존 수정 가능한 주요 파라미터>===========================
#  alpha (스무딩 강도) 
#  현재 설정값 : 0.15
  
#  maxLineGap(선 연결 거리)
#  현재 설정값 : 100

#  minLineLength(최소 선 길이)
#  현재 설정값 : 60

#  slope(기울기 제한)
#  현재 설정값 : 0.4

#  threshold
#  현재 설정값 : 160
#===================================================================

#======================<새로 추가/수정된 파라미터 및 설명>==========================
#  np.polyfit(..., deg=2) [알고리즘 변경]
#  기존 1차(직선)에서 2차(곡선)로 변경되었습니다. 
#  이제 직선이 아닌 '포물선(y=ax^2+bx+c)' 형태로 차선을 그립니다. 
#  회전 구간에서 직선 막대기가 아닌 휘어지는 선으로 표현됩니다.

#  alpha (스무딩 강도) -> 0.1 [값 변경]
#  곡선은 직선보다 모양이 예민하게 변하므로, 값을 낮춰(0.1) 더 부드럽게 움직이도록 했습니다.

#  slope (기울기 제한) -> 0.3 [값 변경]
#  회전할 때 차선이 눕는 것을 감안하여 제한을 0.3으로 낮췄습니다. 

#  threshold (허프 민감도) -> 130 [값 변경]
#  나무 그림자 등에 가려진 희미한 선도 찾을 수 있도록 민감도를 낮췄습니다. (투표수 130)

#  minLineLength (최소 길이) -> 40 [값 변경]
#  곡선은 짧은 직선들의 집합으로 인식되므로, 짧은 선도 차선 후보로 인정하도록 값을 줄였습니다.
#===================================================================