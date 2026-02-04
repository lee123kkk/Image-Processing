import os
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# --- [전역 변수] 곡선 계수(a, b, c)를 기억하기 위한 변수 ---
# 2차 함수식: x = ay^2 + by + c
# 이전 프레임의 계수 [a, b, c] 3개를 저장합니다.
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
    기존 cv2.line(직선) 대신 cv2.polylines(다각형/곡선)를 사용합니다.
    2차 함수 계수(fit)를 받아서 y값에 따른 x좌표를 계산해 점들을 찍고 이어 그립니다.
    """
    line_img = np.zeros_like(img)
    
    # y좌표 생성 (화면 위쪽부터 아래쪽까지)
    plot_y = np.linspace(0, img.shape[0]-1, img.shape[0])
    
    # 왼쪽 차선 그리기
    if left_fit is not None:
        # 2차 함수 공식: x = a*y^2 + b*y + c
        left_fit_x = left_fit[0]*plot_y**2 + left_fit[1]*plot_y + left_fit[2]
        
        # 좌표들을 정수형 배열로 변환 (cv2.polylines 입력 형식 맞춤)
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
    img_result = cv2.addWeighted(img, 1.0, line_img, 0.8, 0.0)
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
    
    # 1. 전처리 (HSV 필터링 등은 생략하고 기본 구조 유지)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cannyed_image = cv2.Canny(gray_image, 100, 200)
 
    cropped_image = region_of_interest(
        cannyed_image,
        np.array([region_of_interest_vertices], np.int32),
    )
 
    # 2. 허프 변환 (곡선 인식을 위해 파라미터가 조금 더 유연했음)
    lines = cv2.HoughLinesP(
        cropped_image,
        rho=6,
        theta=np.pi / 60,
        threshold=130,      
        lines=np.array([]),
        minLineLength=40,   
        maxLineGap=150      
    )
 
    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []
 
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y2 - y1) / (x2 - x1)
                
                # 곡선 구간 감지를 위해 기울기 제한을 낮춤
                if math.fabs(slope) < 0.3: 
                    continue
                    
                if slope <= 0:
                    left_line_x.extend([x1, x2])
                    left_line_y.extend([y1, y2])
                else:
                    right_line_x.extend([x1, x2])
                    right_line_y.extend([y1, y2])
 
    # --- [핵심 변경: 2차 다항식 맞춤 (Polynomial Fit)] ---
    
    # 1. 현재 프레임의 2차 함수 계수 계산 (deg=2)
    # x = ay^2 + by + c 형태이므로 (y, x) 순서로 넣어줍니다.
    curr_left_fit = None
    if left_line_x and left_line_y:
        try:
            # 여기서 2를 넣는 것이 핵심 (2차 함수)
            curr_left_fit = np.polyfit(left_line_y, left_line_x, 2)
        except: pass 

    curr_right_fit = None
    if right_line_x and right_line_y:
        try:
            curr_right_fit = np.polyfit(right_line_y, right_line_x, 2)
        except: pass

    # 2. 이동 평균 (계수 a, b, c 각각을 평균냄)
    alpha = 0.1 

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

    # 3. 곡선 그리기 함수 호출
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