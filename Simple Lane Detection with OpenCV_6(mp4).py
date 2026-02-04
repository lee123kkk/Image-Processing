import os
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# --- [전역 변수] 이동 평균(Smoothing)을 위한 메모리 ---
# 매 프레임 독립적으로 계산하면 차선이 떨리므로, 이전 프레임의 위치를 기억해둡니다.
prev_left_line = None
prev_right_line = None

def region_of_interest(img, vertices):
    """
    [관심 영역(ROI) 설정]
    도로 영상에서 하늘, 배경 등 불필요한 부분을 제외하고
    차선이 존재할 확률이 높은 영역(삼각형)만 남깁니다.
    """
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[0, 255, 0], thickness=8):
    """
    [차선 그리기]
    계산된 좌표를 바탕으로 원본 이미지 위에 선을 그립니다.
    - color=[0, 255, 0]: 네온 그린 색상으로 가시성을 높임
    - addWeighted: 원본 영상과 투명하게 합성
    """
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    
    img_copy = np.copy(img)
    
    if lines is None:
        return img_copy
    
    for line in lines:
        if line is None: continue
        x1, y1, x2, y2 = line
        cv2.line(line_img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
            
    img_copy = cv2.addWeighted(img_copy, 0.8, line_img, 1.0, 0.0)
    return img_copy

def pipeline(image):
    # 전역 변수를 함수 내부로 가져옴 (이전 프레임 값 사용)
    global prev_left_line, prev_right_line

    height = image.shape[0]
    width = image.shape[1]
    
    region_of_interest_vertices = [
        (int(width * 0.1), height),       # 0 -> width * 0.1 (왼쪽 구석 제외)
        (width / 2, height / 2),
        (int(width * 0.9), height),       # width -> width * 0.9 (오른쪽 구석 제외)
    ]
    
    # [수정됨] RGB -> HSV 색상 필터링 적용
    # 기존: 단순히 흑백으로 변환 (그림자 구분이 안 됨)
    # 변경: HSV 색상 공간에서 노란색과 흰색만 따로 추출한 뒤 합침
    
    # 1. BGR(OpenCV 기본) -> HSV 변환
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 2. 흰색 차선 범위 정의 (채도가 낮고 밝기가 매우 높은 색)
    # Sensitivity(민감도)를 조절하려면 이 숫자들을 변경하면 됩니다.
    lower_white = np.array([0, 0, 200])      # H, S, V 최소값
    upper_white = np.array([180, 30, 255])   # H, S, V 최대값

    # 3. 노란색 차선 범위 정의 (Hue값이 노란색 영역인 것)
    lower_yellow = np.array([15, 100, 100])  # 약간 어두운 노랑도 포함
    upper_yellow = np.array([35, 255, 255])  # 밝은 노랑

    # 4. 마스크 생성 (해당 색상 범위에 맞는 픽셀만 1, 나머지는 0)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # 5. 두 마스크 합치기 (흰색 이거나 OR 노란색 인 영역)
    combined_mask = cv2.bitwise_or(mask_white, mask_yellow)

    # 6. 원본 이미지에서 마스크 된 부분만 추출
    masked_image = cv2.bitwise_and(image, image, mask=combined_mask)

    # 7. 그레이스케일 변환 (Canny Edge를 위해)
    # 이제 배경은 검은색이고 차선만 색깔이 남아있으므로 엣지 검출이 훨씬 잘 됩니다.
    gray_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    
    # 2. [Canny Edge Detection] 엣지 검출
    # 픽셀의 밝기가 급격하게 변하는 부분(경계선)을 찾습니다.
    # 100(min), 200(max): 임계값. 이 범위를 벗어나는 흐릿한 선은 무시합니다.
    cannyed_image = cv2.Canny(gray_image, 100, 200)
 
    # ROI 적용: 엣지 검출 후 도로 영역만 잘라냄
    cropped_image = region_of_interest(
        cannyed_image,
        np.array([region_of_interest_vertices], np.int32),
    )
 
    # 3. [Hough Transform] 허프 변환 직선 검출
    # 점(엣지)들이 모여 직선을 이루는 패턴을 수학적으로 찾아냅니다.
    # - minLineLength=40: 잡음(짧은 선)을 무시하기 위해 최소 길이를 설정
    # - maxLineGap=150: 점선 차선처럼 끊겨 있는 선들을 하나의 선으로 연결
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
 
    # 4. 차선 분류 (기울기 기반)
    # 검출된 수많은 선분들을 기울기(slope)를 기준으로 왼쪽/오른쪽 차선으로 나눕니다.
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y2 - y1) / (x2 - x1)
                
                # [기울기 필터링] 
                # 수평에 가까운 선(가드레일, 그림자 등)을 제거합니다.
                # 0.3 미만은 무시하여 회전 구간 인식률을 높입니다.
                if math.fabs(slope) < 0.3: 
                    continue
                    
                # 기울기가 음수면 왼쪽 차선, 양수면 오른쪽 차선
                if slope <= 0:
                    left_line_x.extend([x1, x2])
                    left_line_y.extend([y1, y2])
                else:
                    right_line_x.extend([x1, x2])
                    right_line_y.extend([y1, y2])
 
    min_y = int(image.shape[0] * 0.65)
    max_y = int(image.shape[0])
 
    # --- [스무딩 & 이동 평균 (Smoothing & Moving Average)] ---
    
    # 5. 선형 회귀 (Linear Regression)
    # 흩어진 점들을 대표하는 하나의 직선 방정식(y = mx + b)을 구합니다.
    curr_left = None
    if left_line_x and left_line_y:
        poly_left = np.poly1d(np.polyfit(left_line_y, left_line_x, deg=1))
        curr_left = [int(poly_left(max_y)), max_y, int(poly_left(min_y)), min_y]

    curr_right = None
    if right_line_x and right_line_y:
        poly_right = np.poly1d(np.polyfit(right_line_y, right_line_x, deg=1))
        curr_right = [int(poly_right(max_y)), max_y, int(poly_right(min_y)), min_y]

    # 6. 이동 평균 적용 (핵심 알고리즘)
    # 현재 프레임 값만 쓰면 차선이 떨리므로, 이전 프레임 값과 가중 평균을 냅니다.
    # Formula: 결과 = (이전값 * 0.9) + (현재값 * 0.1)
    # alpha 값이 작을수록 이전 프레임의 영향을 많이 받아 움직임이 부드러워집니다.
    alpha = 0.1

    # 왼쪽 차선 업데이트
    if curr_left is not None:
        if prev_left_line is None:
            prev_left_line = curr_left # 첫 프레임이면 현재 값 그대로 사용
        else:
            prev_left_line = [int(prev * (1-alpha) + curr * alpha) 
                              for prev, curr in zip(prev_left_line, curr_left)]
    
    # 오른쪽 차선 업데이트
    if curr_right is not None:
        if prev_right_line is None:
            prev_right_line = curr_right
        else:
            prev_right_line = [int(prev * (1-alpha) + curr * alpha) 
                               for prev, curr in zip(prev_right_line, curr_right)]

    # 최종적으로 그릴 선들을 리스트로 정리
    final_lines = []
    if prev_left_line is not None: final_lines.append(prev_left_line)
    if prev_right_line is not None: final_lines.append(prev_right_line)

    line_image = draw_lines(image, final_lines, thickness=8)
    return line_image

# --- 메인 실행 코드 ---
files = os.listdir(os.getcwd())
video_files = [f for f in files if f.endswith('.mp4')]

print(f"재생할 비디오 목록: {video_files}")

for filename in video_files:
    print(f"--- 현재 재생 중: {filename} ---")
    
    # 비디오가 바뀔 때마다 이전 기록 초기화 (안 그러면 이전 비디오의 차선 위치가 남음)
    prev_left_line = None
    prev_right_line = None
    
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
            cv2.imshow('Multi-Video Lane Detection', result)
        except Exception as e:
            print(f"처리 중 오류: {e}")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("다음 비디오로 넘어갑니다.")
            break 

    cap.release()

cv2.destroyAllWindows()
print("모든 비디오 재생이 끝났습니다.")
#======================<수정 가능한 주요 파라미터>=======================
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


#======================<수정된 파라미터 적용>==========================
#  alpha (스무딩 강도) 
#  현재 설정값 : 0.1  <-- 0.15에서 낮춤 (떨림 방지)
  
#  maxLineGap (선 연결 거리)
#  현재 설정값 : 150  <-- 100에서 높임 (그림자 구간 연결)

#  minLineLength (최소 선 길이)
#  현재 설정값 : 40   <-- 60에서 낮춤 (곡선 조각 감지)

#  slope (기울기 제한)
#  현재 설정값 : 0.3  <-- 0.4에서 낮춤 (급커브 감지 필수!)

#  threshold
#  현재 설정값 : 130  <-- 160에서 낮춤 (그림자 속 희미한 선 감지)
#===================================================================


#======================<추가 수정된 파라미터 적용>=====================
#  alpha (스무딩 강도) 
#  현재 설정값 : 0.05  <-- 0.1에서 더 낮춤 (잡음이 늘어난 만큼 더 묵직하게 이동)
  
#  maxLineGap (선 연결 거리)
#  현재 설정값 : 150   <-- (유지) 끊긴 선 연결에 좋음

#  minLineLength (최소 선 길이)
#  현재 설정값 : 20    <-- 40에서 낮춤 (곡선의 짧은 마디 감지)

#  slope (기울기 제한)
#  현재 설정값 : 0.1   <-- 0.3에서 낮춤 (거의 누워있는 커브 끝자락 감지)

#  threshold
#  현재 설정값 : 70    <-- 130에서 대폭 낮춤 (그림자 속 어두운 노란 선 감지)
#===================================================================


#======================<추가 수정된 파라미터 적용>=====================
#                     BGR 컬러에서 HSV 컬러로 변경
#                        이전 파라미터 값 복원         

#  alpha (스무딩 강도) 
#  현재 설정값 : 0.1  
  
#  maxLineGap (선 연결 거리)
#  현재 설정값 : 150  

#  minLineLength (최소 선 길이)
#  현재 설정값 : 40   

#  slope (기울기 제한)
#  현재 설정값 : 0.3  

#  threshold
#  현재 설정값 : 130  
#===================================================================



#======================<추가 수정된 파라미터 적용>=====================
# 1. ROI 영역 (관심 영역) 수정 
#    - 문제: 너무 넓은 영역을 보느라 갓길의 연석이나 풀숲을 차선으로 착각함
#    - 해결: 아래쪽 양 끝을 잘라내어 시야를 도로 안쪽으로 좁힘
#    - 적용법: pipeline 함수 내 region_of_interest_vertices 부분 수정
#    (0, height)          -> (width * 0.1, height) 로 변경 (왼쪽 10% 자름)
#    (width, height)      -> (width * 0.9, height) 로 변경 (오른쪽 10% 자름)

# 2. min_y (차선 그리기 길이 제한) 수정
#    - 문제: 곡선 도로에서 직선을 길게 그리면, 끝부분이 휘어지지 못해 차선을 뚫고 나감
#    - 해결: 차선을 그리는 높이를 낮춰서(더 짧게 그려서) 삐져나가기 전에 멈추게 함
#    - 적용법: pipeline 함수 내 min_y 계산식 수정
#    int(image.shape[0] * (3 / 5)) -> int(image.shape[0] * 0.7) 로 변경
#    (0.6 지점까지 그리던 것을 0.7 지점까지만 그리게 하여 선 길이를 줄임)

#===================================================================


#======================<추가 수정된 파라미터 적용>=====================
# 1. ROI 영역 (관심 영역) 수정 

#    (0, height)          -> (width * 0.1, height) 로 변경 (왼쪽 10% 자름)
#    (width, height)      -> (width * 0.9, height) 로 변경 (오른쪽 10% 자름)

# 2. min_y (차선 그리기 길이 제한) 수정

#    int(image.shape[0] * (3 / 5)) -> int(image.shape[0] * 0.65) 로 변경

#===================================================================
