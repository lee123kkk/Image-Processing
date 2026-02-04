import math  
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import cv2


# 관심 영역을 설정하는 함수
def region_of_interest(img, vertices):
    # 이미지와 동일한 크기의 빈 마스크 생성
    mask = np.zeros_like(img)
    
    # 채널 수에 따라 마스크 색상 설정 (그레이스케일 이미지의 경우 255)
    match_mask_color = 255
    
    # 다각형 내부를 마스크 색상으로 채움
    cv2.fillPoly(mask, vertices, match_mask_color)
    
    # 마스크와 원본 이미지를 비트 연산하여 관심 영역만 추출
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# 감지된 선을 이미지에 그리는 함수
def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    # 원본 이미지와 동일한 크기의 빈 이미지 생성
    line_img = np.zeros(
        (
            img.shape[0],
            img.shape[1],
            3
        ),
        dtype=np.uint8
    )
    
    img = np.copy(img)
    
    # 감지된 선이 없으면 원본 이미지 반환
    if lines is None:
        return img
    
    # 모든 선에 대해 반복하며 빈 이미지에 그리기
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
            
    # 선이 그려진 이미지와 원본 이미지를 합침
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    return img

# 전체 이미지 처리 파이프라인
def pipeline(image):
    """
    차선이 표시된 이미지를 출력하는 이미지 처리 파이프라인
    """
    height = image.shape[0]
    width = image.shape[1]
    
    # 관심 영역의 꼭짓점 정의 (삼각형 모양)
    region_of_interest_vertices = [
        (0, height),
        (width / 2, height / 2),
        (width, height),
    ]
    
    # 그레이스케일 변환
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Canny 에지 검출
    cannyed_image = cv2.Canny(gray_image, 100, 200)
 
    # 관심 영역 적용
    cropped_image = region_of_interest(
        cannyed_image,
        np.array(
            [region_of_interest_vertices],
            np.int32
        ),
    )
 
    # 허프 변환을 사용한 선 검출
    lines = cv2.HoughLinesP(
        cropped_image,
        rho=6,
        theta=np.pi / 60,
        threshold=160,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=25
    )
 
    # 왼쪽과 오른쪽 차선 선분을 저장할 리스트
    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []
 
    # 감지된 모든 선분에 대해 반복
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                # 기울기 계산
                slope = (y2 - y1) / (x2 - x1)
                # 수평에 가까운 선은 제외
                if math.fabs(slope) < 0.5:
                    continue
                # 기울기에 따라 왼쪽/오른쪽 차선으로 분류
                if slope <= 0:
                    left_line_x.extend([x1, x2])
                    left_line_y.extend([y1, y2])
                else:
                    right_line_x.extend([x1, x2])
                    right_line_y.extend([y1, y2])
 
    # 차선을 그릴 y 좌표 범위 설정
    min_y = int(image.shape[0] * (3 / 5))
    max_y = int(image.shape[0])
 
    # 왼쪽 차선에 대한 선형 회귀 및 x 좌표 계산
    if left_line_x and left_line_y:
        poly_left = np.poly1d(np.polyfit(
            left_line_y,
            left_line_x,
            deg=1
        ))
        left_x_start = int(poly_left(max_y))
        left_x_end = int(poly_left(min_y))
    else:
        left_x_start, left_x_end = 0, 0

    # 오른쪽 차선에 대한 선형 회귀 및 x 좌표 계산
    if right_line_x and right_line_y:
        poly_right = np.poly1d(np.polyfit(
            right_line_y,
            right_line_x,
            deg=1
        ))
        right_x_start = int(poly_right(max_y))
        right_x_end = int(poly_right(min_y))
    else:
        right_x_start, right_x_end = 0, 0
 
    # 최종 차선 그리기
    line_image = draw_lines(
        image,
        [[
            [left_x_start, max_y, left_x_end, min_y],
            [right_x_start, max_y, right_x_end, min_y],
        ]],
        thickness=5,
    )
    return line_image

# 1. 현재 폴더에 있는 모든 파일 목록을 가져옵니다.
files = os.listdir(os.getcwd())

# 2. 파일들 중에서 이미지 파일(.jpg, .png)만 골라냅니다.
image_files = [f for f in files if f.endswith('.jpg') or f.endswith('.png')]

# 3. 찾은 이미지들을 하나씩 꺼내서 순서대로 처리합니다.
print(f"총 {len(image_files)}개의 이미지 파일을 찾았습니다: {image_files}")

for filename in image_files:
    # 이미지 읽기
    image = cv2.imread(filename)
    
    if image is None:
        continue # 이미지가 안 읽히면 건너뜀

    # 파이프라인 실행
    # (주의: pipeline 함수 내부의 색상 변환을 cv2.COLOR_BGR2GRAY로 수정했는지 확인하세요. 
    # 수정하지 않아도 작동은 하지만, 수정하는 것이 더 정확합니다.)
    result = pipeline(image)

    # 결과 창 띄우기
    cv2.imshow('Lane Detection Result', result)
    
    print(f"현재 표시 중: {filename} (아무 키나 누르면 다음 사진으로 넘어갑니다)")
    
    # 4. 아무 키나 누를 때까지 대기 (키를 누르면 다음 루프로 넘어감)
    cv2.waitKey(0) 

# 모든 처리가 끝나면 창 닫기
cv2.destroyAllWindows()
print("모든 이미지 처리가 완료되었습니다.")