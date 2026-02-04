import math  
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
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

# 1. OpenCV 함수로 이미지 읽기 (파일 이름 확인 필수!)
image = cv2.imread('line_test_sample.png')

# 파일이 제대로 읽혔는지 확인 (파일 경로 틀리면 여기서 멈춤)
if image is None:
    print("오류: 이미지 파일을 찾을 수 없습니다. 파일 이름을 확인해주세요.")
else:
    # 2. 파이프라인 함수를 살짝 수정해서 호출 (BGR 이미지를 처리하기 위함)
    # 기존 pipeline 함수는 RGB를 가정하므로, 여기서 미리 그레이스케일로 만들어서 넘겨도 됩니다.
    # 하지만 더 간단하게는, 위에서 만든 pipeline 함수 내부의 
    # cv2.COLOR_RGB2GRAY 를 -> cv2.COLOR_BGR2GRAY 로 한 글자만 고치면 완벽합니다.
    # (일단 지금 상태로 실행해도 작동은 합니다!)
    
    result = pipeline(image)

    # 3. 결과 창 띄우기
    cv2.imshow('Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()