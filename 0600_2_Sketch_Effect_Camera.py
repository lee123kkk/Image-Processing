'''
2. 스케치 효과 카메라
카메라 영상을 스케치한 그림처럼 보여주는 프로그램. 한 영상에 스케치한 영상과 물감까지 칠한 영상이 모두 나오게 처리
Hint:스케치 영상을 만들려면 그레이 스케일로 바꾸엇 엣지를 얻어야 함. 엣지를 얻기 위해서는 cv2.Laplacian() 함수를 쓰는 것이 좋은데, 그
전에 잡음을 없애야 해서 cv2.GauusianBlur() 함수를 먼저 쓰는 것이 좋음. 엣지를 얻은 후, 스레시홀드로 경계선 이외의 것들은 제거하고 반전
하면 흰 도화지에 검은 펜으로 스케치한 효과를 얻을 수 있음. 선이 너무 흐리면 모폴로지 팽창 연산으로 강조해 주면 스케치 영상이 완성
물감 그림 영상은 컬러 영상을 흐릿하게 만들어서 스케치 영상과 cv2.bitwise_and() 함수로 함성하면 완성. 컬러 영상을 흐릿하게 할 때, 평균
블러 cv2.blur()를 사용하면 좋음
'''
import cv2
import numpy as np

# 카메라 열기
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera open failed!")
    exit()

print("Sketch Effect Camera Started.")
print("Keys: [q] Quit, [Space] Toggle Mode")

# 모폴로지 연산 커널 (검은 선을 더 진하게 연결하기 위함)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 연산 속도 향상 및 잡음 감소를 위한 리사이즈 (선택 사항)
    # frame = cv2.resize(frame, None, fx=0.75, fy=0.75)

    # ==========================================================
    # 1. 스케치 영상 만들기 (흰 배경 + 검은 선)
    # ==========================================================
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 1-1. 잡음 제거 (중요: 잡티가 검은 점으로 찍히는 것 방지)
    blur_gray = cv2.medianBlur(gray, 7)
    
    # 1-2. 적응형 이진화 (Adaptive Threshold) - 여기가 핵심!
    # - cv2.THRESH_BINARY: 배경을 흰색(255), 선을 검은색(0)으로 만듭니다. (INV 아님!)
    # - blockSize=9: 숫자가 작을수록(9, 11) 얇은 선, 클수록(21, 31) 뭉툭한 선
    # - C=5: 이 숫자가 작으면(2~3) 선이 많아지고(지저분), 크면(10~) 선이 줄어듭니다(깨끗).
    sketch = cv2.adaptiveThreshold(blur_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY, blockSize=11, C=5)
    
    # 1-3. 검은 선 강조 (모폴로지 침식 대신 '열기' 연산 추천)
    # 검은색 점(노이즈)은 없애고, 검은색 선은 유지/강조
    # (배경이 흰색일 때는 Erode를 하면 검은 선이 두꺼워집니다)
    sketch = cv2.erode(sketch, kernel, iterations=1)


    # ==========================================================
    # 2. 물감 채색 영상 만들기
    # ==========================================================
    # 양방향 필터로 경계선은 살리고 색깔 뭉개기
    paint = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)


    # ==========================================================
    # 3. 합성 (흰 배경 스케치 + 컬러 영상)
    # ==========================================================
    # 스케치는 1채널이므로 3채널로 변환
    sketch_bgr = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    
    # [합성 원리]
    # bitwise_and를 하면:
    # - 스케치의 흰색(255) 부분 -> paint의 색상이 그대로 나옴
    # - 스케치의 검은색(0) 부분 -> 검은색 선으로 덮어씌워짐
    result = cv2.bitwise_and(paint, sketch_bgr)

    # ==========================================================
    # 결과 출력
    # ==========================================================
    combined = np.hstack((sketch_bgr, result))
    cv2.imshow('Sketch Effect Camera', combined)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




'''
1차 시도 (초기 코드)
결과: 화면이 전체적으로 하얗게 날아가서 윤곽선이 거의 보이지 않음.

원인: 고정된 임계값(threshold=80)을 사용하여, 조명 변화나 약한 엣지 신호를 제대로 구분하지 못하고 모두 배경(흰색)으로 처리함.

해결: 주변 밝기에 따라 유동적으로 반응하는 cv2.adaptiveThreshold(적응형 이진화) 함수 도입.
'''

'''
2차 시도 (중간 수정)
결과: 화면이 온통 검은색이고, 희미한 흰 선만 나타나는 반전 현상 발생.

원인: cv2.THRESH_BINARY_INV(반전) 옵션을 사용하는 바람에 의도와 다르게 '검은 배경에 흰 선'이 출력됨.

해결: 옵션을 cv2.THRESH_BINARY로 변경하여 '흰 배경에 검은 선'을 확보하고, 잡음 제거를 위해 medianBlur 적용.
'''


'''
3차 시도 (최종 완성)
결과: 흰 도화지 위 검은 펜 터치 느낌의 스케치와 부드러운 수채화 채색 효과 구현 성공.

원인: (성공 요인) 배경색 옵션 정상화 및 양방향 필터(bilateralFilter)를 통해 경계선은 살리고 색감은 자연스럽게 뭉갬.

해결: adaptiveThreshold의 파라미터(BlockSize, C)를 조절하여 선의 굵기와 선명도를 최적화함.
'''


