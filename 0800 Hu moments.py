#0800 Hu moments 기반 분류기



import cv2
import numpy as np
import math

# --- 전역 변수 ---
drawing = False
ix, iy = -1, -1
roi_coords = None

def mouse_callback(event, x, y, flags, param):
    global ix, iy, drawing, roi_coords
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, x2 = min(ix, x), max(ix, x)
        y1, y2 = min(iy, y), max(iy, y)
        if (x2 - x1) > 10 and (y2 - y1) > 10:
            roi_coords = (x1, y1, x2, y2)
        else:
            roi_coords = None

def get_hu_moments(contour):
    moments = cv2.moments(contour)
    if moments['m00'] == 0: return None
    hu_moments = cv2.HuMoments(moments)
    for i in range(7):
        if hu_moments[i] == 0: hu_moments[i] = 0
        else: hu_moments[i] = -1 * math.copysign(1.0, hu_moments[i]) * math.log10(abs(hu_moments[i]))
    return hu_moments.flatten()

def main():
    global roi_coords
    cap = cv2.VideoCapture(0) # 웹캠 번호 확인 (0 또는 1)
    
    cv2.namedWindow("Shadow Robust Classifier")
    cv2.setMouseCallback("Shadow Robust Classifier", mouse_callback)

    templates = {}
    THRESHOLD = 0.5 # 적응형 이진화는 윤곽선이 매우 정교하므로 기준을 더 엄격하게(낮게) 잡아도 됨
    
    message = ""
    message_timer = 0

    print("=== Lighting Robust Mode ===")
    print("조명 변화에 강한 [적응형 이진화] 모드가 적용되었습니다.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        display_frame = frame.copy()
        process_area = None
        offset_x, offset_y = 0, 0

        # ROI 그리기
        if roi_coords is not None:
            x1, y1, x2, y2 = roi_coords
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            process_area = frame[y1:y2, x1:x2]
            offset_x, offset_y = x1, y1
        else:
            cv2.putText(display_frame, "Set ROI first!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if process_area is not None and process_area.size > 0:
            # 1. 전처리 (조명 문제 해결의 핵심)
            gray = cv2.cvtColor(process_area, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            
            # [기존] 단순 이진화 -> 조명 변화에 취약
            # _, thresh = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY_INV)
            
            # [변경] 적응형 이진화 (Adaptive Thresholding)
            # blockSize=21: 주변 21x21 픽셀을 보고 판단 (홀수여야 함)
            # C=5: 계산된 평균에서 뺄 값 (이 값을 조절하여 감도 조절)
            thresh = cv2.adaptiveThreshold(
                blurred, 
                255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 
                21, 
                5
            )
            
            # 자잘한 노이즈 제거 (모폴로지)
            kernel = np.ones((3, 3), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel) # 점 잡음 제거
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel) # 구멍 메우기

            # 디버깅: ROI 내부 이진화 화면을 작게 보여줌 (매우 중요)
            cv2.imshow("Binary Debug", thresh)

            # 2. 윤곽선 검출
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # 가장 큰 윤곽선 찾기
                raw_cnt = max(contours, key=cv2.contourArea)
                
                # 적응형 이진화는 외곽선만 따기 때문에 면적이 작게 잡힐 수 있음. 기준 완화 (200~500)
                if cv2.contourArea(raw_cnt) > 300:
                    epsilon = 0.02 * cv2.arcLength(raw_cnt, True)
                    approx_cnt = cv2.approxPolyDP(raw_cnt, epsilon, True)

                    # 화면 그리기
                    draw_cnt = approx_cnt + [offset_x, offset_y]
                    cv2.drawContours(display_frame, [draw_cnt], -1, (0, 255, 0), 2)
                    
                    # 3. Hu Moments & 분류
                    current_hu = get_hu_moments(approx_cnt)
                    
                    if current_hu is not None:
                        min_dist = float('inf')
                        detected_label = "Unknown"

                        if templates:
                            for label, stored_hu in templates.items():
                                dist = np.linalg.norm(current_hu - stored_hu)
                                if dist < min_dist:
                                    min_dist = dist
                                    real_label = label.split('_')[0]
                                    if dist < THRESHOLD:
                                        detected_label = real_label
                        
                        text_pos_y = offset_y - 10 if offset_y > 30 else offset_y + 30
                        cv2.putText(display_frame, f"{detected_label} ({min_dist:.2f})", 
                                    (offset_x, text_pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        
                        # --- 등록 ---
                        key = cv2.waitKey(1) & 0xFF
                        base_name = ""
                        if key == ord('c'): base_name = "Circle"
                        elif key == ord('t'): base_name = "Triangle"
                        elif key == ord('s'): base_name = "Square"
                        elif ord('1') <= key <= ord('9'): base_name = f"Custom{chr(key)}"
                        
                        if base_name:
                            count = sum(1 for k in templates if k.startswith(base_name))
                            save_name = f"{base_name}_{count+1}"
                            templates[save_name] = current_hu
                            message = f"Saved: {save_name}"
                            message_timer = 60
                            print(f"등록됨: {save_name}")

        key_ctrl = cv2.waitKey(1) & 0xFF
        if key_ctrl == ord('r'):
            templates = {}
            message = "Reset All"
            message_timer = 60
        elif key_ctrl == ord('q'):
            break

        if message_timer > 0:
            cv2.putText(display_frame, message, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            message_timer -= 1
            
        cv2.imshow("Shadow Robust Classifier", display_frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()