import cv2
import numpy as np
import time
import sys

# -----------------------------------------------------------
# 설정 변수
# -----------------------------------------------------------
VIDEO_PATH = "conveyor_belt.webm"
TARGET_WIDTH = 800
ROI_WIDTH = 250
CENTER_TOLERANCE = 20
ANGLE_TOLERANCE = 5

# -----------------------------------------------------------
# 유틸리티: 각도 보정 함수
# -----------------------------------------------------------
def get_orientation_error(rect):
    # (수정사항 없음: 기존 코드와 동일)
    (cx, cy), (w, h), angle = rect
    if w < h:
        if angle > 45: error = angle - 90
        else: error = angle
    else:
        error = angle if angle < 45 else angle - 90
        if abs(error) < 45: error += 90 
        if error > 90: error -= 180
    return error

# -----------------------------------------------------------
# 알고리즘별 특징점 검출 함수
# -----------------------------------------------------------
def get_keypoints(gray, algo_name):
    # (수정사항 없음: 기존 코드와 동일)
    keypoints = []
    if algo_name == "FAST":
        detector = cv2.FastFeatureDetector_create(threshold=30, nonmaxSuppression=True)
        keypoints = detector.detect(gray, None)
    elif algo_name == "GFTT":
        pts = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
        if pts is not None:
            keypoints = [cv2.KeyPoint(x=p[0][0], y=p[0][1], size=10) for p in pts]
    elif algo_name == "MSER":
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)
        for p in regions:
            M = cv2.moments(p)
            if M["m00"] > 0:
                keypoints.append(cv2.KeyPoint(x=M["m10"]/M["m00"], y=M["m01"]/M["m00"], size=10))
    elif algo_name == "SimpleBlob":
        params = cv2.SimpleBlobDetector_Params()
        params.minArea = 100; params.maxArea = 5000
        params.filterByCircularity = False
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(gray)
    return keypoints

# -----------------------------------------------------------
# 메인 로직 (수정됨)
# -----------------------------------------------------------
def main():
    algorithms = ["FAST", "GFTT", "MSER", "SimpleBlob"]
    
    for algo in algorithms:
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            print(f"[Error] '{VIDEO_PATH}' 파일을 찾을 수 없습니다.")
            sys.exit()

        print(f"=== {algo} 모드 시작 (Multi-Box Detection) ===")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"[{algo}] 재생 완료")
                break

            # 1. 리사이즈
            ratio = TARGET_WIDTH / frame.shape[1]
            frame = cv2.resize(frame, (TARGET_WIDTH, int(frame.shape[0] * ratio)))
            h, w = frame.shape[:2]
            
            # 2. ROI 설정
            roi_x1 = (w // 2) - (ROI_WIDTH // 2)
            roi_x2 = roi_x1 + ROI_WIDTH
            roi_frame = frame[:, roi_x1:roi_x2]
            
            # 전처리
            gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            gray_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)

            # 3. 특징점 검출
            kps = get_keypoints(gray_roi, algo)
            
            # 4. 시각화 및 데이터 준비
            vis_frame = frame.copy()
            # ROI 라인 표시
            cv2.line(vis_frame, (roi_x1, 0), (roi_x1, h), (0, 255, 0), 1)
            cv2.line(vis_frame, (roi_x2, 0), (roi_x2, h), (0, 255, 0), 1)
            
            # 특징점 마스크 생성 및 팽창 (덩어리 만들기)
            mask = np.zeros_like(gray_roi)
            pts_array = cv2.KeyPoint_convert(kps)
            
            box_data_list = [] # 검출된 박스 정보를 담을 리스트

            if len(pts_array) > 0:
                for pt in pts_array:
                    cv2.circle(mask, (int(pt[0]), int(pt[1])), 5, 255, -1)
                
                kernel = np.ones((15, 15), np.uint8) 
                mask = cv2.dilate(mask, kernel, iterations=2)
                
                # 외곽선 검출
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # [수정] 유효한 박스 필터링 및 정렬
                valid_contours = []
                for cnt in contours:
                    if cv2.contourArea(cnt) > 1000: # 너무 작은 노이즈 제외
                        valid_contours.append(cnt)
                
                # Y축 기준 정렬 (위에서 아래로 내려오는 순서대로 A, B, C...)
                # boundingRect(cnt) -> (x, y, w, h) 이므로 y값([1]) 기준으로 정렬
                valid_contours = sorted(valid_contours, key=lambda c: cv2.boundingRect(c)[1])

                # [수정] 모든 박스 순회
                for i, cnt in enumerate(valid_contours):
                    box_id = chr(65 + (i % 26)) # A, B, C ... 생성
                    
                    # 회전된 사각형 계산
                    rect = cv2.minAreaRect(cnt)
                    (rx, ry), (rw, rh), rangle = rect
                    
                    # 좌표 변환 (ROI -> Global)
                    global_cx = rx + roi_x1
                    global_cy = ry
                    
                    # 박스 그리기
                    box = cv2.boxPoints(((global_cx, global_cy), (rw, rh), rangle))
                    box = np.int32(box)
                    cv2.drawContours(vis_frame, [box], 0, (0, 255, 255), 2)
                    
                    # ID 및 중심점 표시
                    cv2.circle(vis_frame, (int(global_cx), int(global_cy)), 5, (0, 0, 255), -1)
                    cv2.putText(vis_frame, f"Box {box_id}", (int(global_cx)-20, int(global_cy)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    # 오차 계산
                    dx = global_cx - (w // 2)
                    angle_err = get_orientation_error(rect)
                    
                    box_data_list.append({
                        "id": box_id,
                        "dx": dx,
                        "angle_err": angle_err
                    })

                # 특징점 시각화 (녹색 점)
                for pt in pts_array:
                     cv2.circle(vis_frame, (int(pt[0] + roi_x1), int(pt[1])), 2, (0, 255, 0), -1)

            # -------------------------------------
            # [오른쪽 패널] 대시보드 (리스트 형태)
            # -------------------------------------
            dashboard = np.zeros((h, 400, 3), dtype=np.uint8)
            
            # 헤더
            cv2.putText(dashboard, "Algorithm:", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150,150,150), 1)
            cv2.putText(dashboard, algo, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
            
            cv2.line(dashboard, (20, 100), (380, 100), (100, 100, 100), 1)
            cv2.putText(dashboard, "Detected Boxes Status:", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

            # 박스 리스트 출력
            start_y = 170
            line_height = 60
            
            if not box_data_list:
                cv2.putText(dashboard, "Scanning...", (20, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100,100,100), 1)
            
            for item in box_data_list:
                # 텍스트 색상 결정 (오차 크면 빨강, 작으면 초록)
                is_pos_bad = abs(item['dx']) > CENTER_TOLERANCE
                is_ang_bad = abs(item['angle_err']) > ANGLE_TOLERANCE
                
                # 위치 상태 텍스트
                if item['dx'] < -CENTER_TOLERANCE: pos_str = ">> Right"
                elif item['dx'] > CENTER_TOLERANCE: pos_str = "<< Left"
                else: pos_str = "OK"
                
                # 각도 상태 텍스트
                if item['angle_err'] > ANGLE_TOLERANCE: ang_str = "Rot CW"
                elif item['angle_err'] < -ANGLE_TOLERANCE: ang_str = "Rot CCW"
                else: ang_str = "OK"

                # 메인 컬러 (하나라도 불량이면 빨강)
                main_color = (0, 0, 255) if (is_pos_bad or is_ang_bad) else (0, 255, 0)
                
                # 한 줄 출력: [Box A] OK / Rot CW
                info_text = f"[{item['id']}] Pos:{pos_str} | Ang:{ang_str}"
                detail_text = f"    (dx:{int(item['dx'])}, deg:{item['angle_err']:.1f})"
                
                cv2.putText(dashboard, info_text, (20, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, main_color, 2)
                cv2.putText(dashboard, detail_text, (20, start_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
                
                start_y += line_height
                # 화면 넘어가면 생략
                if start_y > h - 20: break

            final = cv2.hconcat([vis_frame, dashboard])
            cv2.imshow("Multi-Box Alignment System", final)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                sys.exit()
            elif key == ord('n'):
                break 

        cap.release()
        time.sleep(1)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()