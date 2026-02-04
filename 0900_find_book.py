import cv2
import numpy as np
import os
import glob

def bookSearcher():
    # 1. 책 표지 데이터베이스 경로 설정
    DB_PATH = './books'
    if not os.path.exists(DB_PATH):
        print(f"오류: '{DB_PATH}' 디렉터리가 없습니다. 폴더를 생성하고 책 이미지를 넣어주세요.")
        return

    # 이미지 파일 리스트 가져오기
    book_files = glob.glob(os.path.join(DB_PATH, '*.*'))
    # 확장자 필터링 (jpg, png 등)
    book_files = [f for f in book_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not book_files:
        print("오류: books 폴더에 이미지 파일이 없습니다.")
        return

    print(f"검색 로드 중... 총 {len(book_files)}권의 책이 있습니다.")

    # 2. 알고리즘 초기화 (SIFT 사용 - 회전, 크기 변화에 강함)
    detector = cv2.SIFT_create()
    
    # FLANN 매처 설정 (속도가 빠름)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)

    # 3. 카메라 실행
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    # 4. ROI(관심 영역) 설정 - 화면 중앙
    win_name = 'Book Searcher'
    cv2.namedWindow(win_name)
    
    # ROI 크기 설정 (320x400 정도의 직사각형)
    roi_w, roi_h = 320, 420 

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        # ROI 좌표 계산 (화면 중앙)
        roi_x = (w - roi_w) // 2
        roi_y = (h - roi_h) // 2
        
        # 화면 복사본에 가이드 사각형 그리기
        display_frame = frame.copy()
        cv2.rectangle(display_frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
        
        # 안내 문구
        cv2.putText(display_frame, "Place book in box & Press SPACE", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(win_name, display_frame)

        key = cv2.waitKey(1)
        if key == 27: # ESC
            break
        elif key == 32: # SPACE BAR -> 검색 시작
            
            # ROI 영역 잘라내기 (Query Image)
            roi_img = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            gray_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
            
            # 특징점 검출
            kp_roi, des_roi = detector.detectAndCompute(gray_roi, None)
            
            # 특징점이 너무 적으면 검색 불가
            if des_roi is None or len(kp_roi) < 10:
                print("특징점을 찾을 수 없습니다. 다시 시도해주세요.")
                continue

            best_match_file = None
            max_good_matches = 0
            best_accuracy = 0.0
            
            print("\n>>> 검색 시작...")

            # DB의 모든 책과 비교
            for file_path in book_files:
                # DB 이미지 로드
                db_img = cv2.imread(file_path)
                if db_img is None: continue
                
                # --- [기능 구현] 검색 중인 영상 빠르게 보여주기 ---
                # 사용자에게 현재 비교 중인 책 표지를 팝업으로 보여줌
                resize_db = cv2.resize(db_img, (300, 400))
                cv2.imshow('Searching...', resize_db)
                cv2.waitKey(10) # 0.01초 대기 (시각적 효과)
                # -----------------------------------------------

                gray_db = cv2.cvtColor(db_img, cv2.COLOR_BGR2GRAY)
                kp_db, des_db = detector.detectAndCompute(gray_db, None)

                if des_db is None: continue

                # 매칭 진행 (k=2 for Ratio Test)
                matches = matcher.knnMatch(des_roi, des_db, k=2)

                # Good Matches 선별 (Lowe's Ratio Test)
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

                # 정확도/매칭 점수 계산
                match_count = len(good_matches)
                
                # (Optional) 정상치 비율 계산: 매칭된 점 / 전체 ROI 특징점
                accuracy = match_count / len(kp_roi) * 100 

                # 디버깅 출력 (필요시 주석 해제)
                # print(f"Checking: {os.path.basename(file_path)} | Matches: {match_count}")

                # 최댓값 갱신
                if match_count > max_good_matches:
                    max_good_matches = match_count
                    best_match_file = file_path
                    best_accuracy = accuracy

            cv2.destroyWindow('Searching...') # 검색창 닫기

            # --- 결과 표시 ---
            if best_match_file and max_good_matches > 10: # 최소 매칭 개수 임계값 설정 (노이즈 방지)
                result_img = cv2.imread(best_match_file)
                book_name = os.path.basename(best_match_file)
                
                print("="*50)
                print(f"검색 완료!")
                print(f"책 이름: {book_name}")
                print(f"매칭 개수: {max_good_matches}")
                print(f"정확도(비율): {best_accuracy:.2f}%")
                print("="*50)
                
                # 결과 이미지에 텍스트 표시
                cv2.putText(result_img, f"Match: {max_good_matches}", (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                cv2.imshow('Best Result', result_img)
            else:
                print("일치하는 책을 찾지 못했습니다.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    bookSearcher()
#===============================================================
#
# ROI (Region of Interest) 활용:
#전체 화면을 검색하면 배경 노이즈가 섞여 정확도가 떨어집니다. frame[roi_y:..., roi_x:...] 슬라이싱을 통해 중앙 박스 안의 이미지만 잘라내어 검색(Query) 이미지로 사용했습니다

# 검색 과정 시각화:
# for file_path in book_files: 반복문 내부에서 cv2.imshow('Searching...', resize_db)를 호출하여 현재 어떤 책과 비교하고 있는지 플래시처럼 빠르게 보여주어 사용자의 지루함을 덜었습니다

# SIFT & FLANN Matcher:
# 책 표지는 글자, 그림 등 디테일이 많으므로 단순 코너 검출(FAST)보다는 디스크립터(특징 기술자)가 포함된 SIFT가 유리합니다.
# 많은 이미지를 검색해야 하므로 무차별 대입(BFMatcher)보다 빠른 FlannBasedMatcher를 사용했습니다.

#Lowe's Ratio Test:
#m.distance < 0.75 * n.distance 조건을 통해 확실히 유사한 특징점(Good Matches)만 카운트하여 정확도를 높였습니다.

# 실시간 영상 처리 애플리케이션에서는 단순히 정확한 결과를 내는 것 뿐만 아니라 ROI를 통한 입력 정규화와 처리 과정의 시각화가 사용자 경험을 결정짓는 핵심 요소이다.