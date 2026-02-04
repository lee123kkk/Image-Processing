import cv2
import numpy as np
import time

VIDEO_PATH = "ConveyorBelt.mp4" 

RED, GREEN, BLUE = (0, 0, 255), (0, 255, 0), (255, 0, 0)
YELLOW, WHITE, GRAY, BLACK = (0, 255, 255), (255, 255, 255), (50, 50, 50), (0, 0, 0)

def detect_fast(img_gray, mask=None):
    fast = cv2.FastFeatureDetector_create(threshold=15)
    kps = fast.detect(img_gray, mask)
    vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    cv2.drawKeypoints(vis, kps, vis, color=RED)
    return vis, len(kps)

def detect_mser(img_gray, mask=None):
    mser = cv2.MSER_create(delta=5, min_area=60, max_area=14400)
    regions, _ = mser.detectRegions(img_gray)
    vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    cv2.polylines(vis, hulls, 1, GREEN)
    return vis, len(regions)

def detect_blob(img_gray, mask=None):
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True; params.minArea = 100
    params.filterByCircularity = False
    detector = cv2.SimpleBlobDetector_create(params)
    kps = detector.detect(img_gray)
    vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    cv2.drawKeypoints(vis, kps, vis, color=BLUE, flags=4)
    return vis, len(kps)

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened(): print("영상 열기 실패!"); return

    FIXED_H = 720  
    cv2.namedWindow("Comparison Dashboard", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Comparison Dashboard", 1280, FIXED_H)

    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret: cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue

        h, w = frame.shape[:2]
        
        roi_x1, roi_x2 = int(w * 0.40), int(w * 0.60)
        roi_w = roi_x2 - roi_x1
        
        roi = frame[:, roi_x1:roi_x2]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)
        _, mask = cv2.threshold(roi_blur, 160, 255, cv2.THRESH_BINARY)

        curr = time.time(); dt = curr - prev_time; fps = 1/dt if dt>0 else 0; prev_time = curr

        img_fast, cnt_fast = detect_fast(roi_blur, mask)
        img_mser, cnt_mser = detect_mser(roi_blur)
        img_blob, cnt_blob = detect_blob(roi_blur)

        corners = cv2.goodFeaturesToTrack(roi_blur, 50, 0.05, 10, mask=mask)
        main_vis = roi.copy()
        
        # [중요] 중심선을 왼쪽으로 이동 (35% 지점)
        # 만약 더 왼쪽으로 가야 하면 0.35 -> 0.3 으로 줄이세요
        # 더 오른쪽으로 가야 하면 0.35 -> 0.4 로 늘리세요
        SHIFT_RATIO = 0.43
        target_cx = int(roi_w * SHIFT_RATIO)
        target_cy = h // 2 
        
        gw, gh = 90, 140
        
        cv2.rectangle(main_vis, (target_cx-gw//2, target_cy-gh//2), (target_cx+gw//2, target_cy+gh//2), GREEN, 1)
        cv2.putText(main_vis, "GOAL", (target_cx-gw//2, target_cy-gh//2-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1)

        err_x, cnt_gftt = 0, 0
        status, guide, col = "SEARCHING", "", GRAY
        guide_col = GRAY

        if corners is not None:
            corners = np.int32(corners)
            cnt_gftt = len(corners)
            cx = int(np.mean(corners[:,0,0])); cy = int(np.mean(corners[:,0,1]))
            err_x = cx - target_cx
            
            is_ok = abs(err_x) < 15 
            col = GREEN if is_ok else RED
            status = "OK (DOCKED)" if is_ok else "ALIGNING"
            
            if err_x < -15: 
                guide = "MOVE RIGHT >>"
                guide_col = YELLOW
            elif err_x > 15: 
                guide = "<< MOVE LEFT"
                guide_col = YELLOW
            else: 
                guide = "PERFECT!"
                guide_col = GREEN

            for c in corners: cv2.circle(main_vis, (c[0,0], c[0,1]), 3, YELLOW, -1)
            cv2.rectangle(main_vis, (cx-gw//2, cy-gh//2), (cx+gw//2, cy+gh//2), col, 2)
            cv2.circle(main_vis, (cx, cy), 6, col, -1)
            if not is_ok: cv2.arrowedLine(main_vis, (target_cx, cy), (cx, cy), RED, 2)

            text_pos_y = max(30, cy - 80)
            (text_w, text_h), _ = cv2.getTextSize(guide, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(main_vis, (cx - text_w//2 - 5, text_pos_y - text_h - 5), 
                          (cx + text_w//2 + 5, text_pos_y + 5), BLACK, -1)
            cv2.putText(main_vis, guide, (cx - text_w//2, text_pos_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, guide_col, 2)

        cv2.line(main_vis, (target_cx, 0), (target_cx, h), GREEN, 1)

        # 레이아웃 조립
        main_view_w = 500
        main_view = cv2.resize(main_vis, (main_view_w, FIXED_H))

        mini_w, mini_h = 250, FIXED_H // 3
        m_fast = cv2.resize(img_fast, (mini_w, mini_h))
        m_mser = cv2.resize(img_mser, (mini_w, mini_h))
        m_blob = cv2.resize(img_blob, (mini_w, mini_h))
        
        for img, txt in [(m_fast, "FAST"), (m_mser, "MSER"), (m_blob, "Blob")]:
            cv2.putText(img, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
            cv2.rectangle(img, (0,0), (mini_w, mini_h), WHITE, 2)

        minimap_col = np.vstack([m_fast, m_mser, m_blob])
        minimap_col = cv2.resize(minimap_col, (mini_w, FIXED_H))

        panel_w = 400
        panel = np.zeros((FIXED_H, panel_w, 3), dtype=np.uint8)

        y = 50
        cv2.putText(panel, "[Dashboard]", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2)
        y += 40
        cv2.putText(panel, f"FPS: {fps:.1f}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2)
        
        y += 40
        cv2.rectangle(panel, (20, y), (380, y+60), col, -1)
        cv2.putText(panel, status, (30, y+40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, BLACK, 2)
        
        y += 90
        cv2.putText(panel, f"Error: {err_x:+d} px", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
        
        y += 50
        cv2.putText(panel, guide, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, guide_col, 3)

        y += 60
        cv2.putText(panel, f"{'Algo':<6} {'Cnt':<6} {'Result'}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 1)
        cv2.line(panel, (20, y+10), (380, y+10), GRAY, 1)
        
        data = [("GFTT", cnt_gftt, "Best"), ("FAST", cnt_fast, "Noisy"), 
                ("MSER", cnt_mser, "Bad"), ("Blob", cnt_blob, "Fail")]
        
        for i, (n, c, res) in enumerate(data):
            col_txt = GREEN if n=="GFTT" else WHITE
            py = y + 35 + (i * 30)
            cv2.putText(panel, f"{n:<6} {c:<6} {res}", (20, py), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col_txt, 1)

        final = np.hstack([main_view, minimap_col, panel])
        cv2.imshow("Comparison Dashboard", final)
        
        if cv2.waitKey(30) & 0xFF == 27: break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__": main()
