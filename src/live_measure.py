"""보드 위 복수 물체를 (색 무관) 고전 필터로 탐지하고 크기/중심/위치를 재는 실시간 모듈.

- measure_objects(frame, ...): 한 프레임 처리 → 물체 리스트 (정지영상으로도 테스트 가능)
- draw_overlay(frame, ...): 결과를 프레임에 그림
- run_live(...): 웹캠 루프 (cv2.imshow, q=종료 s=스냅샷)

측정 원리는 aruco_utils(detect_charuco_pose, board_region_mask, pixels_to_plane,
height_from_vertical_edge) 재사용. 색 무관 탐지는 '보드영역에 닿은 채도 덩어리'.
"""
from __future__ import annotations
import os
import numpy as np
import cv2
import aruco_utils as au


def scale_intrinsics(K, from_wh, to_wh):
    """캘리브레이션 해상도와 프레임 해상도가 다르면 K를 스케일."""
    sx = to_wh[0] / from_wh[0]
    sy = to_wh[1] / from_wh[1]
    K2 = K.copy()
    K2[0, 0] *= sx; K2[0, 2] *= sx
    K2[1, 1] *= sy; K2[1, 2] *= sy
    return K2


def measure_objects(frame, board, K, dist,
                    sat_thresh=120, val_thresh=110,
                    min_area=800, min_touch=30,
                    open_px=5, close_px=9, base_band_frac=0.15):
    """한 프레임에서 보드 위 복수 물체를 탐지·측정.

    반환: dict(
        board_found: bool, rvec, tvec, hull,
        objects: [ {contour, bbox, center_mm(x,y), footprint_mm(w,l),
                     height_mm, base_px, top_px} ... ]  (보드 좌표계, mm)
    )
    """
    out = {"board_found": False, "rvec": None, "tvec": None, "hull": None, "objects": []}
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rvec, tvec, cc, ci = au.detect_charuco_pose(gray, board, K, dist)
    if rvec is None:
        return out
    out.update(board_found=True, rvec=rvec, tvec=tvec)
    region, hull = au.board_region_mask(frame.shape, cc, dilate_px=25)
    out["hull"] = hull

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    fg = ((hsv[:, :, 1] > sat_thresh) & (hsv[:, :, 2] > val_thresh)).astype(np.uint8) * 255
    if open_px:
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px)))
    if close_px:
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px)))

    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg)
    for i in range(1, n):
        comp = (lab == i)
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        if (comp & (region > 0)).sum() < min_touch:   # 보드에 닿은 것만
            continue
        cm = (comp.astype(np.uint8)) * 255
        cnts, _ = cv2.findContours(cm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        pts = c.reshape(-1, 2)

        # 바닥 접점 밴드(윤곽 하단부)만 평면에 역투영 → footprint/center
        yb = y + h - max(3, int(h * base_band_frac))
        base_pixels = pts[pts[:, 1] >= yb].astype(np.float64)
        if len(base_pixels) < 3:
            base_pixels = pts[np.argsort(pts[:, 1])[-5:]].astype(np.float64)
        base_plane = au.pixels_to_plane(base_pixels, K, dist, rvec, tvec)[:, :2]
        center_xy = base_plane.mean(axis=0)
        rect = cv2.minAreaRect(base_plane.astype(np.float32))
        (rw, rl) = rect[1]

        base_center_px = base_pixels.mean(axis=0)
        top_px = pts[np.argmin(pts[:, 1])].astype(np.float64)
        height, _ = au.height_from_vertical_edge(base_center_px, top_px, K, dist, rvec, tvec)

        out["objects"].append({
            "contour": c, "bbox": (int(x), int(y), int(w), int(h)),
            "center_mm": (float(center_xy[0] * 1000), float(center_xy[1] * 1000)),
            "footprint_mm": (float(rw * 1000), float(rl * 1000)),
            "height_mm": float(height * 1000),
            "base_px": base_center_px.tolist(), "top_px": top_px.tolist(),
        })
    return out


def draw_overlay(frame, result, K, dist, square_len=0.038):
    vis = frame.copy()
    if not result["board_found"]:
        cv2.putText(vis, "board not found", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return vis
    cv2.drawFrameAxes(vis, K, dist, result["rvec"], result["tvec"], square_len * 2, 3)
    if result["hull"] is not None:
        cv2.polylines(vis, [result["hull"]], True, (0, 0, 255), 1)
    for idx, o in enumerate(result["objects"]):
        cv2.drawContours(vis, [o["contour"]], -1, (0, 255, 0), 2)
        x, y, w, h = o["bbox"]
        cx, cy = o["center_mm"]; fw, fl = o["footprint_mm"]; hh = o["height_mm"]
        lines = [f"#{idx}", f"pos ({cx:.0f},{cy:.0f})mm",
                 f"{fw:.0f}x{fl:.0f}x{hh:.0f}mm"]
        for j, t in enumerate(lines):
            cv2.putText(vis, t, (x, max(20, y - 8 - j * 22)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(vis, f"objects: {len(result['objects'])}  [q]quit [s]snap",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return vis


def run_live(K, dist, board, cam_index=0, calib_wh=(1920, 1080),
             snapshot_dir=".", **measure_kw):
    """웹캠 실시간 루프. q=종료, s=스냅샷 저장. (로컬 실행 전용 — cv2.imshow 창)"""
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, calib_wh[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, calib_wh[1])
    if not cap.isOpened():
        raise RuntimeError(f"웹캠({cam_index})을 열 수 없습니다.")
    snap = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            wh = (frame.shape[1], frame.shape[0])
            Ku = K if wh == tuple(calib_wh) else scale_intrinsics(K, calib_wh, wh)
            res = measure_objects(frame, board, Ku, dist, **measure_kw)
            vis = draw_overlay(frame, res, Ku, dist)
            cv2.imshow("live object measure (q=quit, s=snap)", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                cv2.imwrite(os.path.join(snapshot_dir, f"snap_vis_{snap:03d}.png"), vis)
                cv2.imwrite(os.path.join(snapshot_dir, f"snap_raw_{snap:03d}.png"), frame)
                print("saved snap", snap, "(vis+raw)"); snap += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
