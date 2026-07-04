"""배경(체스보드)을 아는 이점을 이용한 무학습 물체 추출 (arXiv:2012.07287 IEM 원리 응용).

IEM 핵심 = "배경은 예측 가능, 전경(물체)은 예측 불가 → 예측 잔차가 곧 물체".
우리는 배경(ChArUco 보드)을 정확히 알고, ArUco로 매 프레임 자세를 얻으므로:
  1) 매 프레임을 보드 자세로 정면(canonical)으로 편다(rectify).
  2) '빈 보드'를 같은 방식으로 편 기준 프레임과 차분한다.
  3) 남는 잔차 = 물체. 색/채도 임계 불필요, 흰·검·반투명 물체도 잡힘, 카메라 이동에 강인.

canonical 픽셀 = 보드평면 좌표(1px = square_len/ppc [m])라, 중심·크기가 바로 mm로 나온다.
"""
from __future__ import annotations
import os
import numpy as np
import cv2
import aruco_utils as au


def rectify_to_canonical(img, board, K, dist, square_len, ppc=150, squares_xy=(5, 7)):
    """보드 자세로 이미지를 정면(canonical)으로 펴기.

    반환 (rect_bgr, ppm, pose(rvec,tvec)) 또는 보드 미검출 시 None.
    canonical 크기 = (squares_x*ppc, squares_y*ppc). ppm = ppc/square_len [px/m].
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rvec, tvec, cc, ci = au.detect_charuco_pose(gray, board, K, dist)
    if rvec is None:
        return None
    sx, sy = squares_xy
    Wc, Hc = sx * ppc, sy * ppc
    ppm = ppc / square_len
    R, _ = cv2.Rodrigues(rvec)
    H_img_board = K @ np.column_stack([R[:, 0], R[:, 1], tvec.flatten()])
    H_img_canon = H_img_board @ np.array([[1/ppm, 0, 0], [0, 1/ppm, 0], [0, 0, 1.0]])
    rect = cv2.warpPerspective(img, np.linalg.inv(H_img_canon), (Wc, Hc))
    return rect, ppm, (rvec, tvec)


def canonical_homography(rvec, tvec, K, ppm):
    """이미지 ← canonical(보드평면*ppm) 호모그래피 H_img_canon."""
    R, _ = cv2.Rodrigues(rvec)
    H_img_board = K @ np.column_stack([R[:, 0], R[:, 1], tvec.flatten()])
    return H_img_board @ np.array([[1/ppm, 0, 0], [0, 1/ppm, 0], [0, 0, 1.0]])


def make_reference_canonical(empty_frame, board, K, dist, square_len, ppc=150, squares_xy=(5, 7)):
    """빈 보드 프레임의 canonical(정면) 컬러 기준. 이걸 현재 시점으로 워프해 배경예측에 씀."""
    r = rectify_to_canonical(empty_frame, board, K, dist, square_len, ppc, squares_xy)
    return None if r is None else r[0]


def detect_objects_image(frame, board, K, dist, ref_canon, square_len,
                         ppc=150, squares_xy=(5, 7), thresh=45,
                         min_area_px=1500, open_px=7, close_px=21, shrink=25):
    """원본 이미지 공간에서 배경-차분 물체 검출 (정면 변환 없이, 자연스러운 시점).

    기준(canonical 빈 보드)을 현재 자세로 워프 → 배경 예측 → 원본과 차분.
    물체별 크기/중심은 바닥접점을 보드평면에 역투영해 mm로.
    반환 dict(pose, region, mask, objects[]) 또는 보드 미검출 시 None.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rvec, tvec, cc, ci = au.detect_charuco_pose(gray, board, K, dist)
    if rvec is None:
        return None
    Hf, Wf = frame.shape[:2]
    sx, sy = squares_xy
    ppm = ppc / square_len
    Hic = canonical_homography(rvec, tvec, K, ppm)
    pred = cv2.warpPerspective(ref_canon, Hic, (Wf, Hf))
    ones = np.full((sy * ppc, sx * ppc), 255, np.uint8)
    region = cv2.warpPerspective(ones, Hic, (Wf, Hf))
    if shrink:
        region = cv2.erode(region, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (shrink, shrink)))

    def bp(im):
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return cv2.GaussianBlur(g, (0, 0), 2) - cv2.GaussianBlur(g, (0, 0), 25)

    res = np.abs(bp(frame) - bp(pred))
    res[region == 0] = 0
    res = cv2.GaussianBlur(res, (0, 0), 3)
    resn = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    th = (resn > thresh).astype(np.uint8) * 255
    if open_px:
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px)))
    if close_px:
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px)))

    n, lab, stats, _ = cv2.connectedComponentsWithStats(th)
    objs = []
    mask = np.zeros(th.shape, np.uint8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area_px:
            continue
        mask[lab == i] = 255
        cm = (lab == i).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(cm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        pts = c.reshape(-1, 2)
        yb = y + h - max(3, int(h * 0.15))
        base_px = pts[pts[:, 1] >= yb].astype(np.float64)
        if len(base_px) < 3:
            base_px = pts[np.argsort(pts[:, 1])[-5:]].astype(np.float64)
        base_plane = au.pixels_to_plane(base_px, K, dist, rvec, tvec)[:, :2]
        center = base_plane.mean(axis=0)
        rect = cv2.minAreaRect(base_plane.astype(np.float32))
        objs.append({"contour": c, "bbox": (int(x), int(y), int(w), int(h)),
                     "center_mm": (float(center[0]*1000), float(center[1]*1000)),
                     "size_mm": (float(rect[1][0]*1000), float(rect[1][1]*1000))})
    return {"rvec": rvec, "tvec": tvec, "region": region, "mask": mask, "objects": objs}


def draw_image(frame, det, K, dist, square_len=0.038):
    vis = frame.copy()
    if det is None:
        cv2.putText(vis, "board not found", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return vis
    cv2.drawFrameAxes(vis, K, dist, det["rvec"], det["tvec"], square_len*2, 2)
    for i, o in enumerate(det["objects"]):
        cv2.drawContours(vis, [o["contour"]], -1, (0, 255, 0), 2)
        x, y, w, h = o["bbox"]; cx, cy = o["center_mm"]; sw, sl = o["size_mm"]
        cv2.putText(vis, f"#{i} ({cx:.0f},{cy:.0f})mm", (x, max(18, y-24)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(vis, f"{sw:.0f}x{sl:.0f}mm", (x, max(34, y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(vis, f"objects: {len(det['objects'])}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return vis


def _bandpass(rect_bgr):
    """조명에 강인한 밴드패스(정면 그레이). 기준/현재 프레임에 동일 적용."""
    g = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return cv2.GaussianBlur(g, (0, 0), 2) - cv2.GaussianBlur(g, (0, 0), 25)


def make_reference(empty_frame, board, K, dist, square_len, ppc=150, squares_xy=(5, 7)):
    """물체 없는 '빈 보드' 프레임에서 canonical 기준(밴드패스) 생성. 실패 시 None."""
    r = rectify_to_canonical(empty_frame, board, K, dist, square_len, ppc, squares_xy)
    if r is None:
        return None
    return _bandpass(r[0])


def detect_objects_bgdiff(frame, board, K, dist, ref_bp, square_len,
                          ppc=150, squares_xy=(5, 7),
                          thresh=80, min_area_mm2=200.0,
                          open_px=13, close_px=25, border=25):
    """빈 보드 기준과의 차분으로 물체 검출 (색 무관).

    반환 dict(rect, residual, mask, ppm, objects[]) 또는 보드 미검출 시 None.
    objects: [{bbox_canon(x,y,w,h px), center_mm(x,y), size_mm(w,l), area_mm2}]
    """
    r = rectify_to_canonical(frame, board, K, dist, square_len, ppc, squares_xy)
    if r is None:
        return None
    rect, ppm, pose = r
    cur = _bandpass(rect)
    res = np.abs(cur - ref_bp)
    res = cv2.GaussianBlur(res, (0, 0), 4)
    resn = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    th = (resn > thresh).astype(np.uint8) * 255
    if open_px:
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px)))
    if close_px:
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px)))
    if border:
        th[:border] = 0; th[-border:] = 0; th[:, :border] = 0; th[:, -border:] = 0

    px2mm = 1000.0 / ppm
    min_area_px = min_area_mm2 / (px2mm ** 2)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(th)
    objs = []
    mask = np.zeros(th.shape, np.uint8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area_px:
            continue
        mask[lab == i] = 255
        x, y, w, h = stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3]
        cx, cy = cent[i]
        objs.append({
            "bbox_canon": (int(x), int(y), int(w), int(h)),
            "center_mm": (float(cx * px2mm), float(cy * px2mm)),
            "size_mm": (float(w * px2mm), float(h * px2mm)),
            "area_mm2": float(stats[i, cv2.CC_STAT_AREA] * px2mm ** 2),
        })
    return {"rect": rect, "residual": resn, "mask": mask, "ppm": ppm, "objects": objs}


def draw_canonical(det):
    """detect 결과를 canonical 이미지에 박스+치수로 그림."""
    vis = det["rect"].copy()
    for i, o in enumerate(det["objects"]):
        x, y, w, h = o["bbox_canon"]
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cx, cy = o["center_mm"]; sw, sl = o["size_mm"]
        cv2.putText(vis, f"#{i} ({cx:.0f},{cy:.0f})mm", (x, max(18, y-24)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(vis, f"{sw:.0f}x{sl:.0f}mm", (x, max(34, y-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(vis, f"objects: {len(det['objects'])}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return vis


def run_live_image(K, dist, board, square_len, cam_index=0, calib_wh=(1920, 1080),
                   ppc=150, squares_xy=(5, 7), snapshot_dir=".", **det_kw):
    """실시간(자연 시점): [r]=빈 보드를 기준으로, [s]=스냅(raw+vis), [q]=종료.

    기준을 현재 자세로 워프해 원본 이미지에서 차분 → 자연스러운 카메라 시점 오버레이.
    정면(canonical) 변환 없이 결과가 원본 뷰로 나오고, 세운 물체도 한 덩어리로 잡힘.
    """
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, calib_wh[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, calib_wh[1])
    if not cap.isOpened():
        raise RuntimeError(f"웹캠({cam_index}) 열기 실패")
    ref_canon = None
    snap = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if ref_canon is None:
                view = frame.copy()
                cv2.putText(view, "empty board -> press [r] to set reference", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            else:
                det = detect_objects_image(frame, board, K, dist, ref_canon, square_len,
                                           ppc=ppc, squares_xy=squares_xy, **det_kw)
                view = draw_image(frame, det, K, dist, square_len)
            cv2.imshow("bgdiff live (natural view) [r]ref [s]snap [q]quit", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r'):
                ref_canon = make_reference_canonical(frame, board, K, dist, square_len, ppc, squares_xy)
                print("reference set" if ref_canon is not None else "board not found - reference failed")
            if key == ord('s'):
                cv2.imwrite(os.path.join(snapshot_dir, f"snap_raw_{snap:03d}.png"), frame)
                cv2.imwrite(os.path.join(snapshot_dir, f"snap_vis_{snap:03d}.png"), view)
                print("snapshot", snap); snap += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


def run_live_bgdiff(K, dist, board, square_len, cam_index=0, calib_wh=(1920, 1080),
                    ppc=150, squares_xy=(5, 7), snapshot_dir=".", **det_kw):
    """실시간: [r]=현재(빈 보드)를 기준으로 설정, [q]=종료, [s]=스냅샷(raw+vis).

    사용법: 보드만 놓고 r → 물체 올리면 실시간 검출. 로컬 실행 전용(cv2 창).
    """
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, calib_wh[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, calib_wh[1])
    if not cap.isOpened():
        raise RuntimeError(f"웹캠({cam_index}) 열기 실패")
    ref_bp = None
    snap = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if ref_bp is None:
                view = frame.copy()
                cv2.putText(view, "빈 보드 놓고 [r]로 기준 설정", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            else:
                det = detect_objects_bgdiff(frame, board, K, dist, ref_bp, square_len,
                                            ppc=ppc, squares_xy=squares_xy, **det_kw)
                view = draw_canonical(det) if det else frame.copy()
            cv2.imshow("bgdiff live [r]ref [s]snap [q]quit", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r'):
                ref_bp = make_reference(frame, board, K, dist, square_len, ppc, squares_xy)
                print("기준 설정" if ref_bp is not None else "보드 미검출 — 기준 실패")
            if key == ord('s'):
                cv2.imwrite(os.path.join(snapshot_dir, f"snap_raw_{snap:03d}.png"), frame)
                cv2.imwrite(os.path.join(snapshot_dir, f"snap_vis_{snap:03d}.png"), view)
                print("스냅샷 저장", snap); snap += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
