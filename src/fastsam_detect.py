"""FastSAM(정밀 마스크) + 배경차분(물체 선별) 결합 검출.

문제: FastSAM은 물체를 깔끔히 분할하지만 보드 칸·마커·배경까지 전부 분할한다.
해법: '빈 보드 대비 달라진 영역'(bg_segment 배경차분, 색 무관)과 겹치는 FastSAM 마스크만
      물체로 선별. FastSAM은 경계를, 배경차분은 '무엇이 물체인지'를 담당.
기준(빈 보드)은 라이브에서 [r]로 같은 각도에서 잡아야 깨끗하다.
"""
from __future__ import annotations
import os
import numpy as np
import cv2
import aruco_utils as au
import bg_segment as bg


def load_fastsam(weights="FastSAM-s.pt"):
    from ultralytics import FastSAM
    return FastSAM(weights)


def _measure(contour, K, dist, rvec, tvec):
    x, y, w, h = cv2.boundingRect(contour)
    pts = contour.reshape(-1, 2)
    yb = y + h - max(3, int(h * 0.15))
    base = pts[pts[:, 1] >= yb].astype(np.float64)
    if len(base) < 3:
        base = pts[np.argsort(pts[:, 1])[-5:]].astype(np.float64)
    bp = au.pixels_to_plane(base, K, dist, rvec, tvec)[:, :2]
    center = bp.mean(axis=0) * 1000
    rect = cv2.minAreaRect(bp.astype(np.float32))
    return (int(x), int(y), int(w), int(h)), (float(center[0]), float(center[1])), \
           (float(rect[1][0] * 1000), float(rect[1][1] * 1000))


def detect_objects_fastsam(frame, model, board, K, dist, ref_canon, square_len,
                           imgsz=1024, overlap_thresh=0.30, min_area_px=1000,
                           max_area_frac=0.40, dedup=0.6, device="cuda", **bgkw):
    """FastSAM + 배경차분 선별. ref_canon(빈 보드 canonical) 필요.

    반환 dict(rvec, tvec, fg, objects[]) 또는 보드 미검출 시 None.
    objects: [{contour, bbox, center_mm, size_mm}]
    """
    det = bg.detect_objects_image(frame, board, K, dist, ref_canon, square_len, **bgkw)
    if det is None:
        return None
    fg = det["mask"] > 0
    rvec, tvec = det["rvec"], det["tvec"]
    H, W = frame.shape[:2]

    res = model(frame, device=device, retina_masks=True, imgsz=imgsz,
                conf=0.4, iou=0.9, verbose=False)
    masks = res[0].masks.data.cpu().numpy() if res[0].masks is not None else np.zeros((0, H, W))

    cands = []
    for m in masks:
        b = m > 0.5
        a = int(b.sum())
        if a < min_area_px or a > max_area_frac * H * W:
            continue
        if (b & fg).sum() / max(a, 1) < overlap_thresh:   # 물체영역과 겹치는 것만
            continue
        cands.append((a, b))
    cands.sort(key=lambda o: -o[0])

    objects, acc = [], np.zeros((H, W), bool)
    for a, b in cands:
        if (b & acc).sum() / max(b.sum(), 1) > dedup:      # 중복 제거
            continue
        acc |= b
        cm = b.astype(np.uint8) * 255
        cnts, _ = cv2.findContours(cm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        bbox, center, size = _measure(c, K, dist, rvec, tvec)
        objects.append({"contour": c, "bbox": bbox, "center_mm": center, "size_mm": size})
    return {"rvec": rvec, "tvec": tvec, "fg": det["mask"], "objects": objects}


def draw_fastsam(frame, det, K, dist, square_len=0.038):
    vis = frame.copy()
    if det is None:
        cv2.putText(vis, "board not found", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return vis
    cv2.drawFrameAxes(vis, K, dist, det["rvec"], det["tvec"], square_len * 2, 2)
    for i, o in enumerate(det["objects"]):
        cv2.drawContours(vis, [o["contour"]], -1, (0, 255, 0), 2)
        x, y, w, h = o["bbox"]; cx, cy = o["center_mm"]; sw, sl = o["size_mm"]
        cv2.putText(vis, f"#{i} ({cx:.0f},{cy:.0f})mm", (x, max(18, y-24)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(vis, f"{sw:.0f}x{sl:.0f}mm", (x, max(34, y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(vis, f"objects: {len(det['objects'])}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return vis


def run_live_fastsam(K, dist, board, square_len, model=None, cam_index=0,
                     calib_wh=(1920, 1080), ppc=150, squares_xy=(5, 7),
                     imgsz=640, snapshot_dir=".", **det_kw):
    """실시간: [r]=빈 보드 기준 설정, [s]=스냅(raw+vis), [q]=종료.

    빈 보드에서 r → 물체 올리면 FastSAM+배경차분으로 검출. imgsz=640이 실시간에 유리.
    """
    if model is None:
        model = load_fastsam()
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
                cv2.putText(view, "empty board -> [r] set reference", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            else:
                det = detect_objects_fastsam(frame, model, board, K, dist, ref_canon,
                                             square_len, imgsz=imgsz, **det_kw)
                view = draw_fastsam(frame, det, K, dist, square_len)
            cv2.imshow("FastSAM+bgdiff [r]ref [s]snap [q]quit", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r'):
                ref_canon = bg.make_reference_canonical(frame, board, K, dist, square_len, ppc, squares_xy)
                print("reference set" if ref_canon is not None else "board not found - reference failed")
            if key == ord('s'):
                cv2.imwrite(os.path.join(snapshot_dir, f"snap_raw_{snap:03d}.png"), frame)
                cv2.imwrite(os.path.join(snapshot_dir, f"snap_vis_{snap:03d}.png"), view)
                print("snapshot", snap); snap += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
