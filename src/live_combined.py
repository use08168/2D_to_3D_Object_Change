"""실시간 통합: DA(감지) + FastSAM(실루엣) + ArUco(측정), 자연 웹캠 시점 합성.

각 도구를 강점에만 사용:
- DA 높이맵 → 물체 감지 + 선/누움 판별
- FastSAM → 완전한 실루엣(경계)
- ArUco 기하 → 정확한 높이(수직모서리)·길이·위치

표시: 원본(왜곡보정) 화면 위에 물체 부위만 높이 히트맵을 반투명 합성 + 윤곽/측정 라벨.
우상단에 깊이/높이 맵 인셋. (정면 top-down 변환 안 함 — 자연 시점 유지)
"""
from __future__ import annotations
import os
import numpy as np
import cv2
import aruco_utils as au
import depth_volume as dv
import scene3d as s3


STAND_TH_MM = 25.0   # 이 이상이면 '선 물체'로 보고 수직 높이 측정


def _inset(base, small, x, y, w, h, label):
    s = cv2.resize(small, (w, h))
    if s.ndim == 2:
        s = cv2.cvtColor(s, cv2.COLOR_GRAY2BGR)
    base[y:y+h, x:x+w] = s
    cv2.rectangle(base, (x, y), (x+w, y+h), (255, 255, 255), 1)
    cv2.putText(base, label, (x+4, y+16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


def process_frame_combined(frame, pipe, model, board, K, dist, square_len=0.038,
                           min_height_mm=6, min_area_px=1200, imgsz=640,
                           device="cuda", marker_map=None):
    """한 프레임 → (합성 vis, objects, markers). 보드 미검출 시 (원본, [], []).

    markers = '전체 마커 지도'(검출과 무관, 항상 전부). marker_map을 주면 그걸(분산앵커 지도),
    없으면 보드 정의에서 전체 지도를 만들어 씀. → 가려져도 3D 씬에 마커가 안 사라짐.
    """
    hm = dv.height_map_from_depth(frame, pipe, board, K, dist, square_len=square_len)
    if hm is None:
        v = frame.copy()
        cv2.putText(v, "board not found", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return v, [], []
    imgu = hm["imgu"]; H, W = imgu.shape[:2]
    rvec, tvec = hm["rvec"], hm["tvec"]; z0 = np.zeros((5, 1))

    # 전체 마커 지도(고정) — 검출 개수와 무관하게 항상 전부 표시
    markers = marker_map if marker_map is not None else au.board_marker_map(board)
    height_mm = hm["height_mm"]
    seed = height_mm > min_height_mm                 # 보드 밖도 포함(솟은 것 안 자름)
    qm = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(qm, hm["quad"], 255)          # 보드평면(바닥 판정용)
    qm = qm > 0

    res = model(imgu, device=device, retina_masks=True, imgsz=imgsz,
                conf=0.4, iou=0.9, verbose=False)
    masks = res[0].masks.data.cpu().numpy() if res[0].masks is not None else np.zeros((0, H, W))

    cand = []
    for m in masks:
        b = m > 0.5; a = int(b.sum())
        if a < min_area_px or a > 0.4 * H * W:
            continue
        if (b & seed).sum() / a < 0.12:              # 솟은 영역과 겹침
            continue
        ys, xs = np.where(b)                          # 물체 '바닥'이 보드 위인지
        y0, y1 = ys.min(), ys.max()
        band = b & (np.arange(H)[:, None] >= y1 - max(3, int(0.12*(y1-y0))))
        if band.sum() == 0 or (band & qm).sum() / band.sum() < 0.4:
            continue
        cand.append((a, b))
    cand.sort(key=lambda o: -o[0])
    kept, acc = [], np.zeros((H, W), bool)
    for a, b in cand:
        if (b & acc).sum() / b.sum() > 0.6:
            continue
        kept.append(b); acc |= b

    objects = []
    objunion = np.zeros((H, W), bool)
    for b in kept:
        objunion |= b
        cm = b.astype(np.uint8) * 255
        cnts, _ = cv2.findContours(cm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea); pts = c.reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(c)
        da_h = float(np.nanmax(height_mm[b]))
        top = pts[np.argmin(pts[:, 1])].astype(np.float64)
        base = pts[np.argmax(pts[:, 1])].astype(np.float64)
        # base band -> plane (footprint / diameter)
        yb = y + h - max(3, int(h * 0.12))
        bpx = pts[pts[:, 1] >= yb].astype(np.float64)
        if len(bpx) < 3:
            bpx = pts[np.argsort(pts[:, 1])[-5:]].astype(np.float64)
        bpp = au.pixels_to_plane(bpx, K.astype(np.float64), z0, rvec, tvec)[:, :2]
        base_rect = cv2.minAreaRect(bpp.astype(np.float32))
        base_w, base_l = sorted([base_rect[1][0]*1000, base_rect[1][1]*1000])
        cx_mm, cy_mm = base_rect[0][0]*1000, base_rect[0][1]*1000

        if da_h > STAND_TH_MM:   # 선 물체: 수직 높이
            gh, _ = au.height_from_vertical_edge(base, top, K.astype(np.float64), z0, rvec, tvec)
            o = {"type": "stand", "contour": c, "bbox": (x, y, w, h),
                 "height_mm": gh*1000, "diam_mm": base_w,
                 "center_mm": (cx_mm, cy_mm)}
            o["label"] = f"H{gh*1000:.0f} D~{base_w:.0f}mm"
        else:                    # 누운 물체: 전체 실루엣 길이
            allp = au.pixels_to_plane(pts.astype(np.float64), K.astype(np.float64), z0, rvec, tvec)[:, :2]
            r = cv2.minAreaRect(allp.astype(np.float32))
            L, Wd = max(r[1])*1000, min(r[1])*1000
            o = {"type": "lie", "contour": c, "bbox": (x, y, w, h),
                 "length_mm": L, "width_mm": Wd, "center_mm": (r[0][0]*1000, r[0][1]*1000)}
            o["label"] = f"L{L:.0f} W{Wd:.0f}mm"
        o["cyl"] = s3.fit_cylinder(hm["pts_board"][b] * 1000.0)   # 가상3D용 원통(mm)
        objects.append(o)

    # ---- 합성 화면 ----
    vis = imgu.copy()
    hcolor = dv.colorize_height(hm, 60)
    m3 = objunion[:, :, None]
    vis = np.where(m3, (0.45*vis + 0.55*hcolor).astype(np.uint8), vis)  # 물체에만 높이 히트맵
    cv2.polylines(vis, [hm["quad"]], True, (0, 0, 255), 2)
    cv2.drawFrameAxes(vis, K, dist*0, rvec, tvec, square_len*2, 2)
    for i, o in enumerate(objects):
        cv2.drawContours(vis, [o["contour"]], -1, (0, 255, 0), 2)
        x, y, w, h = o["bbox"]
        cv2.putText(vis, f"#{i} {o['type']}", (x, max(30, y-26)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(vis, o["label"], (x, max(14, y-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(vis, f"objects: {len(objects)}  markers: {len(markers)}  anchor R^2={hm['r2']:.2f}",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    # 인셋: 높이맵
    iw, ih = W//5, H//5
    _inset(vis, hcolor, W-iw-10, 10, iw, ih, "height")
    return vis, objects, markers


def run_live_combined(K, dist, board, square_len=0.038, pipe=None, model=None,
                      cam_index=0, calib_wh=(1920, 1080), imgsz=640,
                      snapshot_dir=".", **proc_kw):
    """실시간 합성 뷰. [s]=스냅(raw+vis), [q]=종료. (모델 2개 매프레임이라 수 fps)"""
    if pipe is None:
        pipe = dv.load_depth_model()
    if model is None:
        from ultralytics import FastSAM
        model = FastSAM("FastSAM-s.pt")
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, calib_wh[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, calib_wh[1])
    if not cap.isOpened():
        raise RuntimeError(f"웹캠({cam_index}) 열기 실패")
    snap = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            vis, objs, markers = process_frame_combined(frame, pipe, model, board, K, dist,
                                                        square_len=square_len, imgsz=imgsz, **proc_kw)
            scene = s3.render_virtual_scene(objs, markers=markers, ws=(240, 320))   # 매 프레임 재렌더=add/remove
            camh = cv2.resize(vis, (int(vis.shape[1]*scene.shape[0]/vis.shape[0]), scene.shape[0]))
            vis = np.hstack([camh, scene])
            cv2.imshow("live: camera | virtual 3D  [s]snap [q]quit", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                cv2.imwrite(os.path.join(snapshot_dir, f"snap_raw_{snap:03d}.png"), frame)
                cv2.imwrite(os.path.join(snapshot_dir, f"snap_vis_{snap:03d}.png"), vis)
                print("snapshot", snap); snap += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
