"""DA 단독 실시간 파이프라인 — FastSAM 없이 Depth Anything V2만으로.

검출: DA 높이맵(ArUco 앵커링)에서 '솟은 영역' → 물체(색·참조 불필요, 클래스 무관).
크기/자세: 물체 3D 점군으로 원통 적합 + 분류(수직/누움). 크기는 DA-only라 09보다 약간 거침.
장점: FastSAM 제거로 더 빠름(~1.7fps 동기), 검출 선별이 깔끔(솟음 기준).
표시: 카메라 합성(높이 히트맵) | 가상 3D 씬(원통 + 전체 마커 지도).
"""
from __future__ import annotations
import os
import numpy as np
import cv2
import aruco_utils as au
import depth_volume as dv
import scene3d as s3

STAND_TH_MM = 25.0


def _build_object(b, hmm, pts, contour):
    """물체 마스크 b → (type, cyl, label). 축은 분류에 고정(DA 기울어짐 방지)."""
    da_h = float(np.nanmax(hmm[b]))
    P = pts[b] * 1000.0
    P = P[np.isfinite(P).all(1)]
    if len(P) < 10:
        return None
    if da_h > STAND_TH_MM:                        # 선 물체 → 수직 스냅
        z = P[:, 2]; zlo, zhi = np.percentile(z, [2, 98]); L = zhi - zlo
        # 중심·지름은 '중간 높이 슬라이스'에서만 → 밑동에 붙은 낮은 배경 띠 배제
        mid = (z > zlo + 0.3*L) & (z < zlo + 0.9*L)
        Pm = P[mid] if mid.sum() >= 10 else P
        cx, cy = Pm[:, 0].mean(), Pm[:, 1].mean()
        r = max(float(np.percentile(np.hypot(Pm[:, 0]-cx, Pm[:, 1]-cy), 80)), 2.0)
        cyl = (np.array([cx, cy, (zlo+zhi)/2]), np.array([0., 0., 1.]), float(L), r)
        return "stand", cyl, f"H{L:.0f} D~{2*r:.0f}mm"
    cyl = s3.fit_cylinder(P)
    if cyl is None:
        return None
    c, ax, L, r = cyl
    axp = np.array([ax[0], ax[1], 0.0]); n = np.linalg.norm(axp)
    axp = axp/n if n > 1e-6 else np.array([1., 0., 0.])
    cyl = (np.array([c[0], c[1], r]), axp, L, r)
    return "lie", cyl, f"L{L:.0f} W{2*r:.0f}mm"


def process_frame_da(frame, pipe, board, K, dist, square_len=0.038,
                     min_height_mm=6, min_area_px=1500, squares_xy=(5, 7),
                     xy_margin_mm=40.0, marker_map=None, shape_mode="auto"):
    """DA 단독 한 프레임 → (vis, objects, markers). 보드 미검출 시 (원본, [], [])."""
    hm = dv.height_map_from_depth(frame, pipe, board, K, dist, square_len=square_len)
    if hm is None:
        v = frame.copy()
        cv2.putText(v, "board not found", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return v, [], []
    imgu = hm["imgu"]; H, W = imgu.shape[:2]
    hmm = hm["height_mm"]; region = hm["region"]; pts = hm["pts_board"]

    # 이미지 영역이 아니라 3D (X,Y)로 거름: 물체는 서 있어도 (X,Y)가 보드 위에 투영되지만
    # 키보드 등 배경은 (X,Y)가 보드 밖 → 이미지상 어디로 뻗든 물체만 통째로 잡힘.
    qm = region > 0
    Xb = pts[:, :, 0] * 1000.0; Yb = pts[:, :, 1] * 1000.0    # 보드좌표 mm
    Wb = squares_xy[0] * square_len * 1000.0
    Hb = squares_xy[1] * square_len * 1000.0
    mg = xy_margin_mm
    on_xy = (np.isfinite(Xb) & (Xb > -mg) & (Xb < Wb + mg) & (Yb > -mg) & (Yb < Hb + mg))
    raised = np.isfinite(hmm) & (hmm > min_height_mm) & (hmm < 500) & on_xy
    m = raised.astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    nc, lab, stats, _ = cv2.connectedComponentsWithStats(m)

    objects = []
    objunion = np.zeros((H, W), bool)
    for i in range(1, nc):
        a = stats[i, cv2.CC_STAT_AREA]
        if a < min_area_px or a > 0.30 * H * W:         # 배경 합쳐진 거대 덩어리 제외
            continue
        b = lab == i
        ys, xs = np.where(b)                             # 바닥밴드가 보드 위인지
        y0, y1 = ys.min(), ys.max()
        band = b & (np.arange(H)[:, None] >= y1 - max(3, int(0.15*(y1-y0))))
        if band.sum() == 0 or (band & qm).sum() / band.sum() < 0.35:
            continue
        cnts, _ = cv2.findContours((b.astype(np.uint8))*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        built = _build_object(b, hmm, pts, c)
        if built is None:
            continue
        typ, cyl, label = built
        x, y, w, h = cv2.boundingRect(c)
        objects.append({"contour": c, "bbox": (x, y, w, h), "type": typ, "cyl": cyl,
                        "label": label, "shape": s3.classify_shape(c) if shape_mode == "auto" else shape_mode})
        objunion |= b

    vis = imgu.copy()
    hcolor = dv.colorize_height(hm, 60)
    if objunion.any():
        vis[objunion] = (0.45*imgu[objunion] + 0.55*hcolor[objunion]).astype(np.uint8)
    cv2.drawFrameAxes(vis, K.astype(np.float64), dist*0, hm["rvec"], hm["tvec"], square_len*2, 2)
    cv2.polylines(vis, [hm["quad"]], True, (0, 0, 255), 1)
    for idx, o in enumerate(objects):
        cv2.drawContours(vis, [o["contour"]], -1, (0, 255, 0), 2)
        x, y = o["bbox"][0], o["bbox"][1]
        cv2.putText(vis, f"#{idx} {o['type']} {o['label']}", (x, max(16, y-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(vis, f"[DA only] objects: {len(objects)}  anchor R^2={hm['r2']:.2f}",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    iw, ih = W//5, H//5
    small = cv2.resize(hcolor, (iw, ih)); vis[10:10+ih, W-iw-10:W-10] = small
    markers = marker_map if marker_map is not None else au.board_marker_map(board)
    return vis, objects, markers


def run_live_da(K, dist, board, pipe=None, square_len=0.038, cam_index=0,
                calib_wh=(1920, 1080), ws=(240, 320), snapshot_dir=".", **proc_kw):
    """DA 단독 실시간: 카메라 합성 | 가상 3D 씬. [s]스냅 [q]종료. (FastSAM 없음)"""
    if pipe is None:
        pipe = dv.load_depth_model()
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, calib_wh[0]); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, calib_wh[1])
    if not cap.isOpened():
        raise RuntimeError(f"웹캠({cam_index}) 열기 실패")
    snap = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            vis, objs, markers = process_frame_da(frame, pipe, board, K, dist,
                                                  square_len=square_len, **proc_kw)
            scene = s3.render_virtual_scene(objs, markers=markers, ws=ws)
            camh = cv2.resize(vis, (int(vis.shape[1]*scene.shape[0]/vis.shape[0]), scene.shape[0]))
            view = np.hstack([camh, scene])
            cv2.imshow("DA-only live: camera | virtual 3D  [s]snap [q]quit", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                cv2.imwrite(os.path.join(snapshot_dir, f"snap_raw_{snap:03d}.png"), frame)
                cv2.imwrite(os.path.join(snapshot_dir, f"snap_vis_{snap:03d}.png"), view)
                print("snapshot", snap); snap += 1
    finally:
        cap.release(); cv2.destroyAllWindows()
