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


def _build_object(b, hmm, pts, contour, max_dim_mm=250.0):
    """물체 마스크 b → (type, cyl, label). 축은 분류에 고정(DA 기울어짐 방지).

    타당성 필터(max_dim_mm): 넓은/비스듬한 뷰에서 DA 상대깊이가 드리프트해 먼 배경이
    '납작한 거대 덩어리'로 잡히는 걸 배제. 실제 손물체는 어느 치수도 max_dim_mm 이하이고,
    서 있는 물체는 높이 > 지름(길쭉)이라 오블레이트(지름≫높이) 블롭을 거른다.
    """
    da_h = float(np.nanmax(hmm[b]))
    P = pts[b] * 1000.0
    P = P[np.isfinite(P).all(1)]
    if len(P) < 10:
        return None
    z = P[:, 2]; zlo, zhi = np.percentile(z, [2, 98]); Lz = zhi - zlo   # 세로(높이)
    up = P[z > zlo + 0.3*Lz]                                            # 밑동 그림자/스커트 제외
    if len(up) < 10:
        up = P
    xspan = np.percentile(up[:, 0], 98) - np.percentile(up[:, 0], 2)
    yspan = np.percentile(up[:, 1], 98) - np.percentile(up[:, 1], 2)
    foot = max(xspan, yspan)                                            # 가로(윗부분 최대폭)
    # 서 있음 = 최소 높이 넘고 세로가 가로만큼 큼(누운 박스는 가로≫세로라 여기서 걸림)
    if da_h > STAND_TH_MM and Lz >= 0.7*foot:
        # 중심·지름은 '중간 높이 슬라이스'에서만 → 밑동에 붙은 낮은 배경 띠 배제
        mid = (z > zlo + 0.3*Lz) & (z < zlo + 0.9*Lz)
        Pm = P[mid] if mid.sum() >= 10 else P
        cx, cy = Pm[:, 0].mean(), Pm[:, 1].mean()
        r = max(float(np.percentile(np.hypot(Pm[:, 0]-cx, Pm[:, 1]-cy), 80)), 2.0)
        if 2*r > max_dim_mm:                       # 절대 크기 상한(비정상 거대)
            return None
        cyl = (np.array([cx, cy, (zlo+zhi)/2]), np.array([0., 0., 1.]), float(Lz), r)
        return "stand", cyl, f"H{Lz:.0f} D~{2*r:.0f}mm"
    cyl = s3.fit_cylinder(P)
    if cyl is None:
        return None
    c, ax, L, r = cyl
    if L > max_dim_mm or 2*r > max_dim_mm:         # 절대 크기 상한
        return None
    axp = np.array([ax[0], ax[1], 0.0]); n = np.linalg.norm(axp)
    axp = axp/n if n > 1e-6 else np.array([1., 0., 0.])
    cyl = (np.array([c[0], c[1], r]), axp, L, r)
    return "lie", cyl, f"L{L:.0f} W{2*r:.0f}mm"


def _tophat_local(hmm, H, W, bg_ksize=121, bg_scale=0.25):
    """높이맵의 국소 배경(롤링볼=다운스케일 grayscale opening)을 빼 국소 대비(top-hat) 반환.

    주변보다 솟은 곳만 남김 → 낮은 물체도 포착, 저주파 평면 드리프트는 제거.
    """
    hf = np.clip(np.where(np.isfinite(hmm), hmm, 0.0), -20, 500).astype(np.float32)
    hs = cv2.resize(hf, (max(1, int(W*bg_scale)), max(1, int(H*bg_scale))), interpolation=cv2.INTER_AREA)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bg_ksize, bg_ksize))
    bg = cv2.resize(cv2.morphologyEx(hs, cv2.MORPH_OPEN, se), (W, H), interpolation=cv2.INTER_LINEAR)
    return np.clip(hf - bg, 0, None)


def process_frame_da(frame, pipe, board, K, dist, square_len=0.038,
                     min_height_mm=6, min_area_px=1500, squares_xy=(5, 7),
                     xy_margin_mm=40.0, marker_map=None, shape_mode="auto",
                     pose=None, plane_xyxy=None, detect="tophat",
                     tophat_mm=4.0, peak_min_mm=10.0, spread_min_mm=6.0,
                     bg_ksize=121, bg_scale=0.25):
    """DA 단독 한 프레임 → (vis, objects, markers). 미검출 시 (원본, [], []).

    분산앵커: pose=(rvec,tvec)+plane_xyxy(작업공간 범위 m)+marker_map을 주면 보드 대신 사용.
    검출(detect): "tophat"=국소대비(낮은물체·드리프트강인, 평평함검증) | "abs"=절대높이임계(구식).
      tophat_mm=국소대비 임계, peak_min_mm/spread_min_mm=봉우리 검증(평평한 아티팩트 거부),
      xy_margin_mm 음수=둘레 안쪽 축소(보드 가장자리 들뜸 배제).
    """
    hm = dv.height_map_from_depth(frame, pipe, board, K, dist, square_len=square_len,
                                  pose=pose, plane_xyxy=plane_xyxy)
    if hm is None:
        v = frame.copy()
        cv2.putText(v, "board/markers not found", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return v, [], []
    imgu = hm["imgu"]; H, W = imgu.shape[:2]
    hmm = hm["height_mm"]; region = hm["region"]; pts = hm["pts_board"]

    # 이미지 영역이 아니라 3D (X,Y)로 거름: 물체는 서 있어도 (X,Y)가 작업평면 위에 투영됨.
    qm = region > 0
    Xb = pts[:, :, 0] * 1000.0; Yb = pts[:, :, 1] * 1000.0    # 작업평면 좌표 mm
    if plane_xyxy is not None:
        x0, y0, x1, y1 = (v*1000.0 for v in plane_xyxy)
    else:
        x0, y0 = 0.0, 0.0
        x1 = squares_xy[0] * square_len * 1000.0
        y1 = squares_xy[1] * square_len * 1000.0
    mg = xy_margin_mm
    on_xy = (np.isfinite(Xb) & (Xb > x0-mg) & (Xb < x1+mg) & (Yb > y0-mg) & (Yb < y1+mg))
    if detect == "tophat":
        th = _tophat_local(hmm, H, W, bg_ksize, bg_scale)      # 국소 대비
        raised = (th > tophat_mm) & on_xy & np.isfinite(hmm) & (hmm < 500)
    else:
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
        if detect == "tophat":                           # 봉우리 검증: 평평한 아티팩트 거부
            hb = hmm[b]; hb = hb[np.isfinite(hb)]
            if hb.size == 0:
                continue
            peak = float(np.percentile(hb, 85))
            if peak < peak_min_mm or peak - float(np.median(hb)) < spread_min_mm:
                continue
        else:
            ys, xs = np.where(b)                          # 바닥밴드가 보드 위인지
            yy0, yy1 = ys.min(), ys.max()
            band = b & (np.arange(H)[:, None] >= yy1 - max(3, int(0.15*(yy1-yy0))))
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


def process_frame_workspace(frame, pipe, K, dist, corners_map, markers_list, plane_xyxy,
                            detector=None, **proc_kw):
    """분산앵커 한 프레임: 로컬라이즈(마커지도) → DA 검출. → (vis, objs, ok, n, err).

    보드 검출 대신 marker_map으로 매 프레임 solvePnP. 로컬라이즈 실패 시 ok=False.
    """
    import workspace as _wsp
    if detector is None:
        detector = _wsp.make_detector()
    proc_kw.setdefault("xy_margin_mm", -35.0)     # 작업공간: 둘레 안쪽 축소(가장자리 들뜸 배제)
    loc = _wsp.localize(frame, corners_map, K, dist, detector)
    if loc is None:
        v = frame.copy()
        cv2.putText(v, "localizing... (need >=2 mapped markers)", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return v, [], False, 0, float("nan"), None
    rvec, tvec, n, err = loc
    vis, objs, _ = process_frame_da(frame, pipe, None, K, dist, pose=(rvec, tvec),
                                    plane_xyxy=plane_xyxy, marker_map=markers_list, **proc_kw)
    cv2.putText(vis, f"localized: {n} markers  reproj {err:.1f}px", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return vis, objs, True, n, err, (rvec, tvec)


def run_live_workspace(K, dist, corners_map, markers_list, plane_xyxy, pipe=None,
                       cam_index=0, calib_wh=(1920, 1080), snapshot_dir=".", **proc_kw):
    """분산앵커 작업공간 실시간: 매 프레임 마커지도 로컬라이즈 → DA 검출 → 카메라 | 가상3D.

    보드가 아니라 넓은 작업공간(marker_map). [s]스냅 [q]종료.
    """
    import workspace as _wsp
    if pipe is None:
        pipe = dv.load_depth_model()
    detector = _wsp.make_detector()
    x0, y0, x1, y1 = plane_xyxy
    origin = (x0*1000.0, y0*1000.0)
    ws_sz = ((x1-x0)*1000.0 + 60, (y1-y0)*1000.0 + 60)
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
            vis, objs, loc_ok, n, err, pose = process_frame_workspace(
                frame, pipe, K, dist, corners_map, markers_list, plane_xyxy,
                detector=detector, **proc_kw)
            scene = s3.render_virtual_scene(objs, markers=markers_list, ws=ws_sz, origin_mm=origin,
                                            cam_pose=pose, plane_xyxy=plane_xyxy)
            camh = cv2.resize(vis, (int(vis.shape[1]*scene.shape[0]/vis.shape[0]), scene.shape[0]))
            view = np.hstack([camh, scene])
            cv2.imshow("workspace live: camera | virtual 3D  [s]snap [q]quit", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                cv2.imwrite(os.path.join(snapshot_dir, f"ws_raw_{snap:03d}.png"), frame)
                cv2.imwrite(os.path.join(snapshot_dir, f"ws_vis_{snap:03d}.png"), view)
                html = os.path.join(snapshot_dir, f"ws_3d_{snap:03d}.html")   # 마우스로 회전/확대/이동
                s3.render_plotly(objs, markers=markers_list, ws=ws_sz, origin_mm=origin, cam_pose=pose,
                                 title=f"workspace 3D (snap {snap})", html_path=html, open_browser=True)
                print(f"snapshot {snap}  (interactive: {html})"); snap += 1
    finally:
        cap.release(); cv2.destroyAllWindows()
