"""단안 깊이(Depth Anything V2) + ArUco 평면 앵커링으로 웹캠 사진 한 장에서
물체별 높이·바닥치수·대략 부피를 추정 (RGB-D 없이).

원리:
- Depth Anything V2 = 상대(affine-invariant) 깊이 추정. 절대 미터가 아님.
- 그러나 ArUco로 '보드평면의 진짜 미터 깊이'를 알므로, 보드 픽셀에서
  1/Z_true = A*pred + B 를 피팅해 상대깊이를 미터로 앵커링(보정).
- 각 픽셀을 3D로 복원 → 보드평면 위 '높이'맵 계산.
- 높이 > 임계 영역 = 물체(평면은 높이≈0이라 자동 제외) → 물체별 측정.

한계(솔직): DA는 '측정'이 아니라 '추정'이라 높이가 과소평가되기 쉽고, 가려진 뒷면은 모름.
근사 바운딩박스/부피 수준. 정밀은 RGB-D/다중시점 필요.
"""
from __future__ import annotations
import numpy as np
import cv2
import aruco_utils as au


def load_depth_model(model_id="depth-anything/Depth-Anything-V2-Small-hf", device=0):
    from transformers import pipeline
    return pipeline("depth-estimation", model=model_id, device=device)


def height_map_from_depth(frame, pipe, board, K, dist, square_len=0.038, squares_xy=(5, 7)):
    """왜곡보정→보드자세→DA깊이→평면앵커링→보드평면 위 높이맵(mm).

    반환 dict(imgu, height_mm, region, quad, rvec, tvec, r2) 또는 보드 미검출 시 None.
    """
    K = K.astype(np.float64)
    imgu = cv2.undistort(frame, K, dist)
    H, W = imgu.shape[:2]
    z0 = np.zeros((5, 1))
    gray = cv2.cvtColor(imgu, cv2.COLOR_BGR2GRAY)
    rvec, tvec, cc, ci = au.detect_charuco_pose(gray, board, K, z0)
    if rvec is None:
        return None
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    n_cam = R[:, 2]
    Kinv = np.linalg.inv(K)
    sx, sy = squares_xy

    def proj(P):
        return cv2.projectPoints(np.asarray(P, np.float64), rvec, tvec, K, z0)[0].reshape(-1, 2)
    quad = proj([[0, 0, 0], [sx*square_len, 0, 0], [sx*square_len, sy*square_len, 0], [0, sy*square_len, 0]]).astype(np.int32)
    region = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(region, quad, 255)

    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    pix = np.stack([uu.ravel(), vv.ravel(), np.ones(W*H)], 0).astype(np.float64)
    dirs = Kinv @ pix
    ndir = n_cam @ dirs
    Zplane = ((n_cam @ t) / ndir * dirs[2]).reshape(H, W)

    from PIL import Image
    pred = pipe(Image.fromarray(cv2.cvtColor(imgu, cv2.COLOR_BGR2RGB)))["predicted_depth"]
    pred = cv2.resize(pred.squeeze().cpu().numpy().astype(np.float64), (W, H))

    m = (region > 0) & np.isfinite(Zplane) & (Zplane > 0.05) & (Zplane < 3)
    A, B = np.polyfit(pred[m], 1.0/Zplane[m], 1)
    yhat = A*pred[m] + B
    r2 = 1 - np.sum((1.0/Zplane[m]-yhat)**2)/np.sum((1.0/Zplane[m]-(1.0/Zplane[m]).mean())**2)

    invZ = A*pred + B
    Zmet = np.where(invZ > 1e-6, 1.0/invZ, np.nan)
    Pcam = dirs * (Zmet.reshape(-1)/dirs[2])
    up = np.sign(-(n_cam @ t))
    height_mm = (up*(n_cam @ (Pcam - t.reshape(3, 1)))).reshape(H, W) * 1000.0
    # 보드(작업공간) 좌표계 3D 점군: P_board = R^T (P_cam - t). z축이 '위'.
    Rt = R.T
    pts_board = (Rt @ (Pcam - t.reshape(3, 1))).T.reshape(H, W, 3)
    pts_board[..., 2] *= up   # 위 방향 +로 통일
    # 주의: 보드 밖을 0으로 자르지 않음 → 보드 위로 솟은 큰 물체가 잘리지 않게.
    return {"imgu": imgu, "height_mm": height_mm, "region": region, "quad": quad,
            "pts_board": pts_board, "rvec": rvec, "tvec": tvec, "r2": float(r2)}


def detect_objects_by_height(hm, K, dist, min_height_mm=6, min_area_px=1500,
                             open_px=7, close_px=15):
    """높이맵에서 솟은 영역(=물체)을 검출하고 물체별 치수/부피 추정.

    objects: [{bbox, footprint_mm(w,l), height_max_mm, height_mean_mm, bbox_vol_cm3}]
    """
    height_mm, region = hm["height_mm"], hm["region"]
    rvec, tvec = hm["rvec"], hm["tvec"]
    z0 = np.zeros((5, 1))
    obj = ((height_mm > min_height_mm) & (region > 0)).astype(np.uint8)*255
    obj = cv2.morphologyEx(obj, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px)))
    obj = cv2.morphologyEx(obj, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px)))
    nc, lab, stats, _ = cv2.connectedComponentsWithStats(obj)
    objs = []
    for i in range(1, nc):
        if stats[i, cv2.CC_STAT_AREA] < min_area_px:
            continue
        comp = lab == i
        x, y, w, h = stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3]
        hmax = float(np.nanmax(height_mm[comp])); hmean = float(np.nanmean(height_mm[comp]))
        ys, xs = np.where(comp)
        order = np.argsort(ys)[-max(30, int(0.15*len(ys))):]
        basepx = np.stack([xs[order], ys[order]], 1).astype(np.float64)
        bp = au.pixels_to_plane(basepx, K.astype(np.float64), z0, rvec, tvec)[:, :2]
        rect = cv2.minAreaRect(bp.astype(np.float32))
        fw, fl = rect[1][0]*1000, rect[1][1]*1000
        objs.append({"bbox": (int(x), int(y), int(w), int(h)),
                     "footprint_mm": (float(fw), float(fl)),
                     "height_max_mm": hmax, "height_mean_mm": hmean,
                     "bbox_vol_cm3": float(fw*fl*hmax/1000.0)})
    return objs


def draw_depth_objects(hm, objs):
    vis = hm["imgu"].copy()
    cv2.polylines(vis, [hm["quad"]], True, (0, 0, 255), 2)
    for i, o in enumerate(objs):
        x, y, w, h = o["bbox"]; fw, fl = o["footprint_mm"]
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(vis, f"#{i} H={o['height_max_mm']:.0f}mm {fw:.0f}x{fl:.0f}",
                    (x, max(18, y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(vis, f"objects: {len(objs)}  (R^2={hm['r2']:.2f})",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return vis


def colorize_height(hm, hmax_mm=60):
    hv = np.clip(hm["height_mm"], 0, hmax_mm)/hmax_mm*255
    return cv2.applyColorMap(hv.astype(np.uint8), cv2.COLORMAP_JET)
