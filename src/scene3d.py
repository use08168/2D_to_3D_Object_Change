"""검출된 물체를 가상 3D 공간에 프리미티브(원통)로 배치·렌더 (OpenCV, 빠름).

물체별 3D 점군(보드좌표계 mm)에서 방향성 원통(center, axis, length, radius)을 PCA로 적합.
가상 카메라(3/4 시점)로 작업공간 평면 격자 + 원통 와이어프레임을 투영해 그린다.
실시간 루프에서 매 프레임 재렌더 → 물체를 놓으면 생기고 치우면 사라짐(add/remove).
"""
from __future__ import annotations
import numpy as np
import cv2


def fit_cylinder(P_mm):
    """점군(N,3, mm) → 방향성 원통 (center, axis, length_mm, radius_mm)."""
    P = P_mm[np.isfinite(P_mm).all(1)]
    if len(P) < 10:
        return None
    c0 = np.median(P, axis=0)
    # 이상치 제거: 중앙값에서 먼 점 컷
    dist = np.linalg.norm(P - c0, axis=1)
    P = P[dist < np.percentile(dist, 97)]
    c = P.mean(0)
    _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
    axis = Vt[0]
    tproj = (P - c) @ axis
    lo, hi = np.percentile(tproj, [2, 98])          # robust 길이
    c = c + axis * ((hi + lo) / 2)
    length = float(hi - lo)
    perp = (P - c) - np.outer((P - c) @ axis, axis)
    radius = float(np.percentile(np.linalg.norm(perp, axis=1), 80))
    return c, axis, length, max(radius, 2.0)


def _lookat(eye, center, Kv):
    f = center - eye; f = f / np.linalg.norm(f)
    r = np.cross(f, np.array([0, 0, 1.0])); r = r / np.linalg.norm(r)
    u = np.cross(r, f)
    Rwc = np.stack([r, -u, f])
    rvec, _ = cv2.Rodrigues(Rwc)
    tvec = (-Rwc @ eye).reshape(3, 1)
    return rvec, tvec


def _circle_pts(c, axis, radius, n=24):
    a = axis / np.linalg.norm(axis)
    tmp = np.array([1, 0, 0.]) if abs(a[0]) < 0.9 else np.array([0, 1, 0.])
    e1 = np.cross(a, tmp); e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    ang = np.linspace(0, 2*np.pi, n, endpoint=False)
    return c + radius*(np.outer(np.cos(ang), e1) + np.outer(np.sin(ang), e2))


def render_virtual_scene(objects, markers=None, img_size=(720, 720), ws=(220, 300),
                         az=45.0, el=28.0, title="virtual 3D scene"):
    """objects[i]['cyl']=(center,axis,length,radius)[mm] 를 가상 3D로 렌더 → BGR 이미지.

    markers: [{'id':int, 'center_mm':(x,y), 'corners_mm':(4,2)}] 있으면 평면에 마커 위치 표시.
    """
    W, H = img_size
    vis = np.full((H, W, 3), 28, np.uint8)
    Kv = np.array([[W*0.9, 0, W/2], [0, W*0.9, H/2], [0, 0, 1.]])
    center = np.array([ws[0]/2, ws[1]/2, 35.])
    dist = max(ws) * 2.4
    ar, er = np.radians(az), np.radians(el)
    eye = center + dist*np.array([np.cos(er)*np.cos(ar), np.cos(er)*np.sin(ar), np.sin(er)])
    rvec, tvec = _lookat(eye, center, Kv)

    def proj(P):
        return cv2.projectPoints(np.asarray(P, float), rvec, tvec, Kv, None)[0].reshape(-1, 2).astype(int)

    # 작업공간 평면 격자
    nx, ny = 6, 8
    for gx in np.linspace(0, ws[0], nx+1):
        p = proj([[gx, 0, 0], [gx, ws[1], 0]]); cv2.line(vis, tuple(p[0]), tuple(p[1]), (65, 65, 65), 1)
    for gy in np.linspace(0, ws[1], ny+1):
        p = proj([[0, gy, 0], [ws[0], gy, 0]]); cv2.line(vis, tuple(p[0]), tuple(p[1]), (65, 65, 65), 1)
    # 원점 축 (id0)
    o = proj([[0, 0, 0], [45, 0, 0], [0, 45, 0], [0, 0, 45]])
    cv2.line(vis, tuple(o[0]), tuple(o[1]), (0, 0, 255), 2)
    cv2.line(vis, tuple(o[0]), tuple(o[2]), (0, 255, 0), 2)
    cv2.line(vis, tuple(o[0]), tuple(o[3]), (255, 60, 0), 2)

    # ArUco 마커 위치 (평면 위 작은 사각형 + ID)
    if markers:
        for mk in markers:
            cm = np.asarray(mk["corners_mm"], float)
            poly = np.c_[cm, np.zeros(len(cm))]      # z=0 평면
            pp = proj(poly)
            col = (0, 0, 255) if mk["id"] == 0 else (180, 180, 180)  # id0=빨강(REF)
            cv2.polylines(vis, [pp], True, col, 2)
            cx = int(pp[:, 0].mean()); cy = int(pp[:, 1].mean())
            cv2.putText(vis, str(mk["id"]), (cx-6, cy+4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

    colors = [(0, 200, 255), (0, 255, 120), (255, 160, 0), (200, 100, 255), (120, 220, 255)]
    for i, ob in enumerate(objects):
        cyl = ob.get("cyl")
        if cyl is None:
            continue
        c, axis, length, radius = cyl
        col = colors[i % len(colors)]
        c0 = c - axis*length/2; c1 = c + axis*length/2
        p0 = proj(_circle_pts(c0, axis, radius)); p1 = proj(_circle_pts(c1, axis, radius))
        cv2.polylines(vis, [p0], True, col, 2); cv2.polylines(vis, [p1], True, col, 2)
        for a_, b_ in zip(p0[::4], p1[::4]):
            cv2.line(vis, tuple(a_), tuple(b_), col, 1)
        cc = proj([c])[0]
        cv2.putText(vis, f"#{i}", tuple(cc), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    cv2.putText(vis, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return vis
