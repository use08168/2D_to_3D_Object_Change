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


def classify_shape(contour):
    """실루엣에서 박스/원통 추정 — **보수적(기본 원통)**.

    ⚠️ 단일 시점 한계: 서 있는 원통의 옆모습이 사각형이라 박스와 실루엣이
    사실상 동일 → 실루엣만으론 신뢰성 있게 구분 불가(다중시점이 정답).
    따라서 아주 뚜렷하게 각진 경우에만 box로 보고, 그 외에는 cylinder를 반환한다.
    확실히 아는 물체는 상위에서 shape를 직접 지정(수동 힌트)하는 것을 권장.
    """
    peri = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    if peri < 1 or area < 1:
        return "cylinder"
    n = len(cv2.approxPolyDP(contour, 0.025 * peri, True))
    rect = cv2.minAreaRect(contour)
    extent = area / max(rect[1][0] * rect[1][1], 1)
    solidity = area / max(cv2.contourArea(cv2.convexHull(contour)), 1)
    # 매우 각지고 꽉 찬(거의 완전한 사각형) 경우에만 box (오탐 최소화)
    return "box" if (n == 4 and extent > 0.96 and solidity > 0.97) else "cylinder"


def _box_corners(c, axis, length, radius):
    """축 방향 length, 단면 정사각형(2r)인 박스의 8꼭짓점."""
    a = axis / np.linalg.norm(axis)
    tmp = np.array([1, 0, 0.]) if abs(a[0]) < 0.9 else np.array([0, 1, 0.])
    e1 = np.cross(a, tmp); e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    hl = length / 2
    corners = []
    for sa in (-1, 1):
        for s1, s2 in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            corners.append(c + sa*hl*a + s1*radius*e1 + s2*radius*e2)
    return np.array(corners)   # [0-3]=한쪽 끝, [4-7]=반대 끝


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


def _cyl_mesh(center, axis, length, radius, n=24):
    a = np.asarray(axis, float); a = a / np.linalg.norm(a)
    tmp = np.array([1, 0, 0.]) if abs(a[0]) < 0.9 else np.array([0, 1, 0.])
    e1 = np.cross(a, tmp); e1 /= np.linalg.norm(e1); e2 = np.cross(a, e1)
    ang = np.linspace(0, 2*np.pi, n, endpoint=False)
    ring = np.outer(np.cos(ang), e1) + np.outer(np.sin(ang), e2)
    bot = center - a*length/2 + radius*ring
    top = center + a*length/2 + radius*ring
    V = np.vstack([bot, top, center - a*length/2, center + a*length/2])  # 2n + 2 centers
    cb, ct = 2*n, 2*n+1
    I, J, Kf = [], [], []
    for k in range(n):
        k2 = (k+1) % n
        I += [k, k2];       J += [k2, n+k2];  Kf += [n+k, n+k]      # side
        I += [cb, ct];      J += [k, n+k2];   Kf += [k2, n+k]       # caps
    return V, np.array(I), np.array(J), np.array(Kf)


def _box_mesh(center, axis, length, radius):
    V = _box_corners(center, axis, length, radius)   # 8 corners
    faces = [(0,1,2),(0,2,3),(4,5,6),(4,6,7),(0,1,5),(0,5,4),
             (1,2,6),(1,6,5),(2,3,7),(2,7,6),(3,0,4),(3,4,7)]
    I = [f[0] for f in faces]; J = [f[1] for f in faces]; Kf = [f[2] for f in faces]
    return V, np.array(I), np.array(J), np.array(Kf)


def render_plotly(objects, markers=None, ws=(240, 320), title="virtual 3D (interactive)",
                  html_path=None):
    """마우스로 회전/확대/이동 가능한 인터랙티브 3D 씬(plotly). fig 반환 + (옵션)HTML 저장.

    원통/박스는 solid mesh, 마커는 평면 위 사각형, 원점(id0) 축 표시.
    다른 기능과 접목하기 좋게 표준 plotly Figure로 반환.
    """
    import plotly.graph_objects as go
    palette = ["#00c8ff", "#00ff78", "#ffa000", "#c864ff", "#78dcff"]
    data = []
    # 작업공간 평면
    data.append(go.Mesh3d(x=[0, ws[0], ws[0], 0], y=[0, 0, ws[1], ws[1]], z=[0, 0, 0, 0],
                          i=[0, 0], j=[1, 2], k=[2, 3], color="#dddddd", opacity=0.25,
                          hoverinfo="skip", name="plane", showscale=False))
    # 원점 축
    for vec, col, nm in ([40, 0, 0], "red", "X"), ([0, 40, 0], "green", "Y"), ([0, 0, 40], "blue", "Z"):
        data.append(go.Scatter3d(x=[0, vec[0]], y=[0, vec[1]], z=[0, vec[2]], mode="lines",
                                 line=dict(color=col, width=5), hoverinfo="skip", showlegend=False))
    # 마커
    if markers:
        for mk in markers:
            cm = np.asarray(mk["corners_mm"], float)
            xs = list(cm[:, 0]) + [cm[0, 0]]; ys = list(cm[:, 1]) + [cm[0, 1]]; zs = [0]*5
            col = "red" if mk["id"] == 0 else "#888888"
            data.append(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=col, width=3),
                                     name=f"id{mk['id']}", hovertext=f"marker {mk['id']}", showlegend=False))
    # 물체
    for idx, ob in enumerate(objects):
        cyl = ob.get("cyl")
        if cyl is None:
            continue
        c, axis, length, radius = cyl
        col = palette[idx % len(palette)]
        if ob.get("shape") == "box":
            V, I, J, Kf = _box_mesh(c, axis, length, radius)
        else:
            V, I, J, Kf = _cyl_mesh(c, axis, length, radius)
        data.append(go.Mesh3d(x=V[:, 0], y=V[:, 1], z=V[:, 2], i=I, j=J, k=Kf,
                              color=col, opacity=0.85, name=f"#{idx} {ob.get('shape','')}",
                              hovertext=ob.get("label", f"#{idx}")))
    fig = go.Figure(data=data)
    fig.update_layout(title=title, showlegend=False,
                      scene=dict(xaxis_title="X(mm)", yaxis_title="Y(mm)", zaxis_title="Z up(mm)",
                                 aspectmode="data"),
                      margin=dict(l=0, r=0, t=30, b=0))
    if html_path:
        fig.write_html(html_path, include_plotlyjs="cdn")
    return fig


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
        if ob.get("shape") == "box":
            bc = _box_corners(c, axis, length, radius)
            pb = proj(bc)
            for a_, b_ in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                           (0, 4), (1, 5), (2, 6), (3, 7)]:
                cv2.line(vis, tuple(pb[a_]), tuple(pb[b_]), col, 2)
        else:
            c0 = c - axis*length/2; c1 = c + axis*length/2
            p0 = proj(_circle_pts(c0, axis, radius)); p1 = proj(_circle_pts(c1, axis, radius))
            cv2.polylines(vis, [p0], True, col, 2); cv2.polylines(vis, [p1], True, col, 2)
            for a_, b_ in zip(p0[::4], p1[::4]):
                cv2.line(vis, tuple(a_), tuple(b_), col, 1)
        cc = proj([c])[0]
        cv2.putText(vis, f"#{i}{'[box]' if ob.get('shape')=='box' else ''}", tuple(cc),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    cv2.putText(vis, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return vis
