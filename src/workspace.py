"""분산 앵커 작업공간 — 마커 지도(marker_map.json)로 카메라 로컬라이제이션.

ChArUco 보드가 하던 '평면·좌표계·스케일' 역할을 넓은 작업공간에 흩뿌린 마커 지도로 대체.
런타임엔 사진에 마커 몇 개만 보여도 지도의 알려진 3D 위치로 solvePnP → 카메라 포즈(id0 기준).
"""
from __future__ import annotations
import json
import numpy as np
import cv2

DICT = cv2.aruco.DICT_4X4_50


def load_marker_map(path):
    """marker_map.json 로드 → (corners_dict{id:(4,2)mm}, markers_list, ref_id, marker_len).

    markers_list = [{id, corners_mm, center_mm}] (scene3d 렌더용, board_marker_map과 동일 형식).
    """
    d = json.load(open(path, encoding="utf-8"))
    corners = {}
    markers_list = []
    for m in d["markers"]:
        c = np.array(m["corners_mm"], dtype=np.float64)
        corners[int(m["id"])] = c
        markers_list.append({"id": int(m["id"]), "corners_mm": c,
                             "center_mm": tuple(c.mean(0))})
    return corners, markers_list, int(d.get("ref_id", 0)), float(d.get("marker_len_mm", 22.0))


def map_extent_m(corners):
    """지도 (X,Y) 범위(m) → (xmin, ymin, xmax, ymax)."""
    allc = np.vstack(list(corners.values())) / 1000.0
    return float(allc[:, 0].min()), float(allc[:, 1].min()), float(allc[:, 0].max()), float(allc[:, 1].max())


def calibrate_from_map(image_dir, corners_map, detector=None, min_markers=6, pattern="*.jpg",
                       max_views=None):
    """마커 지도(3D 알려진)로 여러 뷰에서 카메라 내부파라미터 추정 → (K, dist, rms, n_views).

    지도 구축에 쓴 다중뷰 사진을 그대로 캘리브레이션 타겟으로 재사용(Zhang, 평면 타겟).
    주의: 마커지도 절대오차가 있으면 rms가 크게 나올 수 있음(수십 px). 근사 K와 fx가
    비슷하면 초점거리는 신뢰, cx/cy 보정만 취해도 됨.
    """
    import glob, os
    if detector is None:
        detector = make_detector()
    objpts, imgpts, sz = [], [], None
    for f in sorted(glob.glob(os.path.join(image_dir, pattern))):
        im = cv2.imread(f)
        if im is None:
            continue
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY); sz = g.shape[::-1]
        mc, ids, _ = detector.detectMarkers(g)
        if ids is None:
            continue
        o, p = [], []
        for c, i in zip(mc, ids.flatten()):
            i = int(i)
            if i in corners_map:
                o.append(np.column_stack([corners_map[i] / 1000.0, np.zeros(4)]))
                p.append(c.reshape(4, 2))
        if len(o) >= min_markers:
            objpts.append(np.vstack(o).astype(np.float32))
            imgpts.append(np.vstack(p).astype(np.float32))
    if len(objpts) < 3:
        return None
    if max_views and len(objpts) > max_views:      # 스트림 세션은 프레임이 많으므로 균등 서브샘플
        idx = np.linspace(0, len(objpts)-1, max_views).astype(int)
        objpts = [objpts[i] for i in idx]
        imgpts = [imgpts[i] for i in idx]
    rms, K, dist, _, _ = cv2.calibrateCamera(objpts, imgpts, sz, None, None)
    return K, dist, float(rms), len(objpts)


def make_detector():
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICT), params)


def localize(frame, corners_map, K, dist, detector=None, min_markers=2):
    """사진 → 지도 기준 카메라 포즈. 반환 (rvec, tvec, n_used) 또는 검출부족 시 None.

    검출된 마커 중 지도에 있는 것들의 코너(2D) ↔ 지도 3D(Z=0, m)로 solvePnP.
    """
    if detector is None:
        detector = make_detector()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    mc, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None
    obj, img = [], []
    for c, i in zip(mc, ids.flatten()):
        i = int(i)
        if i in corners_map:
            m3 = np.column_stack([corners_map[i] / 1000.0, np.zeros(4)])  # (4,3) m, Z=0
            obj.append(m3); img.append(c.reshape(4, 2))
    if len(obj) < min_markers:
        return None
    obj = np.vstack(obj).astype(np.float64)
    img = np.vstack(img).astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K.astype(np.float64), dist,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    # 재투영 오차(px) — 로컬라이즈 품질
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K.astype(np.float64), dist)
    err = float(np.linalg.norm(proj.reshape(-1, 2) - img, axis=1).mean())
    return rvec, tvec, len(obj) // 4, err
