"""ArUco 기반 단일카메라 마커 3D 로컬라이제이션 유틸.

핵심 아이디어:
- 각 ArUco 마커(한 변 길이 L)의 4개 코너 3D 좌표를 알고 있으므로, solvePnP로
  카메라 기준 각 마커의 6-DoF 포즈(rvec, tvec)를 구한다.
- 기준 마커(예: id0) 포즈의 역변환을 곱해 모든 마커를 "id0 좌표계"로 옮긴다.
- 카메라 내부파라미터가 없으면 이미지 크기+화각으로 근사 K를 만들어 쓴다(무보정 단일 사진).
"""
from __future__ import annotations
import numpy as np
import cv2


# ---------------------------------------------------------------------------
# 카메라 파라미터
# ---------------------------------------------------------------------------
def approx_camera_matrix(image_size, hfov_deg: float = 60.0):
    """이미지 크기 (w, h)와 수평 화각으로 근사 카메라 행렬 K 생성.

    캘리브레이션 없이 쓰는 근사값. 깊이 정확도는 화각 가정에 민감하다.
    일반 웹캠/폰 후면 ~60도. 알고 있으면 실제 값을 넣을 것.
    """
    w, h = int(image_size[0]), int(image_size[1])
    f = 0.5 * w / np.tan(np.deg2rad(hfov_deg) / 2.0)
    K = np.array([[f, 0, w / 2.0],
                  [0, f, h / 2.0],
                  [0, 0, 1.0]], dtype=np.float64)
    dist = np.zeros((5, 1), dtype=np.float64)
    return K, dist


def load_intrinsics(npz_path):
    """내부파라미터 npz 로드 → (K, dist). 두 키 규약 모두 지원.

    - ChArUco(01): camera_matrix / dist_coeffs
    - workspace(calibrate_from_map): K / dist
    """
    data = np.load(npz_path)
    kkey = "camera_matrix" if "camera_matrix" in data else "K"
    dkey = "dist_coeffs" if "dist_coeffs" in data else "dist"
    return data[kkey].astype(np.float64), data[dkey].astype(np.float64)


# ---------------------------------------------------------------------------
# 마커 검출 & 포즈
# ---------------------------------------------------------------------------
def make_detector(dict_id: int = cv2.aruco.DICT_4X4_50):
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    params = cv2.aruco.DetectorParameters()
    # 서브픽셀 코너 정밀화 → 포즈 안정성 향상
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(dictionary, params)


def marker_object_points(L: float) -> np.ndarray:
    """중심이 원점, z가 마커 바깥을 향하는 마커 좌표계에서의 4코너 3D 좌표.

    OpenCV aruco 코너 순서(좌상→우상→우하→좌하)와 일치.
    """
    return np.array([[-L / 2,  L / 2, 0.0],
                     [ L / 2,  L / 2, 0.0],
                     [ L / 2, -L / 2, 0.0],
                     [-L / 2, -L / 2, 0.0]], dtype=np.float64)


def estimate_marker_poses(corners, ids, K, dist, marker_length: float):
    """검출된 각 마커에 대해 카메라 기준 포즈 추정.

    반환: {id(int): {"rvec","tvec","T"(4x4 camera_from_marker)}}
    estimatePoseSingleMarkers는 OpenCV 4.7+에서 제거되어 solvePnP로 직접 계산.
    """
    objp = marker_object_points(marker_length)
    poses = {}
    if ids is None:
        return poses
    for c, i in zip(corners, ids.flatten()):
        img_pts = c.reshape(-1, 2).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(objp, img_pts, K, dist,
                                      flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            continue
        poses[int(i)] = {"rvec": rvec, "tvec": tvec, "T": rt_to_matrix(rvec, tvec)}
    return poses


# ---------------------------------------------------------------------------
# 변환 유틸
# ---------------------------------------------------------------------------
def rt_to_matrix(rvec, tvec) -> np.ndarray:
    """(rvec, tvec) → 4x4 동차변환 T (marker점 → camera점)."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).ravel()
    return T


def detect_charuco_pose(gray, board, K, dist):
    """ChArUco 보드를 검출해 카메라 기준 보드 포즈(rvec, tvec) 반환.

    반환: (rvec, tvec, charuco_corners, charuco_ids) 또는 검출 실패 시 (None,...).
    rvec/tvec 는 camera_from_board (보드 좌표계 점 → 카메라 좌표계).
    """
    detector = cv2.aruco.CharucoDetector(board)
    ch_corners, ch_ids, _, _ = detector.detectBoard(gray)
    if ch_corners is None or ch_ids is None or len(ch_corners) < 6:
        return None, None, ch_corners, ch_ids
    obj_pts, img_pts = board.matchImagePoints(ch_corners, ch_ids)
    if obj_pts is None or len(obj_pts) < 6:
        return None, None, ch_corners, ch_ids
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist)
    if not ok:
        return None, None, ch_corners, ch_ids
    return rvec, tvec, ch_corners, ch_ids


def pixels_to_plane(pixels, K, dist, rvec, tvec):
    """이미지 픽셀들을 보드 평면(z=0, 보드 좌표계)으로 역투영.

    왜곡 보정 → 카메라 광선 → 보드 평면과 교차. 반환 (N,3), z≈0.
    물체 바닥 윤곽 픽셀을 넣으면 실제 mm(보드 단위) 좌표가 나온다.
    """
    pts = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pts, K, dist).reshape(-1, 2)  # 정규화 좌표(z=1)
    R, _ = cv2.Rodrigues(rvec)
    Rt = R.T
    C = (-Rt @ np.asarray(tvec, dtype=np.float64).reshape(3))  # 보드 좌표계 카메라 중심
    D = np.column_stack([und, np.ones(len(und))])   # (N,3) 정규화 광선
    Dw = D @ Rt.T                                    # (N,3) 보드좌표계 광선 방향 (벡터화)
    s = -C[2] / Dw[:, 2]                             # z=0 까지
    return C[None, :] + s[:, None] * Dw


def height_from_vertical_edge(base_px, top_px, K, dist, rvec, tvec):
    """물체의 수직 모서리(바닥 픽셀, 꼭대기 픽셀)로 높이[m] 추정.

    바닥점을 평면에 투영해 (x0,y0,0)을 얻고, 꼭대기 픽셀 광선이
    수직선 x=x0,y=y0 와 만나는 높이를 최소자승으로 구한다(보정된 카메라 가정).
    """
    P0 = pixels_to_plane([base_px], K, dist, rvec, tvec)[0]
    x0, y0 = P0[0], P0[1]
    und = cv2.undistortPoints(np.array([[top_px]], np.float64), K, dist).reshape(2)
    R, _ = cv2.Rodrigues(rvec)
    Rt = R.T
    C = (-Rt @ np.asarray(tvec, dtype=np.float64).reshape(3))
    d = Rt @ np.array([und[0], und[1], 1.0])
    # C_xy + s d_xy ≈ (x0,y0) 를 만족하는 s (least squares)
    dxy = d[:2]
    s = float(dxy @ (np.array([x0, y0]) - C[:2]) / (dxy @ dxy))
    h = C[2] + s * d[2]
    return abs(h), P0


def board_marker_map(board):
    """보드 정의에서 '전체 마커 지도'를 산출(검출과 무관, 항상 전부).

    반환 [{'id', 'corners_mm'(4,2), 'center_mm'(x,y)}] — 보드 XY평면 mm.
    분산 앵커로 확장 시엔 매핑 단계에서 만든 map(동일 형식)을 저장/로드해 쓴다.
    """
    ids = np.array(board.getIds()).flatten()
    objp = board.getObjPoints()
    out = []
    for i, corners in zip(ids, objp):
        c = np.array(corners, dtype=np.float64)[:, :2] * 1000.0   # XY mm
        out.append({"id": int(i), "corners_mm": c, "center_mm": tuple(c.mean(0))})
    return out


def board_region_mask(shape, charuco_corners, dilate_px: int = 25):
    """검출된 charuco 코너들의 볼록껍질로 보드 영역 마스크 생성(약간 팽창).

    물체가 이 영역에 '닿아 있는지' 판정하는 데 쓴다(세운 물체는 위로 솟아
    영역을 벗어나므로, 영역으로 자르지 말고 연결 판정에만 사용).
    """
    h, w = shape[:2]
    region = np.zeros((h, w), np.uint8)
    hull = cv2.convexHull(charuco_corners.reshape(-1, 2).astype(np.int32))
    cv2.fillConvexPoly(region, hull, 255)
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
        region = cv2.dilate(region, k)
    return region, hull


def auto_hsv_window_from_roi(img_bgr, roi, h_margin=15, s_floor=50, v_floor=50):
    """물체 위 작은 ROI [x,y,w,h]의 색을 샘플링해 HSV 임계창 자동 산출.

    반환 (hsv_low, hsv_high). '물체 색 직접 지정'을 클릭 대신 ROI로 안전하게.
    """
    x, y, w, h = roi
    patch = cv2.cvtColor(img_bgr[y:y+h, x:x+w], cv2.COLOR_BGR2HSV).reshape(-1, 3)
    hmed = int(np.median(patch[:, 0]))
    s10 = int(np.percentile(patch[:, 1], 10))
    v10 = int(np.percentile(patch[:, 2], 10))
    low = np.array([max(0, hmed - h_margin), max(s_floor, s10 - 15), max(v_floor, v10 - 15)])
    high = np.array([min(179, hmed + h_margin), 255, 255])
    return low.astype(np.uint8), high.astype(np.uint8)


def segment_by_color(img_bgr, hsv_low, hsv_high, region_mask=None,
                     open_px=5, close_px=9, min_touch=30):
    """HSV 색창으로 전경 마스크를 만들고, 보드영역에 닿은 최대 덩어리만 남김.

    color 필터(물체 색) + board-connected 필터(배경/키보드 제거)의 결합.
    반환 (mask, contour, bbox) — 실패 시 contour/bbox None.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.asarray(hsv_low), np.asarray(hsv_high))
    if open_px:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px)))
    if close_px:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    best = None
    for i in range(1, n):
        comp = (lab == i)
        if region_mask is not None and (comp & (region_mask > 0)).sum() < min_touch:
            continue
        a = stats[i, cv2.CC_STAT_AREA]
        if best is None or a > best[0]:
            best = (a, i)
    mask = np.zeros(m.shape, np.uint8)
    if best is None:
        return mask, None, None
    i = best[1]
    mask[lab == i] = 255
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    return mask, c, cv2.boundingRect(c)


def poses_relative_to(poses: dict, ref_id: int = 0):
    """모든 마커 포즈를 기준 마커(ref_id) 좌표계로 변환.

    반환: {id: {"T"(4x4 ref_from_marker), "position"(3,), "R"(3x3)}}
    ref_id 자신은 원점(위치 [0,0,0], 회전 단위행렬).
    """
    if ref_id not in poses:
        raise ValueError(f"기준 마커 id{ref_id}가 사진에서 검출되지 않았습니다. "
                         f"검출된 id: {sorted(poses.keys())}")
    T_cam_ref = poses[ref_id]["T"]
    T_ref_cam = np.linalg.inv(T_cam_ref)
    out = {}
    for i, p in poses.items():
        T_ref_marker = T_ref_cam @ p["T"]   # ref_from_marker
        out[i] = {"T": T_ref_marker,
                  "position": T_ref_marker[:3, 3].copy(),
                  "R": T_ref_marker[:3, :3].copy()}
    return out
