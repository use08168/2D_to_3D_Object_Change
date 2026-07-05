# src 모듈 정리

각 `.py` 파일의 함수와 역할. 좌표계는 특별한 언급이 없으면 **보드(작업공간) 좌표계, mm**, z축이 '위'.

---

## aruco_utils
ArUco/ChArUco 검출·포즈, 평면 기하, 색 분할의 기초.

| 함수 | 역할 |
|---|---|
| `approx_camera_matrix(image_size, hfov_deg=60)` | 무보정 근사 K (화각 기반) |
| `load_intrinsics(npz_path)` | 캘리브 결과(K, dist) 로드 |
| `make_detector(dict_id)` | ArUco 검출기(서브픽셀 코너) |
| `marker_object_points(L)` | 마커 4코너 3D(마커 좌표계) |
| `estimate_marker_poses(corners, ids, K, dist, L)` | 마커별 포즈(solvePnP IPPE_SQUARE) |
| `rt_to_matrix(rvec, tvec)` | (rvec,tvec)→4×4 동차변환 |
| `detect_charuco_pose(gray, board, K, dist)` | ChArUco 보드 포즈(카메라←보드) |
| `pixels_to_plane(pixels, K, dist, rvec, tvec)` | 픽셀 → 보드평면(z=0) 역투영 |
| `height_from_vertical_edge(base_px, top_px, K, dist, rvec, tvec)` | 수직 모서리로 높이[m] |
| `board_marker_map(board)` | 보드 정의에서 **전체 마커 지도**(검출 무관) |
| `board_region_mask(shape, charuco_corners, dilate_px)` | 보드 영역 마스크(볼록껍질) |
| `auto_hsv_window_from_roi(img, roi, ...)` | ROI 색 샘플 → HSV 창 자동 |
| `segment_by_color(img, hsv_low, hsv_high, region_mask, ...)` | 색창+보드연결 분할 |
| `poses_relative_to(poses, ref_id)` | 마커 포즈들을 기준 마커 좌표계로 |

---

## depth_volume
Depth Anything V2 + ArUco 평면 앵커링 → 높이맵·3D 점군.

| 함수 | 역할 |
|---|---|
| `load_depth_model(model_id, device)` | DA V2 파이프라인 로드 |
| `height_map_from_depth(frame, pipe, board, K, dist, ...)` | 왜곡보정→포즈→DA→앵커링→dict(`imgu, height_mm, region, quad, pts_board, rvec, tvec, r2`) |
| `detect_objects_by_height(hm, K, dist, ...)` | 높이 임계로 물체 검출·측정 |
| `draw_depth_objects(hm, objs)` / `colorize_height(hm, hmax_mm)` | 시각화 |

> `pts_board`: 픽셀별 3D 점(보드좌표계, m), 높이맵·자세 계산에 사용. 높이 부호는 카메라 쪽이 +.

---

## fastsam_detect
FastSAM(정밀 마스크) + 배경차분(선별).

| 함수 | 역할 |
|---|---|
| `load_fastsam(weights='FastSAM-s.pt')` | 모델 로드(가중치 자동 다운로드) |
| `detect_objects_fastsam(frame, model, board, K, dist, ref_canon, square_len, ...)` | FastSAM∩배경차분 → 물체(중심·크기) |
| `draw_fastsam(...)` / `run_live_fastsam(...)` | 시각화 / 실시간([r]기준 [s]스냅 [q]종료) |

---

## bg_segment
배경차분(무학습, 색 무관) 물체 추출.

| 함수 | 역할 |
|---|---|
| `rectify_to_canonical(img, board, K, dist, square_len, ...)` | 보드 자세로 정면 펴기 |
| `canonical_homography(rvec, tvec, K, ppm)` | 이미지↔canonical 호모그래피 |
| `make_reference(...)` / `make_reference_canonical(...)` | 빈 보드 기준(밴드패스 / 컬러) |
| `detect_objects_bgdiff(...)` | canonical 차분 검출 |
| `detect_objects_image(frame, board, K, dist, ref_canon, square_len, ...)` | **이미지 공간** 차분(자연 시점, 세운 물체 안 잘림) |
| `draw_canonical(...)` / `draw_image(...)` | 시각화 |
| `run_live_bgdiff(...)` / `run_live_image(...)` | 실시간 |

---

## live_combined
DA + FastSAM + ArUco 실시간 통합(자연 시점 합성) + 가상 3D 씬 연동.

| 함수 | 역할 |
|---|---|
| `process_frame_combined(frame, pipe, model, board, K, dist, ..., marker_map=None, shape_mode='auto')` | 한 프레임 → `(vis, objects, markers)`. 물체별 측정·자세·원통(축은 분류에 고정)·모양 포함 |
| `run_live_combined(...)` | 실시간: 카메라 합성 \| 가상 3D 씬 나란히([s]스냅 [q]종료) |

> `shape_mode`: `'auto'`(보수적 자동) / `'cylinder'` / `'box'`(수동 강제). `marker_map`: 전체 마커 지도(없으면 보드에서 생성).

---

## scene3d
가상 3D 씬 렌더(원통/박스 프리미티브 + 마커 지도).

| 함수 | 역할 |
|---|---|
| `fit_cylinder(P_mm)` | 점군 → 방향성 원통(robust 이상치 제거) |
| `classify_shape(contour)` | 박스/원통 추정(**보수적**, 단일시점 한계) |
| `render_virtual_scene(objects, markers=None, ws, az, el)` | OpenCV 3/4 시점 렌더(빠름, 실시간용) |
| `render_plotly(objects, markers=None, ws, html_path=None)` | **인터랙티브** 3D(plotly, 마우스 회전/확대, HTML 저장) |

---

## live_measure
색 기반 실시간 복수 물체 측정(초기 버전).

| 함수 | 역할 |
|---|---|
| `scale_intrinsics(K, from_wh, to_wh)` | 해상도 다를 때 K 스케일 |
| `measure_objects(frame, board, K, dist, ...)` | 채도 기반 복수 물체 검출·측정 |
| `draw_overlay(...)` / `run_live(...)` | 시각화 / 실시간 |
