# src 모듈 정리

각 `.py` 파일의 함수와 역할. 좌표계는 특별한 언급이 없으면 **보드(작업공간) 좌표계, mm**, z축이 '위'.

---

## aruco_utils
ArUco/ChArUco 검출·포즈, 평면 기하, 색 분할의 기초.

| 함수 | 역할 |
|---|---|
| `approx_camera_matrix(image_size, hfov_deg=60)` | 무보정 근사 K (화각 기반) |
| `load_intrinsics(npz_path)` | 캘리브 결과(K, dist) 로드 — `camera_matrix/dist_coeffs`·`K/dist` 두 키규약 자동 |
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
| `height_map_from_depth(frame, pipe, board, K, dist, ..., pose=None, plane_xyxy=None)` | 왜곡보정→포즈→DA→앵커링→dict(`imgu, height_mm, region, quad, pts_board, rvec, tvec, r2`). `pose`(분산앵커 로컬라이제이션)·`plane_xyxy`(작업공간 범위) 주면 보드 대신 사용 |
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

## workspace
분산 앵커 작업공간 — 마커 지도로 카메라 로컬라이제이션(넓은 공간에서 보드 대체).

| 함수 | 역할 |
|---|---|
| `load_marker_map(path)` | `marker_map.json` → (corners{id:(4,2)mm}, markers_list, ref_id, marker_len) |
| `map_extent_m(corners)` | 지도 (X,Y) 범위(m) |
| `calibrate_from_map(image_dir, corners_map, ...)` | 지도 구축 사진들로 카메라 캘리브(Zhang 평면법) → (K, dist, rms, n) |
| `make_detector()` | ArUco 검출기(DICT_4X4_50, 서브픽셀) |
| `localize(frame, corners_map, K, dist, ...)` | 검출 마커 ↔ 지도 3D `solvePnP`(IPPE) → (rvec, tvec, n_used, reproj_px) |

> 마커 지도는 물리 치수라 카메라 무관. 런타임엔 마커 4~5개만 보여도 로컬라이즈(최소앵커 실험 참고).

---

## live_da
DA 단독 실시간 + 분산 앵커 작업공간 파이프라인. 검출 `detect="tophat"`(국소대비)가 기본.

| 함수 | 역할 |
|---|---|
| `_tophat_local(hmm, H, W, bg_ksize, bg_scale)` | 높이맵 국소배경(롤링볼) 차감 → 국소 대비(주변보다 솟은 곳) |
| `process_frame_da(frame, pipe, board, K, dist, ..., detect='tophat', tophat_mm, peak_min_mm, spread_min_mm, pose, plane_xyxy)` | 한 프레임 → (vis, objects, markers). top-hat 검출 + 봉우리 검증, 측정은 앵커 절대높이 |
| `run_live_da(...)` | DA 단독 실시간(보드 모드) |
| `process_frame_workspace(frame, pipe, K, dist, corners_map, markers_list, plane_xyxy, ...)` | 분산앵커 한 프레임: 로컬라이즈 → 검출 → (vis, objs, ok, n, err, pose) |
| `run_live_workspace(K, dist, corners_map, markers_list, plane_xyxy, ...)` | 작업공간 실시간: 카메라 \| 가상 3D. `[s]`스냅=PNG+인터랙티브 HTML, `[q]`종료 |

> 자세: `da_h>25 ∧ 세로≥0.7×가로`면 서있음(원통 수직), 아니면 누움(원통 적합). 가로폭은 밑동 그림자 제외 윗부분에서 계산.

---

## scene3d
가상 3D 씬 렌더(원통/박스 프리미티브 + 마커 지도).

| 함수 | 역할 |
|---|---|
| `fit_cylinder(P_mm)` | 점군 → 방향성 원통(robust 이상치 제거) |
| `classify_shape(contour)` | 박스/원통 추정(**보수적**, 단일시점 한계) |
| `render_virtual_scene(objects, markers=None, ws, az, el, origin_mm, cam_pose=None, plane_xyxy=None)` | OpenCV 3/4 시점 렌더(빠름, 실시간용). `origin_mm`=id0 좌표 격자 정렬, `cam_pose`=실제 카메라 방향 정렬 |
| `_lookat_aligned(cam_pose, center, D, el)` | 실제 카메라 R의 수평 right/forward 승계 → 가상 카메라(좌우·앞뒤 실제 일치) |
| `render_plotly(objects, markers=None, ws, html_path, origin_mm, cam_pose=None)` | **인터랙티브** 3D(plotly, 마우스 회전/확대, 자체 HTML). `cam_pose` 주면 yaxis 반전으로 chirality 교정 + 초기 카메라 정렬(좌표값은 true 유지) |

---

## stream_server
폰(Expo Go) → PC 실시간 프레임 수신 서버(WebSocket, Tailscale 전용). 상세: [16_phone_stream.md](16_phone_stream.md)

| 함수 | 역할 |
|---|---|
| `detect_tailscale_ip()` | PC 자신의 Tailscale IPv4 자동 감지(CLI→인터페이스 스캔) — IP 하드코딩 없음 |
| `_handle_frame(jpg, sess)` | JPEG 원본을 세션 폴더에 저장 + `latest_frame` 갱신(+뷰어 모드 표시) |
| `init_detection()` / `process_and_render(frame)` | 검출 리소스 로드(지도·K·DA) / 한 프레임 검출→카메라\|가상3D 뷰 |
| `detect_worker()` | **최신 프레임만 소비**하는 검출 루프(밀림 없음, executor로 비블로킹) |
| `handler(ws)` / `main()` | ping/pong·프레임 수신 루프 / Tailscale IP 바인딩 서버 |

> 뷰어 `python src/stream_server.py [port] [host]` / 검출 `--detect`. 세션(연결)마다 `output/phone_stream/session_*/`에 전 프레임 자동 저장. K는 phone_stream_intrinsics.npz(해상도 자동 스케일, 없으면 근사 폴백).

---

## calibrate_stream
스트림 세션 프레임으로 폰(스트림 해상도) 캘리브레이션.

| 함수 | 역할 |
|---|---|
| `latest_session()` / `main()` | 최신 session_* 자동 선택 → `ws.calibrate_from_map`(25뷰) → `output/phone_stream_intrinsics.npz` |

> 사진모드와 영상모드 화각이 다름(실측 fx 7% 차이) → 스트림 전용 캘리브 필수. 실측 RMS 5.6px.

---

## live_measure
색 기반 실시간 복수 물체 측정(초기 버전).

| 함수 | 역할 |
|---|---|
| `scale_intrinsics(K, from_wh, to_wh)` | 해상도 다를 때 K 스케일 |
| `measure_objects(frame, board, K, dist, ...)` | 채도 기반 복수 물체 검출·측정 |
| `draw_overlay(...)` / `run_live(...)` | 시각화 / 실시간 |
