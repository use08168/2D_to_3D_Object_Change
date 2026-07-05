# 02 · 보드 위 물체 크기 측정 (기하)

**목적:** ChArUco 보드 위 물체를 한 장에서 **바닥치수(W×L) + 높이(H)** 로 측정. 순수 ArUco 기하(딥러닝 없음).

## 원리
- 보드 검출 → 카메라 기준 **평면 포즈**.
- 물체 **바닥 코너 픽셀**을 보드평면(z=0)에 **역투영**(`pixels_to_plane`) → 실제 mm → 가로·세로·중심.
- 물체 **수직 모서리**(바닥·꼭대기 픽셀)로 **높이**(`height_from_vertical_edge`, 합성검증 오차 0.1mm).

## 입력 / 출력
- 입력: `output/camera_intrinsics.npz`, `data/scene_images/`의 (보드+물체) 사진.
- 출력: `output/object_size.json`.

## 한계
- 단일 시점이라 가려진 바닥/뒷면은 근사. 박스형에 정확, 불규칙 물체는 실루엣 기준.
- 세그멘테이션은 반자동(픽셀 좌표 지정) → 이후 06~09에서 자동화.

## 관련 함수
[`detect_charuco_pose`, `pixels_to_plane`, `height_from_vertical_edge`](modules.md#aruco_utils)
