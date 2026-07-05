# 01 · ChArUco 카메라 캘리브레이션

**목적:** 카메라 내부파라미터(camera matrix `K`, 왜곡계수 `dist`)를 **1회** 산출. 이후 모든 solvePnP·거리·깊이 계산의 기반.

## 입력 / 출력
- 입력: `data/calib_images/`의 ChArUco 보드 사진 15~25장(각도 다양).
- 출력: `output/camera_intrinsics.npz`, `output/camera_intrinsics.yaml`.

## 핵심 단계
1. ChArUco 보드 정의(5×7, `SQUARE_LENGTH_M`·`MARKER_LENGTH_M`는 **인쇄물 실측값**). 실측: 검은 격자 한 변 38mm → SQUARE=0.038, MARKER=0.038×22/30≈0.0279.
2. `CharucoDetector.detectBoard`로 코너 검출 → `board.matchImagePoints` → `cv2.calibrateCamera`.
3. RMS 재투영 오차 확인(1px 이하 목표. 이 프로젝트 실측 ≈ 0.61px).

## 주의
- 촬영에 쓴 **카메라·해상도·초점**을 이후에도 동일하게 유지해야 K가 유효.
- 관련 함수: [`aruco_utils.load_intrinsics`](modules.md#aruco_utils).

## 실행
```
conda activate vision_aruco
jupyter lab  # 커널: Python (vision_aruco)
```
