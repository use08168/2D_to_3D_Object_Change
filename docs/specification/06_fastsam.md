# 06 · FastSAM + 배경차분 선별

**목적:** 클래스 무관 세그멘테이션(FastSAM)으로 정밀 마스크를 얻되, 보드/배경을 걸러 **물체만** 선별.

![fastsam](../images/06_fastsam_segments_everything.png)
*FastSAM은 물체를 잘 자르지만 보드 칸·배경까지 전부 분할한다 → 선별이 필요.*

## 방법
- **FastSAM** = 정밀 마스크(경계).
- **배경차분(05)** = "무엇이 물체인지" 선별 → FastSAM 마스크 중 배경차분 전경과 겹치는 것만 채택.

## 교훈
- 순수 FastSAM(everything)은 보드 칸까지 다 잡아 선별 불가 → 보드 지식(배경차분/깊이) 결합 필수.
- 라이브 `r`로 같은 각도 빈 보드 기준을 잡아야 선별이 깨끗.

## 설치 주의
- `ultralytics`는 `--no-deps`로 설치(안 그러면 `opencv-python`이 `opencv-contrib`를 덮어 `cv2.aruco` 소멸).

## 관련 함수
[`load_fastsam`, `detect_objects_fastsam`, `run_live_fastsam`](modules.md#fastsam_detect)
