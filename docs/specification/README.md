# Specification

이 폴더는 각 노트북과 `src/` 모듈의 상세 설명을 담는다. 프로젝트 전체 개요는 [최상위 README](../../README.md) 참고.

## 파이프라인 개요

```
ChArUco 캘리브레이션 ─▶ ArUco 평면/좌표계 ─▶ 물체 검출 ─▶ 측정 ─▶ 3D 가상 배치
     (01)                    (기하)          (색→깊이)   (기하)     (원통/박스+마커지도)
```

역할 분담:

| 역할 | 담당 |
|---|---|
| 카메라 내부파라미터(1회) | ChArUco 캘리브레이션 |
| 좌표계·평면·스케일 | ArUco 기하 |
| 물체 감지 + 깊이 | Depth Anything V2 |
| 물체 경계(실루엣) | FastSAM |
| 정확한 측정·자세 | ArUco 기하 + PCA |
| 가상 3D 표현 | scene3d (원통/박스 + 마커 지도) |

## 노트북 문서

| 노트북 | 문서 | 요약 |
|---|---|---|
| `01_charuco_calibration.ipynb` | [01_calibration.md](01_calibration.md) | 카메라 내부파라미터 K·왜곡 산출(1회) |
| `02_measure_object_on_board.ipynb` | [02_measure_on_board.md](02_measure_on_board.md) | 보드 위 물체 W×L×H 측정(기하) |
| `03_segment_object.ipynb` | [03_segmentation.md](03_segmentation.md) | 고전 필터 물체 분할(무학습) |
| `05_bgdiff_experiment.ipynb` | [05_bgdiff.md](05_bgdiff.md) | 배경차분(색 무관) 물체 추출 |
| `06_fastsam_measure.ipynb` | [06_fastsam.md](06_fastsam.md) | FastSAM + 배경차분 선별 |
| `07_depth_volume.ipynb` | [07_depth_volume.md](07_depth_volume.md) | 단안 깊이 + 앵커링 → 높이·부피 |
| `08_live_combined.ipynb` | [08_live_combined.md](08_live_combined.md) | 실시간 통합(자연 시점 합성) |
| `09_virtual_scene.ipynb` | [09_virtual_scene.md](09_virtual_scene.md) | 가상 3D 배치 + 인터랙티브 뷰어 |
| `10_fast_live.ipynb` | — | 고속 경량(신경망 없음, bg-diff) ~8~14fps |
| `11_hybrid_live.ipynb` | — | 하이브리드(정확 파이프라인 백그라운드 스레드 + 부드러운 화면) |
| `12_da_only_live.ipynb` | — | **DA 단독 실시간(~2.5fps, 검출 깔끔, 권장)** |

보조: `alt_generate_markers.ipynb`(마커 생성), `alt_markers_to_3d.ipynb`(마커 3D 배치 초기 실험).

### 실시간 모드 비교
| 모드 | 속도 | 검출품질 | 특징 |
|---|---|---|---|
| 10 경량 | ~8~14fps | 약함 | 신경망 없음(bg-diff), 빈보드 `r` 필요 |
| **12 DA만** | ~2.5fps | 깔끔 | **권장.** 깊이 솟음으로 검출, 3D-XY 필터로 보드 밖 물체도 통째 |
| 11 하이브리드 | 화면 부드러움 | 정확 | DA+FastSAM 백그라운드 스레드(비동기) |
| 08/09 동기 | ~1.4fps | 정밀크기 | DA+FastSAM 동기(정지 분석용) |

> 성능: DA 후처리(픽셀당 3D)를 GPU(torch)로 이식(547→~260ms), FastSAM 마스크 선별도 GPU 벡터화. 병목은 두 모델의 추론 자체(DA 185ms+FastSAM 70ms)라 동기 실시간은 GTX1660에서 ~2fps가 천장.

## 모듈 문서

- [modules.md](modules.md) — `src/` 각 `.py` 파일의 함수·역할 정리.

## 개발 계보 (문제 → 해결)

1. **무보정 단일사진** → 화각 민감(부정확) → ChArUco 캘리브레이션으로 K 확보.
2. **색/채도 분할 실패** (03·04) → 배경차분(05, 색 무관) → FastSAM(06).
3. **FastSAM 선별 문제** → 배경차분/깊이로 물체만 선별.
4. **부피(깊이) 한계** → 단안 깊이 DA V2 + ArUco 앵커링(07).
5. **정확도** → DA는 감지, ArUco 기하로 측정(08, 높이 ±10%).
6. **가상 3D + 마커 지도 + 자세/모양** → 09.
