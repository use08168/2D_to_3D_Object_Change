"""스트림 세션 프레임으로 폰(스트림 해상도) 캘리브레이션.

폰으로 마커 보드를 여러 각도로 비추며 스트리밍(자동 저장)한 세션 폴더를 입력으로,
마커 지도를 평면 타겟 삼아 K·왜곡을 추정 → output/phone_stream_intrinsics.npz 저장.

실행:  python src/calibrate_stream.py [세션폴더]
       (생략 시 output/phone_stream/ 아래 가장 최신 session_* 폴더 사용)
"""
from __future__ import annotations
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import workspace as ws

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_NPZ = os.path.join(ROOT, "output", "phone_stream_intrinsics.npz")


def latest_session():
    dirs = sorted(glob.glob(os.path.join(ROOT, "output", "phone_stream", "session_*")))
    return dirs[-1] if dirs else None


def main():
    sess = sys.argv[1] if len(sys.argv) > 1 else latest_session()
    if not sess or not os.path.isdir(sess):
        print("세션 폴더가 없습니다. 먼저 마커 보드를 스트리밍해 프레임을 수집하세요.")
        return
    n_img = len(glob.glob(os.path.join(sess, "*.jpg")))
    print(f"세션: {sess}  ({n_img}장)")

    corners, _, _, _ = ws.load_marker_map(os.path.join(ROOT, "output", "marker_map.json"))
    res = ws.calibrate_from_map(sess, corners, min_markers=6, max_views=25)
    if res is None:
        print("실패: 마커 6개 이상 보이는 프레임이 3장 미만입니다. 보드를 더 가까이/다양한 각도로 다시 스트리밍하세요.")
        return
    K, dist, rms, nv = res
    # 프레임 크기 기록(검출 시 해상도 불일치 스케일링용)
    import cv2
    im = cv2.imread(sorted(glob.glob(os.path.join(sess, "*.jpg")))[0])
    h, w = im.shape[:2]
    np.savez(OUT_NPZ, K=K, dist=dist, image_size=(w, h), rms=rms)
    print(f"뷰 {nv}개, RMS {rms:.1f}px")
    print(f"fx={K[0,0]:.0f} fy={K[1,1]:.0f} cx={K[0,2]:.0f} cy={K[1,2]:.0f}  ({w}x{h})")
    print(f"저장: {OUT_NPZ}")
    print("이제 검출 모드 실행:  python src/stream_server.py --detect")


if __name__ == "__main__":
    main()
