"""하이브리드 실시간 — 정확도(DA+FastSAM) + 부드러운 화면을 동시에.

무거운 정확 파이프라인(process_frame_combined)을 **백그라운드 스레드(GPU)** 에서 계속 돌리고,
메인 루프는 웹캠을 **매 프레임 부드럽게** 표시하면서 최신 검출 결과를 덧그린다.
→ 화면 fps는 카메라 속도(부드러움), 검출 갱신은 추론 속도(초당 1~수 회, GPU 상시 활용).

카메라가 크게 움직이지 않고 물체를 놓고 확인하는 용도에 적합(검출 오버레이가 최신 프레임보다
약간 지연될 수 있음 — 추론 1회 지연).
"""
from __future__ import annotations
import os
import time
import threading
import numpy as np
import cv2
import scene3d as s3
import live_combined as lc


def run_live_hybrid(K, dist, board, pipe=None, model=None, square_len=0.038,
                    cam_index=0, calib_wh=(1920, 1080), imgsz=640, ws=(240, 320),
                    snapshot_dir=".", **proc_kw):
    """정확 파이프라인은 백그라운드 스레드, 화면은 부드럽게. [s]스냅 [q]종료."""
    import depth_volume as dv
    if pipe is None:
        pipe = dv.load_depth_model()
    if model is None:
        from ultralytics import FastSAM
        model = FastSAM("FastSAM-s.pt")

    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, calib_wh[0]); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, calib_wh[1])
    if not cap.isOpened():
        raise RuntimeError(f"웹캠({cam_index}) 열기 실패")

    state = {"frame": None, "result": None, "run": True, "infer_ms": 0.0}
    lock = threading.Lock()

    def worker():
        while state["run"]:
            with lock:
                f = None if state["frame"] is None else state["frame"].copy()
            if f is None:
                time.sleep(0.005); continue
            try:
                s = time.time()
                _, objs, markers = lc.process_frame_combined(
                    f, pipe, model, board, K, dist, square_len=square_len, imgsz=imgsz, **proc_kw)
                scene = s3.render_virtual_scene(objs, markers=markers, ws=ws)
                with lock:
                    state["result"] = (objs, markers, scene)
                    state["infer_ms"] = (time.time()-s)*1000
            except Exception as e:
                print("worker error:", e); time.sleep(0.05)

    th = threading.Thread(target=worker, daemon=True); th.start()

    last_scene = np.full((int(ws[1]*2.4), int(ws[1]*2.4), 3), 28, np.uint8)
    snap = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            with lock:
                state["frame"] = frame
                res = state["result"]; infer_ms = state["infer_ms"]
            disp = frame.copy()
            objs = []
            if res is not None:
                objs, markers, last_scene = res
                for i, o in enumerate(objs):                 # 최신 검출을 현재 프레임에 덧그림
                    cv2.drawContours(disp, [o["contour"]], -1, (0, 255, 0), 2)
                    x, y = o["bbox"][0], o["bbox"][1]
                    cv2.putText(disp, f"#{i} {o.get('type','')} {o.get('label','')}",
                                (x, max(16, y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            fps_txt = f"display live | infer {infer_ms:.0f}ms ({1000/max(infer_ms,1):.1f} Hz) | objs {len(objs)}"
            cv2.putText(disp, fps_txt, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            camh = cv2.resize(disp, (int(disp.shape[1]*last_scene.shape[0]/disp.shape[0]), last_scene.shape[0]))
            view = np.hstack([camh, last_scene])
            cv2.imshow("HYBRID: smooth camera + async detect  [s]snap [q]quit", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                cv2.imwrite(os.path.join(snapshot_dir, f"snap_raw_{snap:03d}.png"), frame)
                cv2.imwrite(os.path.join(snapshot_dir, f"snap_vis_{snap:03d}.png"), view)
                print("snapshot", snap); snap += 1
    finally:
        state["run"] = False
        time.sleep(0.1)
        cap.release(); cv2.destroyAllWindows()
