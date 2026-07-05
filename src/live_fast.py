"""고속(경량) 실시간 파이프라인 — 신경망(DA/FastSAM) 없이 높은 fps.

병목이던 DA(≈570ms)·무거운 bg-diff(≈360ms)를 제거/경량화:
- 검출: bg_segment 배경차분을 **저해상도(scale)** 로 → 4~9배 빠름
- 크기/높이: ArUco 기하(평면 역투영 + 수직모서리) — ms 단위
- 자세: 높이로 선/누움 판별(DA 불필요)
- 렌더: scene3d(OpenCV, 빠름)

정확도는 DA+FastSAM 조합보다 낮지만 실시간(수~십수 fps) 확보. 기준(빈 보드)은 [r]로.
"""
from __future__ import annotations
import os
import numpy as np
import cv2
import aruco_utils as au
import bg_segment as bg
import scene3d as s3

STAND_TH_MM = 25.0


def _build_prim(contour, K, dist, rvec, tvec):
    """윤곽 → (kind, cyl=(center,axis,length,radius), center_mm, size_label)."""
    pts = contour.reshape(-1, 2).astype(np.float64)
    x, y, w, h = cv2.boundingRect(contour)
    top = pts[np.argmin(pts[:, 1])]; base = pts[np.argmax(pts[:, 1])]
    H, _ = au.height_from_vertical_edge(base, top, K, dist, rvec, tvec)
    H *= 1000.0
    plane = au.pixels_to_plane(pts, K, dist, rvec, tvec)[:, :2] * 1000.0   # mm
    rect = cv2.minAreaRect(plane.astype(np.float32))
    (cx, cy), (rw, rl), ang = rect
    w_mm, l_mm = sorted([rw, rl])
    if H > STAND_TH_MM:      # 선 물체: 수직
        radius = max(w_mm / 2, 2)
        cyl = (np.array([cx, cy, H / 2]), np.array([0., 0., 1.]), float(H), float(radius))
        return "stand", cyl, (cx, cy), f"H{H:.0f} D~{w_mm:.0f}mm"
    else:                    # 누운 물체: 평면 안
        a = np.deg2rad(ang)
        axis = np.array([np.cos(a), np.sin(a), 0.]) if rl >= rw else np.array([-np.sin(a), np.cos(a), 0.])
        radius = max(w_mm / 2, 2)
        cyl = (np.array([cx, cy, radius]), axis, float(l_mm), float(radius))
        return "lie", cyl, (cx, cy), f"L{l_mm:.0f} W{w_mm:.0f}mm"


def process_frame_fast(frame, board, K, dist, ref_canon, square_len=0.038,
                       scale=0.5, marker_map=None, shape_mode="auto", **bgkw):
    """경량 한 프레임 → (vis, objects, markers). ref_canon(빈 보드) 필요."""
    if ref_canon is None:
        v = frame.copy()
        cv2.putText(v, "empty board -> [r] set reference", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return v, [], []
    small = cv2.resize(frame, None, fx=scale, fy=scale) if scale != 1.0 else frame
    Ks = K.astype(np.float64).copy(); Ks[:2, :] *= scale     # fx,fy,cx,cy 스케일
    det = bg.detect_objects_image(small, board, Ks, dist, ref_canon, square_len, **bgkw)
    if det is None:
        v = frame.copy(); cv2.putText(v, "board not found", (30, 50),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return v, [], []
    rvec, tvec = det["rvec"], det["tvec"]
    objects = []
    vis = small.copy()
    for o in det["objects"]:
        c = o["contour"]
        kind, cyl, center_mm, label = _build_prim(c, Ks, dist, rvec, tvec)
        shape = s3.classify_shape(c) if shape_mode == "auto" else shape_mode
        objects.append({"contour": c, "bbox": o["bbox"], "type": kind, "cyl": cyl,
                        "center_mm": center_mm, "label": label, "shape": shape})
        cv2.drawContours(vis, [c], -1, (0, 255, 0), 2)
        x, y = o["bbox"][0], o["bbox"][1]
        cv2.putText(vis, f"{kind} {label}", (x, max(14, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(vis, f"objects: {len(objects)}  [FAST no-NN]", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    markers = marker_map if marker_map is not None else au.board_marker_map(board)
    return vis, objects, markers


def run_live_fast(K, dist, board, square_len=0.038, cam_index=0, calib_wh=(1920, 1080),
                  scale=0.5, ppc=150, squares_xy=(5, 7), snapshot_dir=".", **proc_kw):
    """고속 실시간: [r]=빈 보드 기준, [s]=스냅, [q]=종료. 카메라 | 가상3D 나란히."""
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, calib_wh[0]); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, calib_wh[1])
    if not cap.isOpened():
        raise RuntimeError(f"웹캠({cam_index}) 열기 실패")
    ref_canon = None; snap = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            vis, objs, markers = process_frame_fast(frame, board, K, dist, ref_canon,
                                                    square_len=square_len, scale=scale, **proc_kw)
            scene = s3.render_virtual_scene(objs, markers=markers, ws=(240, 320))
            camh = cv2.resize(vis, (int(vis.shape[1]*scene.shape[0]/vis.shape[0]), scene.shape[0]))
            view = np.hstack([camh, scene])
            cv2.imshow("FAST live: camera | virtual 3D  [r]ref [s]snap [q]quit", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r'):
                ref_canon = bg.make_reference_canonical(frame, board, K, dist, square_len, ppc, squares_xy)
                print("reference set" if ref_canon is not None else "board not found")
            if key == ord('s'):
                cv2.imwrite(os.path.join(snapshot_dir, f"snap_raw_{snap:03d}.png"), frame)
                cv2.imwrite(os.path.join(snapshot_dir, f"snap_vis_{snap:03d}.png"), view)
                print("snapshot", snap); snap += 1
    finally:
        cap.release(); cv2.destroyAllWindows()
