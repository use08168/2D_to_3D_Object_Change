"""폰(Expo Go) → PC 실시간 프레임 수신 서버 (WebSocket, Tailscale 전용).

- 텍스트 "ping" → "pong <ms>" (연결 테스트)
- 프레임: 바이너리(JPEG) 또는 "f:<base64 JPEG>" 텍스트 → 디코드 → 표시/검출 + 자동 저장
- 저장: 연결(스트리밍 세션)마다 output/phone_stream/session_YYYYMMDD_HHMMSS/ 폴더에
        수신한 원본 JPEG을 전부 저장(재인코딩 없음 → 화질 보존)

모드
  뷰어(기본):  python src/stream_server.py [port] [host]
  검출:        python src/stream_server.py --detect [port] [host]
    → 분산앵커 파이프라인(process_frame_workspace) 연결. **최신 프레임만 소비**
      (수신은 latest_frame 덮어쓰기, 검출 워커는 자기 속도 ~2fps로 최신 것만 처리
       → 큐가 쌓이지 않아 밀림/지연 누적 없음. 오래된 프레임 자연 폐기)
    → K: output/phone_stream_intrinsics.npz(스트림 캘리브, src/calibrate_stream.py로 생성)
         없으면 근사 K(hfov 60) 폴백(위치 정확도↓ — 캘리브 권장)

host 생략 시 **PC의 Tailscale IP를 자동 감지**해 바인딩 — tailnet 기기만 접속(인터넷 비노출).
(IP를 코드에 하드코딩하지 않음 — 공개 저장소 안전)
뷰어/검출 창 키: [q] 종료
"""
from __future__ import annotations
import asyncio
import base64
import functools
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import websockets

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_SAVE = os.path.join(ROOT, "output", "phone_stream")
STREAM_NPZ = os.path.join(ROOT, "output", "phone_stream_intrinsics.npz")

args = [a for a in sys.argv[1:] if a != "--detect"]
DETECT = "--detect" in sys.argv[1:]
PORT = int(args[0]) if len(args) > 0 else 8765


def detect_tailscale_ip():
    """PC 자신의 Tailscale IPv4 자동 감지(tailscale CLI → 인터페이스 스캔 순)."""
    try:
        out = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=5)
        ip = out.stdout.strip().splitlines()[0].strip() if out.stdout.strip() else ""
        if ip.startswith("100."):
            return ip
    except Exception:
        pass
    try:
        import socket
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip.startswith("100."):        # Tailscale CGNAT 대역
                return ip
    except Exception:
        pass
    return None


HOST = args[1] if len(args) > 1 else detect_tailscale_ip()

latest_frame = None      # 최신 디코드 프레임(BGR) — 검출 워커가 소비
frame_id = 0             # 수신 프레임 카운터(워커의 '새 프레임' 판별용)
_t_prev = None
_fps = 0.0
_quit = False

# ---------------- 검출 파이프라인 (--detect) ----------------
D = {}                   # 검출 리소스(지도·K·모델 등)


def init_detection():
    """마커 지도·K·DA 모델 로드(1회, 수 초). 검출 워커 시작 전에 executor에서 실행."""
    import aruco_utils as au
    import workspace as wsp
    import depth_volume as dv
    import live_da as ld
    import scene3d as s3
    corners, mlist, _, _ = wsp.load_marker_map(os.path.join(ROOT, "output", "marker_map.json"))
    x0, y0, x1, y1 = wsp.map_extent_m(corners)
    D.update(au=au, wsp=wsp, dv=dv, ld=ld, s3=s3, corners=corners, mlist=mlist,
             plane=(x0, y0, x1, y1), origin=(x0*1000, y0*1000),
             ws_sz=((x1-x0)*1000+60, (y1-y0)*1000+60), K0=None, size0=None, Kcache={})
    if os.path.exists(STREAM_NPZ):
        d = np.load(STREAM_NPZ)
        D["K0"] = d["K"].astype(np.float64); D["dist"] = d["dist"].astype(np.float64)
        D["size0"] = tuple(int(v) for v in d["image_size"])
        print(f"[검출] 스트림 캘리브 K 로드 (fx={D['K0'][0,0]:.0f}, {D['size0'][0]}x{D['size0'][1]}, RMS {float(d['rms']):.1f}px)")
    else:
        print("[검출] 스트림 캘리브 없음 → 근사 K(hfov 60) 폴백. 정확도 위해 calibrate_stream.py 권장")
    print("[검출] Depth Anything V2 로드 중…")
    D["pipe"] = dv.load_depth_model()
    print("[검출] 준비 완료 — 폰으로 작업공간을 비추세요")


def _K_for(w, h):
    """프레임 크기에 맞는 (K, dist). 캘리브 해상도와 다르면 스케일, 캘리브 없으면 근사."""
    key = (w, h)
    if key in D["Kcache"]:
        return D["Kcache"][key]
    if D["K0"] is None:
        K, dist = D["au"].approx_camera_matrix((w, h), hfov_deg=60)
    else:
        w0, h0 = D["size0"]
        sx, sy = w / w0, h / h0
        if abs(sx - sy) > 0.02 * sx:
            print(f"[검출] 경고: 프레임 종횡비가 캘리브({w0}x{h0})와 다름({w}x{h}) — 크롭/모드 차이 가능")
        K = D["K0"].copy()
        K[0, 0] *= sx; K[0, 2] *= sx
        K[1, 1] *= sy; K[1, 2] *= sy
        dist = D["dist"]
    D["Kcache"][key] = (K, dist)
    return K, dist


def process_and_render(frame):
    """한 프레임 검출 → '카메라|가상3D' 나란히 뷰 반환(동기, executor에서 실행)."""
    h, w = frame.shape[:2]
    K, dist = _K_for(w, h)
    vis, objs, ok, n, err, pose = D["ld"].process_frame_workspace(
        frame, D["pipe"], K, dist, D["corners"], D["mlist"], D["plane"], min_area_px=800)
    scene = D["s3"].render_virtual_scene(objs, markers=D["mlist"], ws=D["ws_sz"],
                                         origin_mm=D["origin"], cam_pose=pose, plane_xyxy=D["plane"])
    camh = cv2.resize(vis, (int(vis.shape[1]*scene.shape[0]/vis.shape[0]), scene.shape[0]))
    return np.hstack([camh, scene]), objs, ok


async def detect_worker():
    """최신 프레임만 소비하는 검출 루프 — 수신 속도와 무관하게 밀리지 않음."""
    global _quit
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, init_detection)
    last_id = 0
    t_prev = None
    dfps = 0.0
    while not _quit:
        if frame_id == last_id or latest_frame is None:
            await asyncio.sleep(0.02)
            continue
        last_id = frame_id                     # 이 시점의 최신 프레임만 처리(중간 프레임 폐기)
        frame = latest_frame
        try:
            view, objs, ok = await loop.run_in_executor(
                None, functools.partial(process_and_render, frame))
        except Exception:
            import traceback; traceback.print_exc()
            continue
        now = time.time()
        if t_prev is not None:
            dfps = 0.8*dfps + 0.2/(max(now - t_prev, 1e-3)) if dfps > 0 else 1/max(now-t_prev, 1e-3)
        t_prev = now
        cv2.putText(view, f"detect {dfps:.1f} fps  (recv {_fps:.1f} fps)", (14, view.shape[0]-14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("phone live detect: camera | virtual 3D  [q]quit", view)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            _quit = True
    cv2.destroyAllWindows()


# ---------------- 수신 ----------------

def _handle_frame(jpg: bytes, sess: dict):
    """JPEG bytes → 세션 폴더 저장 + 최신 프레임 갱신 (+뷰어 모드면 표시)."""
    global latest_frame, frame_id, _t_prev, _fps, _quit
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"    [!] JPEG 디코드 실패 ({len(jpg)} bytes)")
        return
    if sess["dir"] is None:
        sess["dir"] = os.path.join(BASE_SAVE, time.strftime("session_%Y%m%d_%H%M%S"))
        os.makedirs(sess["dir"], exist_ok=True)
        print(f"    [프레임 수신 시작] {img.shape[1]}x{img.shape[0]}  {len(jpg)/1024:.0f} KB")
        print(f"    [세션 폴더] {sess['dir']}")
    with open(os.path.join(sess["dir"], f"frame_{sess['n']:04d}.jpg"), "wb") as f:
        f.write(jpg)
    sess["n"] += 1

    now = time.time()
    if _t_prev is not None:
        inst = 1.0 / max(now - _t_prev, 1e-3)
        _fps = 0.85 * _fps + 0.15 * inst if _fps > 0 else inst
    _t_prev = now
    latest_frame = img
    frame_id += 1

    if not DETECT:                             # 뷰어 모드에서만 원본 표시(검출 모드는 워커가 표시)
        vis = img.copy()
        cv2.putText(vis, f"phone stream  {img.shape[1]}x{img.shape[0]}  {_fps:.1f} fps  #{sess['n']}",
                    (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("phone stream  [q]quit", vis)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            _quit = True


async def handler(ws):
    global _quit
    peer = ws.remote_address
    sess = {"dir": None, "n": 0}          # 이 연결(스트리밍 세션)의 저장 폴더·카운터
    print(f"[+] 연결됨: {peer}")
    try:
        async for msg in ws:
            if isinstance(msg, bytes) or msg.startswith("f:"):
                try:
                    _handle_frame(msg if isinstance(msg, bytes) else base64.b64decode(msg[2:]), sess)
                except Exception:
                    import traceback
                    traceback.print_exc()      # 표시/저장 에러를 숨기지 않고 출력
            elif msg.strip().lower() == "ping":
                await ws.send(f"pong {int(time.time()*1000)}")
                print(f"    ping → pong  ({peer[0]})")
            else:
                await ws.send(f"echo: {msg}")
            if sess["n"] and sess["n"] % 30 == 0:
                print(f"    프레임 #{sess['n']}  수신 {_fps:.1f} fps")
            if _quit:
                break
    except websockets.ConnectionClosed:
        pass
    finally:
        if sess["n"]:
            print(f"[-] 세션 종료: {peer}  — {sess['n']}장 저장됨 ({sess['dir']})")
        else:
            print(f"[-] 연결 종료: {peer}")
        if _quit:
            cv2.destroyAllWindows()
            asyncio.get_event_loop().stop()


async def main():
    if not HOST:
        print("Tailscale IP를 찾지 못했습니다. Tailscale을 켜거나 host를 직접 지정하세요:")
        print("  python src/stream_server.py [--detect] 8765 <바인딩할 IP>")
        return
    try:
        server = await websockets.serve(handler, HOST, PORT, max_size=20*1024*1024)
    except OSError as e:
        print(f"바인딩 실패({HOST}:{PORT}): {e}")
        print("→ Tailscale이 꺼져 있으면 켜거나, 임시로 'python src/stream_server.py 8765 0.0.0.0'")
        return
    async with server:
        print(f"WebSocket 서버 대기 중: ws://{HOST}:{PORT}  (Ctrl+C 종료)")
        if HOST != "0.0.0.0":
            print("(Tailscale 전용 바인딩 — tailnet 기기만 접속 가능)")
        print(f"세션 저장 루트: {BASE_SAVE}")
        if DETECT:
            asyncio.create_task(detect_worker())
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, RuntimeError):
        print("종료")
