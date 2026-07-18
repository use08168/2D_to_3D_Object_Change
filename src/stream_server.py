"""폰(Expo Go) → PC 실시간 프레임 수신 서버 (WebSocket, Tailscale 전용).

- 텍스트 "ping" → "pong <ms>" (연결 테스트)
- 프레임: 바이너리(JPEG) 또는 "f:<base64 JPEG>" 텍스트 → 디코드 → 실시간 표시 + 자동 저장
- 저장: 연결(스트리밍 세션)마다 output/phone_stream/session_YYYYMMDD_HHMMSS/ 폴더를 새로 만들어
        수신한 원본 JPEG을 전부 저장(재인코딩 없음 → 화질 보존)
- 뷰어 창 키: [q] 종료

실행:  python src/stream_server.py [port] [host]
host 생략 시 **PC의 Tailscale IP를 자동 감지**해 바인딩 — tailnet 기기만 접속 가능(인터넷 비노출).
(IP를 코드에 하드코딩하지 않음 — 공개 저장소 안전)
모든 인터페이스로 열려면: python src/stream_server.py 8765 0.0.0.0 (공인IP 직결 PC라 비권장)
"""
from __future__ import annotations
import asyncio
import base64
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import websockets

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
BASE_SAVE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "output", "phone_stream")


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


HOST = sys.argv[2] if len(sys.argv) > 2 else detect_tailscale_ip()

latest_frame = None      # 최신 디코드 프레임(BGR) — 검출 파이프라인이 소비할 대상
_t_prev = None
_fps = 0.0
_quit = False


def _handle_frame(jpg: bytes, sess: dict):
    """JPEG bytes → 저장(원본) + 디코드 → 뷰어 표시(+fps). [q]종료."""
    global latest_frame, _t_prev, _fps, _quit
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"    [!] JPEG 디코드 실패 ({len(jpg)} bytes)")
        return
    # 세션 폴더(첫 프레임에 생성) + 원본 JPEG 그대로 저장
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
                print(f"    프레임 #{sess['n']}  {_fps:.1f} fps")
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
        print("  python src/stream_server.py 8765 <바인딩할 IP>")
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
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, RuntimeError):
        print("종료")
