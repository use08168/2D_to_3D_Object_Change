// 2단계: 폰 카메라 → PC 실시간 프레임 스트리밍 (Tailscale 전용)
// - [연결]: PC 서버(config.js의 Tailscale 주소)와 WebSocket 연결 + ping RTT
// - [스트리밍 시작]: 촬영 루프(takePictureAsync) → JPEG base64를 "f:" 접두어로 연사 전송
// - PC(stream_server.py)가 수신·표시. 전송 fps/용량 표시.
import { useEffect, useRef, useState } from 'react';
import { SafeAreaView, StatusBar, StyleSheet, Text, TextInput, TouchableOpacity, View } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { SERVER } from './config';   // PC Tailscale IP — config.js(gitignore)에서 로드

export default function App() {
  const [url, setUrl] = useState(SERVER);
  const [status, setStatus] = useState('대기');
  const [streaming, setStreaming] = useState(false);
  const [stats, setStats] = useState({ sent: 0, fps: 0, kb: 0 });
  const [picSize, setPicSize] = useState(undefined);
  const [permission, requestPermission] = useCameraPermissions();
  const camRef = useRef(null);
  const wsRef = useRef(null);
  const streamingRef = useRef(false);
  const pingT0 = useRef(0);

  useEffect(() => { if (permission && !permission.granted) requestPermission(); }, [permission]);

  const connect = () => {
    try { wsRef.current?.close(); } catch {}
    setStatus('연결 중…');
    const ws = new WebSocket(url);
    wsRef.current = ws;
    const timeout = setTimeout(() => {
      if (ws.readyState !== 1) { ws.close(); setStatus('실패(타임아웃 5s)'); }
    }, 5000);
    ws.onopen = () => { clearTimeout(timeout); pingT0.current = Date.now(); ws.send('ping'); };
    ws.onmessage = (e) => {
      if (typeof e.data === 'string' && e.data.startsWith('pong')) {
        setStatus(`연결 OK  (RTT ${Date.now() - pingT0.current}ms)`);
      }
    };
    ws.onerror = () => setStatus('오류');
    ws.onclose = () => { setStatus('연결 종료'); stopStream(); };
  };

  const onCameraReady = async () => {
    // 가용 사진 크기 중 ~1920x1440(2.7MP)에 가장 가까운 걸 선택 — 화질↑ (캘리브·검출용)
    try {
      const sizes = (await camRef.current?.getAvailablePictureSizesAsync?.()) ?? [];
      let best, bestDiff = Infinity;
      for (const s of sizes) {
        const [w, h] = s.split('x').map(Number);
        if (!w || !h) continue;
        const d = Math.abs(w * h - 1920 * 1440);
        if (d < bestDiff) { bestDiff = d; best = s; }
      }
      if (best) setPicSize(best);
    } catch {}
  };

  const loop = async () => {
    while (streamingRef.current) {
      const ws = wsRef.current;
      if (!ws || ws.readyState !== 1) break;
      try {
        const t0 = Date.now();
        const photo = await camRef.current.takePictureAsync({
          base64: true, quality: 0.75, skipProcessing: true, shutterSound: false, exif: false,
        });
        if (!streamingRef.current) break;
        ws.send('f:' + photo.base64);
        const dt = Math.max(Date.now() - t0, 1);
        setStats((s) => ({
          sent: s.sent + 1,
          fps: Math.round((1000 / dt) * 10) / 10,
          kb: Math.round((photo.base64.length * 0.75) / 1024),
        }));
      } catch { break; }
    }
    streamingRef.current = false;
    setStreaming(false);
  };

  const startStream = () => {
    if (!wsRef.current || wsRef.current.readyState !== 1) { setStatus('먼저 연결하세요'); return; }
    streamingRef.current = true;
    setStreaming(true);
    loop();
  };
  const stopStream = () => { streamingRef.current = false; setStreaming(false); };

  const ok = status.startsWith('연결 OK');
  if (!permission) return <View style={styles.container} />;
  if (!permission.granted) {
    return (
      <SafeAreaView style={styles.container}>
        <Text style={styles.title}>카메라 권한 필요</Text>
        <TouchableOpacity style={styles.btn} onPress={requestPermission}>
          <Text style={styles.btnText}>권한 허용</Text>
        </TouchableOpacity>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" />
      <CameraView ref={camRef} style={styles.camera} facing="back"
        pictureSize={picSize} animateShutter={false} onCameraReady={onCameraReady} />

      <View style={styles.panel}>
        <Text style={[styles.status, { color: ok ? '#4ade80' : '#facc15' }]}>{status}</Text>
        <Text style={styles.stats}>
          전송 {stats.sent}장 · {stats.fps} fps · {stats.kb} KB/장 {picSize ? `· ${picSize}` : ''}
        </Text>
        <TextInput style={styles.input} value={url} onChangeText={setUrl}
          autoCapitalize="none" autoCorrect={false} />
        <View style={styles.row}>
          <TouchableOpacity style={[styles.btn, { flex: 1 }]} onPress={connect}>
            <Text style={styles.btnText}>연결</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.btn, { flex: 1, backgroundColor: streaming ? '#dc2626' : '#16a34a' }]}
            onPress={streaming ? stopStream : startStream}>
            <Text style={styles.btnText}>{streaming ? '정지' : '스트리밍 시작'}</Text>
          </TouchableOpacity>
        </View>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#111827' },
  camera: { flex: 1 },
  panel: { padding: 16, paddingBottom: 28, backgroundColor: '#111827' },
  title: { color: 'white', fontSize: 22, fontWeight: 'bold', margin: 20 },
  status: { fontSize: 16, fontWeight: '600', marginBottom: 4 },
  stats: { color: '#9ca3af', fontSize: 13, marginBottom: 10 },
  input: { backgroundColor: '#1f2937', color: 'white', borderRadius: 8, padding: 10, fontSize: 14, marginBottom: 10 },
  row: { flexDirection: 'row', gap: 10 },
  btn: { backgroundColor: '#2563eb', borderRadius: 8, padding: 14, alignItems: 'center' },
  btnText: { color: 'white', fontSize: 15, fontWeight: '700' },
});
