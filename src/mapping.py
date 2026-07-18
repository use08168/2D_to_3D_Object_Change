"""증분(전이) 마커 매핑 — 확정된 마커가 새 마커의 기준이 되어 지도가 바깥으로 자란다.

기존 방식은 사진에 지도의 기준(사실상 id0 근방)이 보여야 등록됐지만, 여기서는:
  1) seed(id0)만 아는 상태에서 시작
  2) '앵커'(신뢰 게이트 통과 마커)가 min_anchor_markers개 이상 보이는 사진마다
     findHomography(이미지→지도, RANSAC) → 사진 속 모든 마커 코너를 지도로 투영해 관측 누적
  3) 마커 위치 = 관측 평균, 관측 std 계산 → 게이트(관측수·std) 통과 시 앵커 승격
  4) 수렴까지 전역 반복(간이 번들 조정) → 등록 순서 의존성·드리프트 완화
  hop = seed로부터의 전이 단수(BFS) — 드리프트 분석용.

주의(한계): 평면 가정(호모그래피) — 마커들이 한 평면 위에 있어야 함.
전이는 홉마다 오차가 누적될 수 있으므로, 먼 영역을 잇는 '루프 클로저' 사진
(두 영역 마커를 한 프레임에)이 있으면 오차가 크게 줄어든다.
"""
from __future__ import annotations
import glob
import os

import cv2
import numpy as np


def detect_observations(files, detector=None, K=None, dist=None, min_markers=1):
    """사진들에서 마커 코너 관측 수집 → [{id: (4,2) px}] (빈 프레임 제외).

    K, dist를 주면 왜곡 보정된 이상적 픽셀 좌표로 변환(호모그래피 정확도↑).
    """
    if detector is None:
        import workspace as ws
        detector = ws.make_detector()
    obs = []
    for f in files:
        g = cv2.imdecode(np.fromfile(f, np.uint8), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        mc, ids, _ = detector.detectMarkers(g)
        if ids is None or len(ids) < min_markers:
            continue
        rec = {}
        for c, i in zip(mc, ids.flatten()):
            pts = c.reshape(4, 2).astype(np.float64)
            if K is not None:
                pts = cv2.undistortPoints(pts.reshape(-1, 1, 2), K, dist, P=K).reshape(4, 2)
            rec[int(i)] = pts
        obs.append(rec)
    return obs


def _fix_gauge(corners, seed_id, L):
    """게이지 고정: ①전체 마커 변길이 중앙값=L(mm)로 스케일 ②seed 사각형을 원점 정렬.

    호모그래피 지도는 전역 배율이 자유변수(게이지) — seed 4점만으론 스케일 핀이 약해
    부트스트랩 오차가 보존된다. 모든 마커가 물리 L mm임을 이용해 강하게 고정.
    """
    edges = []
    for c in corners.values():
        q = np.asarray(c)
        for k in range(4):
            edges.append(np.linalg.norm(q[k] - q[(k + 1) % 4]))
    s = L / np.median(edges)
    corners = {i: np.asarray(c) * s for i, c in corners.items()}
    # seed를 캐논 사각형에 강체 정렬(회전+병진, 스케일은 위에서 고정)
    seed_sq = np.array([[0, 0], [L, 0], [L, L], [0, L]], np.float64)
    A = corners[seed_id]
    ca, cb = A.mean(0), seed_sq.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (seed_sq - cb))
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return {i: (R @ (c - ca).T).T + cb for i, c in corners.items()}


def _frame_H(rec, corners, known, ransac_px):
    """한 프레임의 이미지→지도 호모그래피(공통 마커 전부, RANSAC)."""
    src = np.vstack([rec[i] for i in known])
    dst = np.vstack([corners[i] for i in known])
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, ransac_px)
    if H is None or inl is None or inl.sum() < 4:
        return None
    return H


def _hull_gate(rec, known, margin_mult):
    """외삽 금지 게이트: 프레임이 아는 앵커들의 이미지상 볼록껍질 '근처' 마커만 통과.

    호모그래피는 관측 영역 '밖'으로 갈수록 오차가 폭발(외삽) → 각 프레임은 자기가 본
    앵커 주변만 갱신하게 제한. 허용 반경 = margin_mult × (앵커 마커 한 변의 픽셀 길이)
    — "마커 몇 칸 거리까지 전이 허용"이라는 물리적 의미(기본 8칸 ≈ 이웃 마커 한 겹).
    반환: 통과한 마커 id 집합 (margin_mult=None이면 게이트 비활성).
    """
    if margin_mult is None:
        return set(rec)
    pts = np.vstack([rec[i] for i in known]).astype(np.float32)
    hull = cv2.convexHull(pts)
    edges = []
    for i in known:
        q = rec[i]
        edges.append(np.mean([np.linalg.norm(q[k] - q[(k+1) % 4]) for k in range(4)]))
    size = max(float(np.mean(edges)), 1.0)          # 마커 한 변(px)
    ok = set()
    for i, p in rec.items():
        c = p.mean(0).astype(np.float32)
        d = cv2.pointPolygonTest(hull, (float(c[0]), float(c[1])), True)
        if d >= -margin_mult * size:
            ok.add(i)
    return ok


def build_map_incremental(obs, marker_len_mm=22.0, seed_id=0,
                          min_anchor_markers=2, gate_obs=2,
                          global_refine=True, max_iters=40, tol_mm=0.02, ransac_px=4.0,
                          interp_margin=8.0):
    """관측들로부터 증분 지도 구축(2단계) → (corners{id:(4,2)mm}, stats, info).

    1단계(전이 성장, BFS): 위치가 잡힌 마커 min_anchor_markers개 이상 보이는 프레임을
      웨이브 단위로 등록(seed만 있을 땐 seed 프레임으로 부트스트랩). hop = 전이 단수.
      ※ 이 단계만으로는 홉마다 오차가 누적된다(단일/소수 마커 외삽).
    2단계(전역 정제, global_refine=True): 모든 프레임을 공통 마커 전부로 반복 재정합
      (간이 번들 조정) → 등록 순서 의존성·전이 드리프트 제거. 검증된 방식(std 0.35mm).

    stats[id] = {n_obs, std_mm, hop}, info = {n_frames_used, n_frames_total, unregistered, iters}
    gate_obs 미만 관측 마커는 stats에 low_conf=True 표시(위치는 유지 — 사용자 판단).
    """
    L = float(marker_len_mm)
    seed_sq = np.array([[0, 0], [L, 0], [L, L], [0, L]], np.float64)
    corners = {seed_id: seed_sq.copy()}
    hop = {seed_id: 0}

    # ---------- 1단계: BFS 전이 성장 ----------
    # 매 웨이브 전 프레임 재평가(앵커가 늘면 같은 프레임이 다시 기여 가능),
    # 새 마커가 더 안 나올 때까지 반복.
    for _wave in range(64):
        acc: dict[int, list[np.ndarray]] = {}
        hop_cand: dict[int, int] = {}
        for rec in obs:
            known = [i for i in rec if i in corners]
            if not (len(known) >= min_anchor_markers or (seed_id in known)):
                continue                                   # seed 프레임=부트스트랩 허용
            H = _frame_H(rec, corners, known, ransac_px)
            if H is None:
                continue
            gate = _hull_gate(rec, known, interp_margin)   # 외삽 금지(주변만 전이)
            img_hop = min(hop[i] for i in known)
            for i, pts in rec.items():
                if i in corners or i not in gate:
                    continue
                proj = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(4, 2)
                acc.setdefault(i, []).append(proj)
                hop_cand[i] = min(hop_cand.get(i, 99), img_hop + 1)
        if not acc:
            break
        for i, plist in acc.items():
            corners[i] = np.stack(plist).mean(0)
            hop[i] = hop_cand[i]
    if len(corners) > 1:
        corners = _fix_gauge(corners, seed_id, L)   # 부트스트랩 스케일 오차 즉시 교정

    # ---------- 2단계: 전역 반복 정제 ----------
    it = 0
    used = 0
    stats: dict = {}
    prev_pos = {i: c.mean(0) for i, c in corners.items()}
    n_pass = max_iters if global_refine else 1     # refine 안 해도 1패스는 돌려 stats 산출
    for it in range(1, n_pass + 1):
        acc = {}
        used = 0
        for rec in obs:
            known = [i for i in rec if i in corners]
            if len(known) < max(2, min_anchor_markers):
                continue
            H = _frame_H(rec, corners, known, ransac_px)
            if H is None:
                continue
            used += 1
            gate = _hull_gate(rec, known, interp_margin)   # 외삽 금지
            for i, pts in rec.items():
                if i not in gate:
                    continue
                proj = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(4, 2)
                acc.setdefault(i, []).append(proj)
        if not acc:
            break
        stats = {}
        for i, plist in acc.items():
            P = np.stack(plist)
            std = float(np.linalg.norm(P.std(0), axis=1).mean())
            stats[i] = {"n_obs": len(plist), "std_mm": std, "hop": hop.get(i, 99),
                        "low_conf": len(plist) < gate_obs}
            if global_refine:
                corners[i] = P.mean(0)          # seed 포함 전부 자유 추정
        if not global_refine:
            break
        corners = _fix_gauge(corners, seed_id, L)   # 스케일(전 마커 22mm)·원점 게이지 고정
        pos = {i: c.mean(0) for i, c in corners.items()}
        common = set(pos) & set(prev_pos)
        if common and max(np.linalg.norm(pos[i] - prev_pos[i]) for i in common) < tol_mm:
            break
        prev_pos = pos

    all_ids = set()
    for rec in obs:
        all_ids |= set(rec)
    info = {"n_frames_used": used, "n_frames_total": len(obs),
            "unregistered": sorted(all_ids - set(corners)), "iters": it}
    if seed_id in stats:
        stats[seed_id]["hop"] = 0
    return corners, stats, info


def align_rigid(corners_a, corners_b):
    """두 지도의 공통 마커 중심으로 2D 강체 정렬(Kabsch, 스케일 없음) → a를 b에 맞춘 사본.

    반환 (aligned_a, common_ids). 스케일은 양쪽 다 실물 마커 크기 기준이라 고정.
    """
    common = sorted(set(corners_a) & set(corners_b))
    A = np.array([corners_a[i].mean(0) for i in common])
    B = np.array([corners_b[i].mean(0) for i in common])
    ca, cb = A.mean(0), B.mean(0)
    Hm = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(Hm)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    out = {i: (R @ (c - ca).T).T + cb for i, c in corners_a.items()}
    return out, common


def compare_maps(corners_test, corners_ref):
    """테스트 지도를 기준 지도에 강체 정렬 후 마커별 중심 오차(mm) → {id: err}."""
    aligned, common = align_rigid(corners_test, corners_ref)
    return {i: float(np.linalg.norm(aligned[i].mean(0) - corners_ref[i].mean(0))) for i in common}


def save_map(corners, path, ref_id=0, marker_len_mm=22.0):
    """marker_map.json 형식으로 저장(기존 파이프라인과 호환)."""
    import json
    markers = [{"id": int(i), "corners_mm": np.asarray(c, float).tolist(),
                "center_mm": np.asarray(c, float).mean(0).tolist()}
               for i, c in sorted(corners.items())]
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"ref_id": ref_id, "marker_len_mm": marker_len_mm, "markers": markers}, f, indent=1)
