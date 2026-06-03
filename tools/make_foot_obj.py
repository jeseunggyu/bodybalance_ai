"""
시연용 합성 발 3D 메시(.obj) 생성기

실제 발 스캔 데이터를 즉시 받기 어려워, 앱 LiDAR 파이프라인(src/lidar.py)을
테스트·시연하기 위한 그럴듯한 발 형상 메시를 생성한다.

발을 길이(x) 방향의 단면 스택으로 모델링:
  - 단면은 반타원(발바닥 z=0 평평, 발등 위로 볼록)
  - 너비/높이가 위치에 따라 변함: 뒤꿈치(둥글고 좁음) → 중족부(아치로 안쪽 들림)
    → 전족부(가장 넓음) → 발가락(테이퍼)
  - 좌/우는 mirror(y축 반전) + 치수 파라미터로 비대칭 구현

단위: mm (앱은 max<1이면 ×1000 하지만 여기선 mm로 바로 출력)
"""
import argparse
import math
from pathlib import Path
import numpy as np


def _profile(t):
    """t in [0,1] (heel→toe) → (width_scale, height_scale, y_center_offset)
    width/height는 최대치 대비 비율, y_center_offset은 아치 안쪽 들림 표현(0~1)."""
    # 길이방향 너비 프로파일(t: 0=뒤꿈치 → 1=발가락).
    # 실제 발 비율: 뒤꿈치 0~25%, 아치 25~55%, 발볼(최대너비) 55~72%,
    # 발가락 72~100%. 발볼을 t≈0.68에 둬 측정 ball 구간(60~78%)과 일치.
    if t < 0.25:                      # 뒤꿈치(둥글고 중간 폭)
        w = 0.62 + (t / 0.25) * 0.04
    elif t < 0.55:                    # 중족부(아치, 약간 좁아짐)
        w = 0.66 - (t - 0.25) / 0.30 * 0.06
    elif t < 0.68:                    # 발볼로 넓어짐 → t=0.68 최대
        w = 0.60 + (t - 0.55) / 0.13 * 0.40
    else:                             # 발가락 테이퍼
        w = 1.0 - (t - 0.68) / 0.32 * 0.55
    w = max(w, 0.26)

    # 높이 프로파일: 뒤꿈치 높음, 중족부 발등 가장 높음, 발가락 낮음
    if t < 0.1:
        h = 0.55 + (t / 0.1) * 0.15
    elif t < 0.5:
        h = 0.70 + (t - 0.1) / 0.4 * 0.30   # 발등 최고
    elif t < 0.75:
        h = 1.0 - (t - 0.5) / 0.25 * 0.45
    else:
        h = 0.55 - (t - 0.75) / 0.25 * 0.35
    h = max(h, 0.12)

    # 아치: 중족부(0.3~0.6) 안쪽에서 바닥이 살짝 떠오름
    arch = 0.0
    if 0.30 < t < 0.62:
        arch = math.sin((t - 0.30) / 0.32 * math.pi)  # 0→1→0
    return w, h, arch


def make_foot(length_mm=260.0, width_mm=98.0, height_mm=62.0,
              arch_mm=20.0, side="L",
              n_len=80, n_theta=40):
    """발 메시 생성 → (vertices Nx3, faces Mx3)"""
    verts = []
    grid = np.full((n_len, n_theta), -1, dtype=int)

    half_w = width_mm / 2.0
    for i in range(n_len):
        t = i / (n_len - 1)
        w_s, h_s, arch = _profile(t)
        sec_w = half_w * w_s
        sec_h = height_mm * h_s
        x = t * length_mm
        arch_lift = arch_mm * arch  # 바닥 들림 높이

        for j in range(n_theta):
            # theta: 0=바깥쪽 바닥 → pi=안쪽 바닥, 위로 반원
            theta = math.pi * j / (n_theta - 1)
            y = -sec_w * math.cos(theta)            # -sec_w..+sec_w
            z = sec_h * math.sin(theta)             # 0..sec_h..0 (반타원 위쪽)
            # 아치: 안쪽(y>0)의 바닥 근처를 들어올림
            if z < arch_lift and y > 0:
                z = arch_lift * (y / sec_w)
            verts.append((x, y, z))
            grid[i, j] = len(verts) - 1

    faces = []
    for i in range(n_len - 1):
        for j in range(n_theta - 1):
            a, b = grid[i, j], grid[i, j + 1]
            c, d = grid[i + 1, j], grid[i + 1, j + 1]
            faces.append((a, c, b))
            faces.append((b, c, d))

    # 양 끝(뒤꿈치/발가락) 캡: 중심점으로 팬
    for end_i, reverse in [(0, True), (n_len - 1, False)]:
        cx = verts[grid[end_i, 0]][0]
        ring = [grid[end_i, j] for j in range(n_theta)]
        cy = np.mean([verts[k][1] for k in ring])
        cz = np.mean([verts[k][2] for k in ring]) * 0.5
        verts.append((cx, cy, cz))
        center = len(verts) - 1
        for j in range(n_theta - 1):
            a, b = ring[j], ring[j + 1]
            faces.append((center, b, a) if reverse else (center, a, b))

    V = np.array(verts, dtype=float)
    if side.upper() == "R":
        V[:, 1] *= -1   # 우발은 y 반전(mirror)
    return V, np.array(faces, dtype=int)


def write_obj(path, V, F):
    lines = [f"# synthetic foot mesh for demo ({len(V)} verts)"]
    lines += [f"v {x:.3f} {y:.3f} {z:.3f}" for x, y, z in V]
    lines += [f"f {a+1} {b+1} {c+1}" for a, b, c in F]   # OBJ는 1-based
    Path(path).write_text("\n".join(lines), encoding="utf-8")


PRESETS = {
    "L": dict(length_mm=262.0, width_mm=99.0, height_mm=63.0, arch_mm=22.0, side="L"),
    # 우발: 의도적으로 약간 작고 아치 낮게 → 좌우 비대칭 시연용
    "R": dict(length_mm=255.0, width_mm=94.0, height_mm=60.0, arch_mm=15.0, side="R"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="sample_data")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    for name, params in PRESETS.items():
        V, F = make_foot(**params)
        fp = out / f"foot_{name}.obj"
        write_obj(fp, V, F)
        print(f"  [OK] {fp}  (verts={len(V)}, faces={len(F)}, "
              f"L={params['length_mm']}mm W={params['width_mm']}mm)")


if __name__ == "__main__":
    main()
