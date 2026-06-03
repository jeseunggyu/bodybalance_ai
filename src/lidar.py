"""LiDAR 스캔 파일 처리"""
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np


@dataclass
class FootMeasurement:
    foot_length_mm:    float
    foot_width_mm:     float
    heel_width_mm:     float
    arch_height_mm:    float
    instep_height_mm:  float

    def to_dict(self):
        return asdict(self)


def load_scan(path: str | Path) -> np.ndarray:
    """스캔 파일 로드 → 정규화된 point cloud"""
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError(
            "open3d 패키지가 필요합니다: pip install open3d"
        )
    p = Path(path)
    if p.suffix == ".ply":
        pcd = o3d.io.read_point_cloud(str(p))
    elif p.suffix in (".obj", ".stl"):
        mesh = o3d.io.read_triangle_mesh(str(p))
        pcd = mesh.sample_points_uniformly(number_of_points=50000)
    else:
        raise ValueError(f"지원하지 않는 형식: {p.suffix}")

    pcd, _ = pcd.remove_statistical_outlier(20, 2.0)
    pts = np.asarray(pcd.points)
    if pts.max() < 1.0:
        pts *= 1000  # 미터 → 밀리미터
    return pts


def align_foot(points: np.ndarray) -> np.ndarray:
    """PCA로 표준 좌표계 정렬 (축 방향까지 일관되게 보정)"""
    centered = points - points.mean(axis=0)
    _, eigvecs = np.linalg.eigh(np.cov(centered.T))
    eigvecs = eigvecs[:, ::-1]          # 분산 큰 순: x=길이, y=너비, z=높이
    aligned = centered @ eigvecs

    # z(높이): 바닥이 0이 되도록
    if aligned[:, 2].mean() > 0:
        aligned[:, 2] *= -1
    aligned[:, 2] -= aligned[:, 2].min()

    # x(길이) 방향 보정: 표준 정렬은 '뒤꿈치=x_min, 발가락=x_max'.
    # 발끝(발가락)은 뾰족해 끝으로 갈수록 급히 좁아지고, 뒤꿈치는 둥글어
    # 끝에서도 폭이 어느 정도 유지된다. 양 끝 5% 구간의 폭을 비교해
    # 더 좁은(뾰족한) 끝이 x_max(발가락)에 오도록 맞춘다.
    x = aligned[:, 0]
    L = x.max() - x.min()
    end_lo = aligned[x < x.min() + L * 0.05]
    end_hi = aligned[x > x.max() - L * 0.05]
    w_lo = (end_lo[:, 1].max() - end_lo[:, 1].min()) if len(end_lo) else 0
    w_hi = (end_hi[:, 1].max() - end_hi[:, 1].min()) if len(end_hi) else 0
    if w_lo < w_hi:                     # 좁은(발가락) 끝이 x_min이면 뒤집기
        aligned[:, 0] *= -1
        x = aligned[:, 0]

    return aligned


def extract_measurements(points: np.ndarray) -> FootMeasurement:
    """정렬된 포인트클라우드 → 인체측정 5개 값"""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    L = x.max() - x.min()

    # 발 길이/너비
    # 발 너비는 '발볼(ball)' 부위 = 발 길이의 약 60~78% 지점의 최대 횡폭으로
    # 측정한다(해부학적 ball width). 발가락 끝(상위 0~10%)은 좁아서 제외.
    ball = (x > x.min() + L * 0.60) & (x < x.min() + L * 0.78)
    heel = x < (x.min() + L * 0.2)
    mid  = (x > x.min() + L * 0.4) & (x < x.min() + L * 0.6)

    foot_width = float(y[ball].max() - y[ball].min()) if ball.any() else 0.0
    heel_width = float(y[heel].max() - y[heel].min())

    # 아치 높이 (내측 중족부의 z 최대값)
    arch_height = float(z[mid].max()) if mid.any() else 0.0

    # 발등 높이 (전족부와 중족부 경계의 z 최대값)
    instep_mask = (x > x.min() + L * 0.5) & (x < x.min() + L * 0.7)
    instep_height = float(z[instep_mask].max()) if instep_mask.any() else 0.0

    return FootMeasurement(
        foot_length_mm=    round(L, 1),
        foot_width_mm=     round(foot_width, 1),
        heel_width_mm=     round(heel_width, 1),
        arch_height_mm=    round(arch_height, 1),
        instep_height_mm=  round(instep_height, 1),
    )


def process_scan(file_path: str) -> dict:
    """one-shot: 파일 → 측정값 dict"""
    return extract_measurements(align_foot(load_scan(file_path))).to_dict()