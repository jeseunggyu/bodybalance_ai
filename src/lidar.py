"""LiDAR 스캔 파일 처리"""
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import open3d as o3d


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
    """PCA로 표준 좌표계 정렬"""
    centered = points - points.mean(axis=0)
    _, eigvecs = np.linalg.eigh(np.cov(centered.T))
    eigvecs = eigvecs[:, ::-1]
    aligned = centered @ eigvecs

    if aligned[:, 2].mean() > 0:
        aligned[:, 2] *= -1
    aligned[:, 2] -= aligned[:, 2].min()
    return aligned


def extract_measurements(points: np.ndarray) -> FootMeasurement:
    """정렬된 포인트클라우드 → 인체측정 5개 값"""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    L = x.max() - x.min()

    # 발 길이/너비
    fore = x > (x.max() - L * 0.3)
    heel = x < (x.min() + L * 0.2)
    mid  = (x > x.min() + L * 0.4) & (x < x.min() + L * 0.6)

    foot_width = float(y[fore].max() - y[fore].min())
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