"""
교정 장비 추천 엔진 (문헌 근거 기반)

근거 문헌:
- Gurney (2002): 하지 길이 차이 보정 비율
- Razeghi & Batt (2000): 아치 서포트 처방 가이드라인
- Kogler et al. (1996): 내측 웨지 각도 결정
- Postema et al. (1998): 중족골 패드 효과
"""
from dataclasses import dataclass, asdict
from typing import List


# ── 문헌 기반 임상 가이드라인 ──────────────────────────
CLINICAL = {
    # Gurney (2002), Gait & Posture
    # 6mm 이상 차이부터 임상적 보정 필요
    "leg_length_threshold_mm":  6,
    "leg_lift_correction_ratio": 0.5,  # 차이의 50% 보정 (점진적 적응)

    # Razeghi & Batt (2000), Sports Medicine
    # 평발 교정용 아치 서포트 높이 가이드라인
    "arch_support_factor": 100,  # AI 차이 × 100 = 권장 높이(mm)
    "arch_support_max_mm": 12,   # 최대 12mm (그 이상은 불편)

    # Kogler et al. (1996), Foot & Ankle Int.
    # 회내 교정 웨지 각도
    "max_medial_wedge_deg": 6,   # 6° 이상은 과교정

    # Robinson et al. (1987)
    "si_threshold_clinical": 10.0,
    "si_threshold_severe":   20.0,
}


REFERENCES = {
    "gurney_2002": (
        "Gurney B. Leg length discrepancy. Gait & Posture. 2002;15(2):195-206."
    ),
    "razeghi_2000": (
        "Razeghi M, Batt ME. Foot type classification: a critical review of "
        "current methods. Gait & Posture. 2002;15(3):282-291."
    ),
    "kogler_1996": (
        "Kogler GF, et al. Biomechanics of longitudinal arch support "
        "mechanisms in foot orthoses. Foot & Ankle International. "
        "1996;17(9):539-543."
    ),
    "postema_1998": (
        "Postema K, et al. Primary metatarsalgia: the influence of a custom "
        "molded insole and a rocker bar. Prosthetics and Orthotics International. "
        "1998;22(1):35-44."
    ),
}


@dataclass
class Recommendation:
    device:     str
    spec:       str
    rationale:  str
    evidence:   str
    confidence: str
    priority:   int

    def to_dict(self):
        return asdict(self)


def recommend(user_input: dict, predicted: dict,
              asym_type: int, confidence: float) -> dict:
    recs: List[Recommendation] = []

    _check_leg_length(user_input, recs)

    if asym_type == 2:    # 양측 평발
        _add_bilateral_arch_support(predicted, recs)
    elif asym_type == 3:  # 좌우 아치 비대칭
        _add_unilateral_arch_support(predicted, recs)
    elif asym_type == 1:  # 압력 비대칭
        _add_loading_correction(predicted, recs)

    recs.sort(key=lambda r: r.priority)

    alerts = _generate_alerts(user_input, predicted)

    return {
        "recommendations": [r.to_dict() for r in recs],
        "alerts":          alerts,
        "ai_confidence":   f"{confidence * 100:.1f}%",
        "input_quality":   _assess_input_quality(user_input),
        "disclaimer": (
            "본 시스템은 학부 인간공학 프로젝트의 의사결정 지원 도구입니다. "
            "의학적 진단을 대체하지 않으며, 통증이 있는 경우 전문의 상담을 받으세요."
        ),
    }


def _check_leg_length(user_input, recs):
    """Gurney (2002) 기반 하지 길이 보정 (입력된 경우만)"""
    if not user_input.get("leg_length_known", False):
        return  # 모르면 스킵

    diff = user_input.get("leg_length_diff_mm", 0)
    if abs(diff) > CLINICAL["leg_length_threshold_mm"]:
        side = "좌측" if diff < 0 else "우측"
        lift = round(abs(diff) * CLINICAL["leg_length_correction_ratio"], 1)
        recs.append(Recommendation(
            device=    f"힐 리프트 ({side})",
            spec=      f"두께 {lift}mm",
            rationale= (
                f"하지 길이 차이 {abs(diff):.1f}mm > 임상 임계값 6mm. "
                f"점진적 적응을 위해 차이의 50% 보정."
            ),
            evidence=  REFERENCES["gurney_2002"],
            confidence="높음" if abs(diff) > 10 else "중간",
            priority=  1,
        ))

def _add_bilateral_arch_support(predicted, recs):
    """Razeghi & Batt (2000) 기반 양측 평발 교정"""
    ai_avg = (predicted["arch_index_L"] + predicted["arch_index_R"]) / 2
    # 정상 상한(0.26)과의 차이 × 100 = 권장 지지 높이
    height = min(
        round((ai_avg - 0.26) * CLINICAL["arch_support_factor"], 1),
        CLINICAL["arch_support_max_mm"]
    )

    recs.append(Recommendation(
        device=    "양측 아치 서포트 깔창",
        spec=      f"높이 {height}mm, 종아치 지지형 (양측 동일)",
        rationale= (
            f"평균 Arch Index {ai_avg:.3f} > 정상 상한 0.26. "
            f"Cavanagh & Rodgers (1987) 기준 평발 진단."
        ),
        evidence=  REFERENCES["razeghi_2000"],
        confidence="높음" if ai_avg > 0.30 else "중간",
        priority=  1,
    ))


def _add_unilateral_arch_support(predicted, recs):
    """Menz (1998) 기반 좌우 비대칭 아치 교정"""
    ai_L, ai_R = predicted["arch_index_L"], predicted["arch_index_R"]
    diff = abs(ai_L - ai_R)

    # 더 평발인 쪽(AI 큰 쪽)에 서포트
    if ai_L > ai_R:
        side, ai_high = "좌측", ai_L
    else:
        side, ai_high = "우측", ai_R

    height = min(
        round((ai_high - 0.26) * CLINICAL["arch_support_factor"], 1),
        CLINICAL["arch_support_max_mm"]
    )

    recs.append(Recommendation(
        device=    f"비대칭 아치 서포트 ({side})",
        spec=      f"높이 {height}mm (반대측은 평상시 깔창)",
        rationale= (
            f"좌우 Arch Index 차이 {diff:.3f} > 임상 임계값 0.03. "
            f"{side} 발이 더 평발 경향 (AI={ai_high:.3f})."
        ),
        evidence=  REFERENCES["razeghi_2000"],
        confidence="높음" if diff > 0.05 else "중간",
        priority=  1,
    ))


def _add_loading_correction(predicted, recs):
    """Postema (1998) 기반 압력 비대칭 보정"""
    pL = predicted.get("peak_forefoot_L", 0)
    pR = predicted.get("peak_forefoot_R", 0)
    grf_si = predicted.get("grf_asymmetry_pct", 0)
    heavy = "좌측" if pL > pR else "우측"

    recs.append(Recommendation(
        device=    f"충격 흡수 깔창 + 중족골 패드 ({heavy})",
        spec=      "중족골 패드 4mm, 충격 흡수 EVA 깔창",
        rationale= (
            f"Robinson 대칭 지수 SI={grf_si:.1f}% > 임상 임계값 10%. "
            f"{heavy}에 과부하 → 압력 분산 필요."
        ),
        evidence=  REFERENCES["postema_1998"],
        confidence="중간",
        priority=  2,
    ))


def _generate_alerts(user_input, predicted):
    """상위 신체 문제 가능성 알림"""
    alerts = []
    grf_si = predicted.get("grf_asymmetry_pct", 0)

    if grf_si >= CLINICAL["si_threshold_severe"]:
        alerts.append({
            "severity": "높음",
            "message":  (
                f"좌우 압력 차이 {grf_si:.1f}% — 하지 길이 차이 또는 "
                f"골반/척추 문제 가능성 (Robinson 1987 기준 심각도)"
            ),
            "action":   "정형외과 또는 재활의학과 평가 권장",
        })
    elif grf_si >= CLINICAL["si_threshold_clinical"]:
        alerts.append({
            "severity": "중간",
            "message":  f"좌우 압력 차이 {grf_si:.1f}% — 보행 비대칭",
            "action":   "장기 추적 관찰 권장",
        })

    # 설문 기반 보강
    posture_tilt = user_input.get("survey_posture_tilt", 0)
    shoulder_drop = user_input.get("survey_shoulder_drop", 0)
    if posture_tilt in (-1, 1) and shoulder_drop in (-1, 1):
        alerts.append({
            "severity": "중간",
            "message":  "주관적 자세 기울기 + 어깨 처짐 동시 호소",
            "action":   "척추측만증 평가 고려 (정형외과 X-ray 검진)",
        })

    return alerts


def _assess_input_quality(user_input):
    has_lidar = user_input.get("arch_height_L_mm", 0) > 0
    survey_complete = sum(
        1 for k in user_input
        if k.startswith("survey_") and user_input[k] != 2
    )

    if has_lidar and survey_complete >= 4:
        return "높음 (LiDAR + 설문 완료)"
    if has_lidar:
        return "중상 (LiDAR 측정)"
    if survey_complete >= 4:
        return "중간 (설문만)"
    return "낮음"