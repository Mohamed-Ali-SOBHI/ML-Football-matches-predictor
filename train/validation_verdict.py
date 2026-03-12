from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from validation_context import ValidationContext


@dataclass(frozen=True)
class ValidationVerdict:
    evidence_level: str
    strengths: list[str]
    risks: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_validation_verdict(
    metrics: dict[str, Any],
    *,
    context: ValidationContext,
) -> ValidationVerdict:
    reasons: list[str] = []
    strengths: list[str] = []
    score = 0

    roi = metrics.get("roi")
    roi_ci_low = metrics.get("roi_ci_low")
    bet_count = metrics.get("bet_count") or 0

    if context.selection_mode == "val":
        strengths.append("Le portefeuille a ete choisi sur une saison de validation separee.")
        score += 1
    elif context.selection_mode == "test":
        reasons.append("Le portefeuille a ete choisi en regardant la meme saison 2025/26 qu'il gagne.")
        score -= 2
    else:
        reasons.append("Le mode de selection du portefeuille n'est pas clairement identifie.")
        score -= 1

    if roi is not None and roi > 0.0:
        strengths.append("Le ROI observe est positif.")
        score += 1
    else:
        reasons.append("Le ROI observe n'est pas positif.")
        score -= 1

    if roi_ci_low is not None and roi_ci_low > 0.0:
        strengths.append("La borne basse bootstrap du ROI reste au-dessus de zero.")
        score += 1
    else:
        reasons.append("L'intervalle bootstrap du ROI recouvre encore zero.")

    if bet_count >= 200:
        strengths.append("Le volume de paris commence a etre correct.")
        score += 1
    elif bet_count >= 100:
        reasons.append("Le volume de paris est encore moyen pour parler de robustesse forte.")
    else:
        reasons.append("Le volume de paris est trop faible pour conclure fortement.")
        score -= 1

    if context.clv_available:
        strengths.append("Des donnees de closing line sont disponibles pour verifier le CLV.")
        score += 1
    else:
        reasons.append("Le CLV n'est pas encore mesurable car les cotes de cloture ne sont pas stockees.")

    if score >= 3:
        evidence_level = "encourageante"
    elif score >= 1:
        evidence_level = "moyenne"
    else:
        evidence_level = "faible"

    return ValidationVerdict(
        evidence_level=evidence_level,
        strengths=strengths,
        risks=reasons,
    )
