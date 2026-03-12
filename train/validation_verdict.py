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
    clv_metrics: dict[str, Any] | None = None,
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

    if not context.clv_available:
        reasons.append("Le CLV n'est pas encore mesurable car les cotes de cloture ne sont pas stockees.")
    else:
        matched_bet_count = int((clv_metrics or {}).get("matched_bet_count") or 0)
        positive_clv_rate = (clv_metrics or {}).get("positive_clv_rate")
        avg_clv_odds_diff = (clv_metrics or {}).get("avg_clv_odds_diff")

        if matched_bet_count >= bet_count and bet_count > 0:
            strengths.append("La closing line historique est disponible sur tous les paris du test 2025/26.")
            score += 1
        else:
            reasons.append("La couverture historique de closing line est incomplete sur le test 2025/26.")

        if (
            positive_clv_rate is not None
            and avg_clv_odds_diff is not None
            and positive_clv_rate > 0.5
            and avg_clv_odds_diff > 0.0
        ):
            strengths.append("Le CLV historique est positif : le portefeuille prend en moyenne une meilleure cote que la cloture.")
            score += 1
        else:
            reasons.append("Le CLV historique sur 2025/26 n'est pas positif.")
            score -= 1

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
