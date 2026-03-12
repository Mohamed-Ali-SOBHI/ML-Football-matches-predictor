from __future__ import annotations

from validation_context import ValidationContext
from validation_verdict import ValidationVerdict


MONTH_NAMES_FR = {
    1: "janvier",
    2: "fevrier",
    3: "mars",
    4: "avril",
    5: "mai",
    6: "juin",
    7: "juillet",
    8: "aout",
    9: "septembre",
    10: "octobre",
    11: "novembre",
    12: "decembre",
}


def format_date_fr(day_value) -> str:
    return f"{day_value.day} {MONTH_NAMES_FR[day_value.month]} {day_value.year}"


def render_group_lines(rows: list[dict], *, key: str, empty_label: str) -> list[str]:
    if not rows:
        return [f"- {empty_label}"]
    return [
        f"- `{row[key]}` : `{row['bets']}` paris, ROI `{row['roi']*100:.2f}%`, profit `{row['profit']:.2f}`"
        for row in rows
    ]


def render_verdict_lines(items: list[str], *, fallback: str) -> list[str]:
    if not items:
        return [f"- {fallback}"]
    return [f"- {item}" for item in items]


def build_markdown_report(
    *,
    context: ValidationContext,
    metrics: dict,
    verdict: ValidationVerdict,
    monthly_rows: list[dict],
    league_rows: list[dict],
    strategy_rows: list[dict],
) -> str:
    summary_text = str(context.summary_path) if context.summary_path else "None"
    strategy_key = "strategy_name" if strategy_rows and "strategy_name" in strategy_rows[0] else "strategy_names"

    monthly_lines = render_group_lines(
        monthly_rows,
        key="month",
        empty_label="Aucun regroupement mensuel disponible.",
    )
    league_lines = render_group_lines(
        league_rows[:8],
        key="league",
        empty_label="Aucun regroupement par ligue disponible.",
    )
    strategy_lines = render_group_lines(
        strategy_rows[:8],
        key=strategy_key,
        empty_label="Aucun regroupement par strategie disponible.",
    )
    strength_lines = render_verdict_lines(verdict.strengths, fallback="Aucun point fort net.")
    risk_lines = render_verdict_lines(verdict.risks, fallback="Aucun risque majeur identifie.")

    current_date_text = format_date_fr(context.current_date)
    next_step = (
        f"Comme nous sommes le {current_date_text} et que la saison en cours est 2025/26, "
        "le prochain test prospectif propre n'est pas oblige d'attendre 2026/27. "
        "Le bon test live a geler maintenant est la fin de saison 2025/26, "
        f"sur les matchs joues apres le {current_date_text}."
    )

    return f"""# Rapport de validation scientifique

## Snapshot

- Date du rapport : `{context.current_date.isoformat()}`
- Fichier paris : `{context.bets_path}`
- Fichier resume : `{summary_text}`
- Mode de selection detecte : `{context.selection_mode}`
- Nombre de strategies selectionnees : `{context.strategy_count if context.strategy_count is not None else 'unknown'}`

## Chiffres cles

- Nombre de paris : `{metrics['bet_count']}`
- Profit total : `{metrics['total_profit']:.2f}` unites
- ROI moyen : `{metrics['roi']*100:.2f}%`
- IC bootstrap 95% du ROI : `[{metrics['roi_ci_low']*100:.2f}%; {metrics['roi_ci_high']*100:.2f}%]`
- Proba bootstrap que le ROI soit > 0 : `{metrics['bootstrap_prob_roi_positive']*100:.2f}%`
- Hit rate : `{metrics['hit_rate']*100:.2f}%`
- IC bootstrap 95% du hit rate : `[{metrics['hit_rate_ci_low']*100:.2f}%; {metrics['hit_rate_ci_high']*100:.2f}%]`
- Cote moyenne : `{metrics['avg_odds']:.2f}`
- Edge moyen : `{metrics['avg_edge']:.4f}`
- EV moyen : `{metrics['avg_expected_value']:.4f}`
- Max drawdown : `{metrics['max_drawdown']:.2f}` unites
- Plus longue serie de pertes : `{metrics['longest_losing_streak']}`
- Periode couverte : `{metrics['start_date']} -> {metrics['end_date']}`

## Verdict

Niveau de preuve actuel : `{verdict.evidence_level}`

Points qui vont dans le bon sens :
{chr(10).join(strength_lines)}

Points qui empechent encore de parler de robustesse forte :
{chr(10).join(risk_lines)}

## Lecture correcte

- Ce rapport ne peut pas prouver mathematiquement que la strategie gagnera dans le futur.
- Il peut seulement mesurer si le signal observe a l'air fragile, moyen ou encourageant.
- Le vrai test propre reste une strategie gelee suivie en live sans retouche.

## Repartition mensuelle

{chr(10).join(monthly_lines)}

## Repartition par ligue

{chr(10).join(league_lines)}

## Repartition par strategie

{chr(10).join(strategy_lines)}

## Ce qu'il faut faire maintenant

- Geler le portefeuille utilise en live.
- Logger chaque pari recommande avec sa date de generation.
- Ajouter plus tard la cote de cloture pour mesurer le CLV.
- Evaluer uniquement les matchs joues apres la date de gel.

{next_step}
"""
