from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from validation_context import build_validation_context
from validation_io import load_bets, load_summary, resolve_path
from validation_markdown import build_markdown_report
from validation_metrics import group_roi_table, monthly_roi_table, summarize_bets
from validation_verdict import build_validation_verdict


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BETS_PATH = SCRIPT_DIR / "output" / "positive_strategy_portfolio_bets.csv"
DEFAULT_SUMMARY_PATH = SCRIPT_DIR / "output" / "positive_strategy_portfolio_summary.csv"


def default_report_path(bets_path: Path, suffix: str) -> Path:
    return bets_path.with_name(f"{bets_path.stem}_scientific_report.{suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bets", default=str(DEFAULT_BETS_PATH))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-md", default="")
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bets_path = resolve_path(args.bets)
    summary_path = resolve_path(args.summary) if args.summary else None

    bets_df = load_bets(bets_path)
    summary_df = load_summary(summary_path)

    metrics = summarize_bets(
        bets_df,
        iterations=args.bootstrap_iterations,
        confidence_level=args.confidence_level,
        seed=args.seed,
    )
    league_rows = group_roi_table(bets_df, "league")
    strategy_group_col = "strategy_name" if "strategy_name" in bets_df.columns else "strategy_names"
    strategy_rows = group_roi_table(bets_df, strategy_group_col)
    monthly_rows = monthly_roi_table(bets_df)
    current_date = date.today()
    context = build_validation_context(
        bets_df=bets_df,
        summary_df=summary_df,
        bets_path=bets_path,
        summary_path=summary_path,
        current_date=current_date,
    )
    verdict = build_validation_verdict(metrics, context=context)
    report_md = build_markdown_report(
        context=context,
        metrics=metrics,
        monthly_rows=monthly_rows,
        league_rows=league_rows,
        strategy_rows=strategy_rows,
        verdict=verdict,
    )

    output_md = resolve_path(args.output_md) if args.output_md else default_report_path(bets_path, "md")
    output_json = resolve_path(args.output_json) if args.output_json else default_report_path(bets_path, "json")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "bets_path": str(bets_path),
        "summary_path": str(summary_path) if summary_path else None,
        "selection_mode": context.selection_mode,
        "strategy_count": context.strategy_count,
        "clv_available": context.clv_available,
        "metrics": metrics,
        "verdict": verdict.to_dict(),
        "monthly_rows": monthly_rows,
        "league_rows": league_rows,
        "strategy_rows": strategy_rows,
        "generated_on": current_date.isoformat(),
    }

    output_md.write_text(report_md, encoding="utf-8")
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        {
            "bets": str(bets_path),
            "selection_mode": context.selection_mode,
            "bet_count": metrics["bet_count"],
            "roi": round(metrics["roi"], 6) if metrics["roi"] is not None else None,
            "roi_ci_low": round(metrics["roi_ci_low"], 6) if metrics["roi_ci_low"] is not None else None,
            "roi_ci_high": round(metrics["roi_ci_high"], 6) if metrics["roi_ci_high"] is not None else None,
            "prob_roi_positive": round(metrics["bootstrap_prob_roi_positive"], 6)
            if metrics["bootstrap_prob_roi_positive"] is not None
            else None,
            "evidence_level": verdict.evidence_level,
            "output_md": str(output_md),
            "output_json": str(output_json),
        }
    )


if __name__ == "__main__":
    main()
