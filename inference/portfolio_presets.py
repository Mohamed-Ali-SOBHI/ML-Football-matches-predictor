from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FrozenStrategy:
    name: str
    train_league: str
    bet_league: str
    outcome: str
    odds_min: float
    odds_max: float
    market_favorite_mode: str
    threshold: float
    edge_min: float
    params: dict[str, float]


EXPLORATORY_MULTI_STRATEGY_PORTFOLIO_2025 = [
    FrozenStrategy(
        name="serie_a_draw_long_nonfavorite",
        train_league="Serie_A",
        bet_league="Serie_A",
        outcome="draw",
        odds_min=4.0,
        odds_max=10.0,
        market_favorite_mode="nonfavorite",
        threshold=0.10,
        edge_min=0.10,
        params={
            "max_depth": 3,
            "min_child_weight": 8.291506340399406,
            "subsample": 0.9800304656711365,
            "colsample_bytree": 0.6789008020969248,
            "gamma": 3.6992337172481085,
            "reg_lambda": 0.6864461853969224,
            "learning_rate": 0.0510878727512436,
        },
    ),
    FrozenStrategy(
        name="epl_draw_long_nonfavorite",
        train_league="EPL",
        bet_league="EPL",
        outcome="draw",
        odds_min=4.0,
        odds_max=10.0,
        market_favorite_mode="nonfavorite",
        threshold=0.30,
        edge_min=0.0,
        params={
            "max_depth": 7,
            "min_child_weight": 2.2598308088957078,
            "subsample": 0.8507813328057123,
            "colsample_bytree": 0.7619932927644096,
            "gamma": 2.2609444259247553,
            "reg_lambda": 6.237491430620192,
            "learning_rate": 0.0562566908000384,
        },
    ),
    FrozenStrategy(
        name="bundesliga_draw_mid_nonfavorite",
        train_league="",
        bet_league="Bundesliga",
        outcome="draw",
        odds_min=2.2,
        odds_max=4.0,
        market_favorite_mode="nonfavorite",
        threshold=0.45,
        edge_min=0.0,
        params={
            "max_depth": 3,
            "min_child_weight": 5.827662837272576,
            "subsample": 0.9363690639601221,
            "colsample_bytree": 0.8638156130767138,
            "gamma": 0.3767093915505981,
            "reg_lambda": 7.817167637275669,
            "learning_rate": 0.06447408062937295,
        },
    ),
    FrozenStrategy(
        name="ligue1_draw_wide_nonfavorite",
        train_league="",
        bet_league="Ligue_1",
        outcome="draw",
        odds_min=2.0,
        odds_max=10.0,
        market_favorite_mode="nonfavorite",
        threshold=0.70,
        edge_min=0.10,
        params={
            "max_depth": 5,
            "min_child_weight": 5.0750567663835575,
            "subsample": 0.7613001150741135,
            "colsample_bytree": 0.6352621115879286,
            "gamma": 0.5196860213418866,
            "reg_lambda": 4.067786946694503,
            "learning_rate": 0.029749107688307467,
        },
    ),
]

VALIDATION_MULTI_STRATEGY_PORTFOLIO_2024 = [
    FrozenStrategy(
        name="bundesliga_draw_long_nonfavorite",
        train_league="",
        bet_league="Bundesliga",
        outcome="draw",
        odds_min=4.0,
        odds_max=10.0,
        market_favorite_mode="nonfavorite",
        threshold=0.60,
        edge_min=0.10,
        params={
            "max_depth": 5,
            "min_child_weight": 1.4818414236595165,
            "subsample": 0.6194302714303965,
            "colsample_bytree": 0.8573720289591046,
            "gamma": 2.9790486236312685,
            "reg_lambda": 7.756322993256575,
            "learning_rate": 0.036178648278979875,
        },
    ),
    FrozenStrategy(
        name="epl_draw_wide_nonfavorite",
        train_league="",
        bet_league="EPL",
        outcome="draw",
        odds_min=2.0,
        odds_max=10.0,
        market_favorite_mode="nonfavorite",
        threshold=0.70,
        edge_min=0.08,
        params={
            "max_depth": 5,
            "min_child_weight": 1.4818414236595165,
            "subsample": 0.6194302714303965,
            "colsample_bytree": 0.8573720289591046,
            "gamma": 2.9790486236312685,
            "reg_lambda": 7.756322993256575,
            "learning_rate": 0.036178648278979875,
        },
    ),
    FrozenStrategy(
        name="serie_a_draw_wide_nonfavorite",
        train_league="",
        bet_league="Serie_A",
        outcome="draw",
        odds_min=2.0,
        odds_max=10.0,
        market_favorite_mode="nonfavorite",
        threshold=0.55,
        edge_min=0.08,
        params={
            "max_depth": 7,
            "min_child_weight": 9.338965140939113,
            "subsample": 0.7095366856584409,
            "colsample_bytree": 0.9868141109777064,
            "gamma": 3.5724844852887907,
            "reg_lambda": 6.337876228053214,
            "learning_rate": 0.027651516010377894,
        },
    ),
    FrozenStrategy(
        name="laliga_draw_mid_nonfavorite",
        train_league="",
        bet_league="La_liga",
        outcome="draw",
        odds_min=2.2,
        odds_max=4.0,
        market_favorite_mode="nonfavorite",
        threshold=0.45,
        edge_min=0.0,
        params={
            "max_depth": 3,
            "min_child_weight": 5.827662837272576,
            "subsample": 0.9363690639601221,
            "colsample_bytree": 0.8638156130767138,
            "gamma": 0.3767093915505981,
            "reg_lambda": 7.817167637275669,
            "learning_rate": 0.06447408062937295,
        },
    ),
]


DEFAULT_PORTFOLIO_NAME = "validation_multi_strategy_portfolio_2024"
PORTFOLIO_PRESETS = {
    DEFAULT_PORTFOLIO_NAME: VALIDATION_MULTI_STRATEGY_PORTFOLIO_2024,
    "exploratory_multi_strategy_portfolio_2025": EXPLORATORY_MULTI_STRATEGY_PORTFOLIO_2025,
}
