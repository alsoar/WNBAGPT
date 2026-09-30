"""Research-only WNBA totals probabilities from the uploaded history.

No network, credentials, order placement, or changes to the source data.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
GAMES = ROOT / "data/processed/wnba_history/games.csv"
SECONDS = ROOT / "data/processed/wnba_history/seconds"
AUDIT = ROOT / "wnba-trade-audit.sqlite3"
CURRENT_TABLE = ROOT / "data/processed/wnba_history_2026/ou_differential_over_table.csv"
REGULATION_SECONDS = 2400
FORM_GAMES = 5
HALF_LIFE_DAYS = 365.25 * 1.5


def load_games() -> pd.DataFrame:
    games = pd.read_csv(GAMES, dtype={"game_id": str, "home_team_id": str, "away_team_id": str})
    games["game_date"] = pd.to_datetime(games["game_date"])
    games = games.loc[games["timeline_available"] & games["total_points"].notna()].copy()
    return games.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def load_states(seconds: list[int], seasons: list[int] | None = None) -> pd.DataFrame:
    paths = sorted(SECONDS.glob("season_end=*/part-0.parquet"))
    if seasons is not None:
        paths = [p for p in paths if int(p.parent.name.split("=")[1]) in seasons]
    frames = []
    for path in paths:
        frame = pd.read_parquet(
            path,
            columns=["game_id", "second", "home_score", "away_score"],
            filters=[("second", "in", seconds)],
        )
        frames.append(frame)
    if not frames:
        raise ValueError("No timeline files for the requested seasons")
    states = pd.concat(frames, ignore_index=True)
    states["game_id"] = states["game_id"].astype(str)
    states["score"] = states["home_score"] + states["away_score"]
    return states[["game_id", "second", "score"]]


def form_features(games: pd.DataFrame) -> pd.DataFrame:
    """Use only games on earlier dates, including when teams play twice on a date."""
    histories: dict[str, deque] = defaultdict(lambda: deque(maxlen=FORM_GAMES))
    records = []
    for game_date, day in games.groupby("game_date", sort=True):
        for game in day.itertuples():
            home = list(histories[game.home_team_id])
            away = list(histories[game.away_team_id])
            if len(home) >= 3 and len(away) >= 3 and pd.notna(game.ou_line):
                # Each entry is the combined score in one of that team's previous games.
                recent_total = (np.mean(home) + np.mean(away)) / 2
                form_gap = recent_total - float(game.ou_line)
            else:
                recent_total, form_gap = np.nan, np.nan
            records.append((game.game_id, recent_total, form_gap))
        for game in day.itertuples():
            histories[game.home_team_id].append(float(game.total_points))
            histories[game.away_team_id].append(float(game.total_points))
    return pd.DataFrame(records, columns=["game_id", "recent_total", "form_gap"])


def recent_total_for_match(games: pd.DataFrame, asof: pd.Timestamp, home: str, away: str) -> float:
    prior = games.loc[games.game_date < asof]
    totals = []
    for team in (home, away):
        recent = prior.loc[(prior.home_team == team) | (prior.away_team == team)].tail(FORM_GAMES)
        if len(recent) < 3:
            return math.nan
        totals.append(float(recent.total_points.mean()))
    return float(np.mean(totals))


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values, sorted_weights = values[order], weights[order]
    return float(sorted_values[np.searchsorted(np.cumsum(sorted_weights), weights.sum() / 2)])


def sample_weights(sample: pd.DataFrame, asof: pd.Timestamp, playoff: bool) -> np.ndarray:
    age = (asof - sample.game_date).dt.days.to_numpy()
    weights = np.exp2(-age / HALF_LIFE_DAYS)
    # Prefer the current season, especially with the 2026 scoring level shift.
    current_season = int(sample.season_end.max())
    weights *= np.where(sample.season_end.to_numpy() == current_season, 3.0, 1.0)
    if playoff:
        weights *= np.where(
            (sample.season_type.to_numpy() == "postseason")
            & (sample.season_end.to_numpy() >= current_season - 4),
            3.0,
            1.0,
        )
    return weights


def over_probability(remaining: np.ndarray, differential: float, adjustment: float,
                     weights: np.ndarray) -> float:
    """Spread a fractional point shift between adjacent integer outcomes."""
    hits = np.clip(remaining - differential + adjustment + 0.5, 0, 1)
    return float(np.average(hits, weights=weights))


def fit_form_coefficient(games: pd.DataFrame, asof: pd.Timestamp, playoff: bool) -> float:
    training = games.loc[
        (games.game_date < asof) & games.form_gap.notna()
        & (games.ou_line_status == "explicit_close")
        & (games.game_date >= asof - pd.Timedelta(days=365 * 4))
    ]
    if len(training) < 80:
        return 0.0
    weights = sample_weights(training, asof, playoff)
    x = training.form_gap.to_numpy(dtype=float)
    y = (training.total_points - training.ou_line).to_numpy(dtype=float)
    # Zero is the prior: the market line may already include recent team form.
    coefficient = np.sum(weights * x * y) / (np.sum(weights * x * x) + 10000.0)
    return float(np.clip(coefficient, -0.5, 0.5))


def predict_from_sample(sample: pd.DataFrame, *, asof: pd.Timestamp, second: int,
                        score: int, line: float, market_median: float,
                        recent_total: float = math.nan, coefficient: float = 0.0,
                        model: str = "replica") -> dict:
    if not (0 <= second < REGULATION_SECONDS):
        raise ValueError("Only live regulation seconds 0..2399 are supported")
    if not math.isclose(line % 1, 0.5, abs_tol=1e-9):
        raise ValueError("The line must end in .5; integer lines can push")
    if sample.empty:
        raise ValueError("No prior games reached this second")
    if len(sample) < 30:
        raise ValueError("Fewer than 30 prior games reached this second")
    remaining = (sample.total_points - sample.score).to_numpy(dtype=float)
    weights = sample_weights(sample, asof, model == "playoff")
    if model in ("replica", "team"):
        weights = np.ones(len(sample))
    historical_median = weighted_median(sample.total_points.to_numpy(dtype=float), weights)
    decay = (REGULATION_SECONDS - second) / REGULATION_SECONDS
    anchor_shift = (market_median - historical_median) * decay
    team_shift = 0.0 if not np.isfinite(recent_total) else coefficient * (recent_total - market_median) * decay
    differential = line - score
    probability = over_probability(remaining, differential, anchor_shift + team_shift, weights)
    return {
        "model": model,
        "probability_over": probability,
        "probability_under": 1.0 - probability,
        "historical_median": historical_median,
        "anchor_shift_points": anchor_shift,
        "team_shift_points": team_shift,
        "recent_team_total": None if not np.isfinite(recent_total) else recent_total,
        "form_coefficient": coefficient,
        "sample_games": len(sample),
        "effective_sample_games": float(weights.sum() ** 2 / np.sum(weights ** 2)),
    }


def sample_for_state(games: pd.DataFrame, states: pd.DataFrame, asof: pd.Timestamp,
                     second: int, season: int, replica: bool) -> pd.DataFrame:
    prior = games.loc[games.game_date < asof]
    if replica:
        prior = prior.loc[prior.season_end == season]
    sample = states.loc[states.second == second].merge(prior, on="game_id", how="inner")
    return sample


def prediction_set(games: pd.DataFrame, states: pd.DataFrame, *, asof: pd.Timestamp,
                   season: int, second: int, score: int, line: float,
                   market_median: float, recent_total: float) -> list[dict]:
    out = []
    for model in ("replica", "team", "pooled", "playoff"):
        sample = sample_for_state(games, states, asof, second, season, model in ("replica", "team"))
        coefficient = 0.0 if model == "replica" else fit_form_coefficient(games, asof, model == "playoff")
        out.append(predict_from_sample(
            sample, asof=asof, second=second, score=score, line=line,
            market_median=market_median, recent_total=recent_total,
            coefficient=coefficient, model=model,
        ))
    return out


def audit_check() -> dict:
    table = pd.read_csv(CURRENT_TABLE)
    with sqlite3.connect(f"file:{AUDIT}?mode=ro", uri=True) as connection:
        payloads = (row[0] for row in connection.execute("SELECT payload_json FROM fills"))
        contexts = [json.loads(payload).get("submission_context") for payload in payloads]
    contexts = [context for context in contexts if isinstance(context, dict)]
    raw_errors, adjusted_errors = [], []
    for context in contexts:
        second = int(context["elapsed_second"])
        differential = float(context["differential"])
        raw = float(table.loc[second, str(differential)])
        raw_errors.append(abs(raw - context["raw_theo"]))
        # The archived table has one-point jumps. Linear shifts reproduce most
        # adjusted values; some cannot be recovered without the original code.
        values = table.iloc[second, 1::2].to_numpy(dtype=float)
        grid = np.arange(-0.5, -0.5 + len(values), 1.0)
        adjusted = float(np.interp(differential - context["remaining_adjustment"], grid, values))
        adjusted_errors.append(abs(adjusted - context["adjusted_theo"]))
    return {
        "context_fills": len(contexts),
        "raw_exact_count": sum(error < 1e-10 for error in raw_errors),
        "raw_max_abs_error": max(raw_errors),
        "adjusted_exact_count": sum(error < 1e-10 for error in adjusted_errors),
        "adjusted_mean_abs_error": float(np.mean(adjusted_errors)),
        "adjusted_max_abs_error": max(adjusted_errors),
    }


def backtest() -> pd.DataFrame:
    games = load_games()
    games = games.merge(form_features(games), on="game_id", how="left")
    seconds = [0, 600, 1200, 1800, 2100]
    states = load_states(seconds)
    merged = states.merge(games, on="game_id", how="inner")
    targets = merged.loc[
        merged.season_end.isin([2024, 2025, 2026])
        & merged.ou_line.notna()
        & (merged.ou_line_status == "explicit_close")
        & np.isclose(merged.ou_line.mod(1), 0.5)
    ].copy()
    records = []
    for day, day_rows in targets.groupby("game_date", sort=True):
        prior_games = games.loc[games.game_date < day]
        season = int(day_rows.season_end.iloc[0])
        if sum(prior_games.season_end == season) < 80:
            continue
        recent_samples = {}
        full_samples = {}
        coefficients = {
            "team": fit_form_coefficient(games, day, False),
            "playoff": fit_form_coefficient(games, day, True),
        }
        coefficients["pooled"] = coefficients["team"]
        for second in seconds:
            at_second = merged.loc[merged.second == second]
            historical = at_second.loc[at_second.game_date < day]
            full_samples[second] = historical
            recent_samples[second] = historical.loc[historical.season_end == season]
        for row in day_rows.itertuples():
            label = int(row.total_points > row.ou_line)
            for model in ("replica", "team", "pooled", "playoff"):
                sample = recent_samples[row.second] if model in ("replica", "team") else full_samples[row.second]
                if len(sample) < 30:
                    continue
                result = predict_from_sample(
                    sample, asof=day, second=int(row.second), score=int(row.score),
                    line=float(row.ou_line), market_median=float(row.ou_line),
                    recent_total=float(row.recent_total),
                    coefficient=coefficients.get(model, 0.0), model=model,
                )
                records.append({
                    "game_id": row.game_id, "game_date": day.date().isoformat(),
                    "season_end": season, "season_type": row.season_type,
                    "second": int(row.second), "model": model,
                    "probability": result["probability_over"], "actual_over": label,
                    "form_coefficient": result["form_coefficient"],
                })
    return pd.DataFrame.from_records(records)


def summarize(backtest_rows: pd.DataFrame) -> pd.DataFrame:
    rows = backtest_rows.copy()
    rows["brier"] = (rows.probability - rows.actual_over) ** 2
    clipped = rows.probability.clip(1e-6, 1 - 1e-6)
    rows["log_loss"] = -(rows.actual_over * np.log(clipped) + (1 - rows.actual_over) * np.log(1 - clipped))
    rows["segment"] = np.where(rows.season_type == "postseason", "playoff", "all")
    all_rows = rows.copy()
    all_rows["segment"] = "all"
    rows = pd.concat([all_rows, rows.loc[rows.segment == "playoff"]], ignore_index=True)
    summary = rows.groupby(["segment", "second", "model"], as_index=False).agg(
        games=("game_id", "nunique"), brier=("brier", "mean"),
        log_loss=("log_loss", "mean"), mean_probability=("probability", "mean"),
        over_rate=("actual_over", "mean"),
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit-check")
    sub.add_parser("backtest")
    predict = sub.add_parser("predict")
    predict.add_argument("--date", required=True, help="Game date, YYYY-MM-DD")
    predict.add_argument("--season", type=int, required=True)
    predict.add_argument("--second", type=int, required=True, help="Elapsed regulation second")
    predict.add_argument("--score", type=int, required=True, help="Combined current score")
    predict.add_argument("--line", type=float, required=True)
    predict.add_argument("--market-median", type=float, required=True, help="Pregame 50/50 total")
    predict.add_argument("--home", required=True, help="Home team name from games.csv")
    predict.add_argument("--away", required=True, help="Away team name from games.csv")
    args = parser.parse_args()
    if args.command == "audit-check":
        print(json.dumps(audit_check(), indent=2))
    elif args.command == "backtest":
        print(summarize(backtest()).to_csv(index=False), end="")
    else:
        games = load_games()
        games = games.merge(form_features(games), on="game_id", how="left")
        asof = pd.Timestamp(args.date)
        known_teams = set(games.home_team) | set(games.away_team)
        if args.home not in known_teams or args.away not in known_teams:
            parser.error("--home and --away must match team names in games.csv")
        recent_total = recent_total_for_match(games, asof, args.home, args.away)
        states = load_states([args.second])
        result = prediction_set(
            games, states, asof=asof, season=args.season, second=args.second,
            score=args.score, line=args.line, market_median=args.market_median,
            recent_total=recent_total,
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
