"""Train and use a WNBA totals model from game history, without trade-audit data."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import defaultdict, deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss


ROOT = Path(__file__).resolve().parent
GAMES_PATH = ROOT / "data/processed/wnba_history/games.csv"
SECONDS_DIR = ROOT / "data/processed/wnba_history/seconds"
RATINGS_PATH = ROOT / "data/processed/wnba_history/latest_team_ratings.csv"
MODEL_PATH = ROOT / "models/wnba_independent.joblib"
REPORT_PATH = ROOT / "models/wnba_independent_report.json"
REGULATION_SECONDS = 2400
SNAPSHOT_SECONDS = list(range(0, 2251, 150))
EVAL_SECONDS = (0, 600, 1200, 1800, 2100)
MAX_SUPPORTED_SECOND = max(SNAPSHOT_SECONDS)
MODEL_SCHEMA_VERSION = 2
STRIKE_OFFSETS = np.array([-24, -16, -8, -4, 0, 4, 8, 16, 24], dtype=np.float32)

FEATURES = [
    "second", "remaining_seconds", "period", "home_score", "away_score",
    "score_total", "score_gap", "points_last_300", "home_last_300",
    "away_last_300", "points_period", "pace_to_now", "pace_last_300",
    "anchor_line", "points_needed", "season_end", "is_postseason",
    "neutral_site", "league_recent_50", "season_games_prior",
    "home_total_5", "home_scored_5", "home_allowed_5", "home_games_prior",
    "home_rest_days", "away_total_5", "away_scored_5", "away_allowed_5",
    "away_games_prior", "away_rest_days", "home_off_rating_20",
    "home_def_rating_20", "away_off_rating_20", "away_def_rating_20",
    "home_elo_pre", "away_elo_pre", "home_elo_expected", "away_elo_expected",
]

CENTERED_FEATURES = [
    "remaining_seconds", "score_gap", "points_needed_centered", "scoring_deviation",
    "recent_scoring_deviation", "is_postseason", "neutral_site", "league_gap",
    "home_form_gap", "away_form_gap", "home_scoring_balance", "away_scoring_balance",
    "home_games_prior", "away_games_prior", "home_rest_days", "away_rest_days",
    "rating_total", "rating_gap", "elo_gap",
]


def closing_probabilities(games: pd.DataFrame) -> dict[str, float]:
    """Match the declared closing provider/line; normalize the two implied odds."""
    result = {}
    for game in games.loc[games.ou_line_status == "explicit_close"].itertuples():
        path = ROOT / "data/raw/wnba_history/odds" / f"{game.game_id}.json.gz"
        if not path.exists():
            continue
        with gzip.open(path, "rt") as stream:
            items = json.load(stream).get("items", [])
        for item in items:
            if item.get("provider", {}).get("name") != game.ou_provider:
                continue
            close = item.get("close", {})
            try:
                line = float(close["total"]["american"])
                over, under = (float(close[side]["american"]) for side in ("over", "under"))
                if not math.isclose(line, game.ou_line) or not all(math.isfinite(v) and v != 0 for v in (over, under)):
                    continue
                implied = lambda odds: 100 / (100 + odds) if odds > 0 else -odds / (100 - odds)
                result[str(game.game_id)] = implied(over) / (implied(over) + implied(under))
                break
            except (KeyError, TypeError, ValueError):
                continue
    return result


def load_games() -> pd.DataFrame:
    games = pd.read_csv(GAMES_PATH, dtype={"game_id": str, "home_team_id": str, "away_team_id": str})
    games["game_date"] = pd.to_datetime(games.game_date)
    games = games.loc[games.total_points.notna()].copy()
    if games.game_id.duplicated().any():
        raise ValueError("Duplicate game IDs in history")
    probabilities = closing_probabilities(games)
    games["anchor_probability"] = games.game_id.map(probabilities).fillna(0.5)
    games["anchor_probability_source"] = np.where(games.game_id.isin(probabilities), "closing_odds", "assumed_half")
    return games.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def pregame_features(games: pd.DataFrame) -> pd.DataFrame:
    """Compute team and league form before each date, never from that day's games."""
    team_games = defaultdict(lambda: deque(maxlen=10))
    last_date = {}
    league_games = deque(maxlen=50)
    season_counts = defaultdict(int)
    records = []
    for day, day_games in games.groupby("game_date", sort=True):
        for game in day_games.itertuples():
            record = {
                "game_id": game.game_id,
                "league_recent_50": np.mean(league_games) if league_games else np.nan,
                "season_games_prior": season_counts[int(game.season_end)],
            }
            for side in ("home", "away"):
                team_id = getattr(game, f"{side}_team_id")
                recent = list(team_games[team_id])[-5:]
                record[f"{side}_games_prior"] = len(recent)
                record[f"{side}_rest_days"] = (
                    (day - last_date[team_id]).days if team_id in last_date else np.nan
                )
                for key, index in (("total", 0), ("scored", 1), ("allowed", 2)):
                    record[f"{side}_{key}_5"] = (
                        float(np.mean([entry[index] for entry in recent])) if recent else np.nan
                    )
            records.append(record)
        for game in day_games.itertuples():
            if pd.isna(game.total_points):
                continue
            total = float(game.total_points)
            league_games.append(total)
            season_counts[int(game.season_end)] += 1
            team_games[game.home_team_id].append((total, float(game.home_score), float(game.away_score)))
            team_games[game.away_team_id].append((total, float(game.away_score), float(game.home_score)))
            last_date[game.home_team_id] = day
            last_date[game.away_team_id] = day
    return pd.DataFrame(records)


def load_states(target_seconds: list[int] | None = None) -> pd.DataFrame:
    target_seconds = SNAPSHOT_SECONDS if target_seconds is None else target_seconds
    requested = sorted(set(target_seconds) | {max(0, t - 300) for t in target_seconds}
                       | {(t // 600) * 600 for t in target_seconds})
    frames = []
    for path in sorted(SECONDS_DIR.glob("season_end=*/part-0.parquet")):
        frame = pd.read_parquet(
            path,
            columns=["game_id", "second", "home_score", "away_score"],
            filters=[("second", "in", requested)],
        )
        frames.append(frame)
    states = pd.concat(frames, ignore_index=True)
    states["game_id"] = states.game_id.astype(str)
    previous = states[["game_id", "second", "home_score", "away_score"]].copy()
    previous["second"] += 300
    previous = previous.rename(columns={"home_score": "home_prev_300", "away_score": "away_prev_300"})
    states = states.merge(previous, on=["game_id", "second"], how="left", validate="one_to_one")
    states.loc[states.second < 300, ["home_prev_300", "away_prev_300"]] = 0
    starts = states[["game_id", "second", "home_score", "away_score"]].copy()
    starts = starts.rename(columns={
        "second": "period_start", "home_score": "home_period_start",
        "away_score": "away_period_start",
    })
    states["period_start"] = (states.second // 600) * 600
    states = states.merge(starts, on=["game_id", "period_start"], how="left", validate="many_to_one")
    return states.loc[states.second.isin(target_seconds)].copy()


def centered_features(rows: pd.DataFrame) -> pd.DataFrame:
    result = rows.copy()
    elapsed = result.second / REGULATION_SECONDS
    result["points_needed_centered"] = result.points_needed - result.anchor_line * (1 - elapsed)
    result["scoring_deviation"] = result.score_total - result.anchor_line * elapsed
    result["recent_scoring_deviation"] = result.points_last_300 - result.anchor_line * result.second.clip(upper=300) / REGULATION_SECONDS
    result["league_gap"] = result.league_recent_50 - result.anchor_line
    for side in ("home", "away"):
        result[f"{side}_form_gap"] = result[f"{side}_total_5"] - result.anchor_line
        result[f"{side}_scoring_balance"] = result[f"{side}_scored_5"] - result[f"{side}_allowed_5"]
    result["rating_total"] = (result.home_off_rating_20 + result.home_def_rating_20
                              + result.away_off_rating_20 + result.away_def_rating_20) / 2
    result["rating_gap"] = (result.home_off_rating_20 - result.home_def_rating_20
                            - result.away_off_rating_20 + result.away_def_rating_20)
    result["elo_gap"] = result.home_elo_pre - result.away_elo_pre
    return result


def state_features(rows: pd.DataFrame) -> pd.DataFrame:
    result = rows.copy()
    result["remaining_seconds"] = REGULATION_SECONDS - result.second
    result["period"] = result.second // 600 + 1
    result["score_total"] = result.home_score + result.away_score
    result["score_gap"] = (result.home_score - result.away_score).abs()
    result["home_last_300"] = result.home_score - result.home_prev_300
    result["away_last_300"] = result.away_score - result.away_prev_300
    result["points_last_300"] = result.home_last_300 + result.away_last_300
    result["points_period"] = (
        result.score_total - result.home_period_start - result.away_period_start
    )
    result["pace_to_now"] = result.score_total * REGULATION_SECONDS / result.second.clip(lower=1)
    result["pace_last_300"] = result.points_last_300 * REGULATION_SECONDS / result.second.clip(upper=300, lower=1)
    result.loc[result.second == 0, ["pace_to_now", "pace_last_300"]] = np.nan
    result["anchor_line"] = result.ou_line
    result["points_needed"] = result.anchor_line - result.score_total
    result["is_postseason"] = (result.season_type == "postseason").astype(int)
    result["neutral_site"] = result.neutral_site.astype(int)
    return centered_features(result)


def build_history(target_seconds: list[int] | None = None) -> pd.DataFrame:
    games = load_games()
    form = pregame_features(games)
    states = load_states(target_seconds)
    history = states.merge(
        games.drop(columns=["home_score", "away_score"]),
        on="game_id", how="inner", validate="many_to_one",
    )
    history = history.merge(form, on="game_id", how="left", validate="many_to_one")
    history = history.loc[
        history.ou_line.notna()
        & history.timeline_available
        & history.ou_line_status.isin(["explicit_close", "current_snapshot"])
        & (history.season_end >= 2018)
    ].copy()
    return state_features(history)


def as_matrix(rows: pd.DataFrame, offsets: np.ndarray | None = None, features: list[str] | None = None):
    features = FEATURES if features is None else features
    matrix = rows[features].to_numpy(dtype=np.float32, copy=True)
    if offsets is None:
        return matrix
    count = len(offsets)
    expanded = np.repeat(matrix, count, axis=0)
    half_lines = np.floor(rows.anchor_line.to_numpy(dtype=np.float32)) + 0.5
    lines = np.repeat(half_lines, count) + np.tile(offsets, len(rows))
    needed = lines - np.repeat(rows.score_total.to_numpy(dtype=np.float32), count)
    if "points_needed" in features:
        expanded[:, features.index("points_needed")] = needed
    if "points_needed_centered" in features:
        expected = rows.anchor_line * (REGULATION_SECONDS - rows.second) / REGULATION_SECONDS
        expanded[:, features.index("points_needed_centered")] = needed - np.repeat(expected.to_numpy(), count)
    finals = np.repeat(rows.total_points.to_numpy(dtype=np.float32), count)
    labels = (finals > lines).astype(np.int8)
    return expanded, labels


def row_weights(rows: pd.DataFrame, cutoff: pd.Timestamp, half_life_years: float,
                playoff_weight: float) -> np.ndarray:
    age_years = (cutoff - rows.game_date).dt.days.to_numpy() / 365.25
    if np.any(age_years <= 0):
        raise ValueError("Training includes a game on or after its cutoff")
    weights = np.exp2(-age_years / half_life_years)
    weights *= np.where(rows.ou_line_status.to_numpy() == "explicit_close", 1.0, 0.6)
    weights *= np.where(rows.season_type.to_numpy() == "postseason", playoff_weight, 1.0)
    return weights.astype(np.float32)


def fit_model(rows: pd.DataFrame, cutoff: pd.Timestamp, settings: dict):
    features = CENTERED_FEATURES if settings.get("features") == "centered" else FEATURES
    train_x, train_y = as_matrix(rows, STRIKE_OFFSETS, features)
    weights = np.repeat(
        row_weights(rows, cutoff, settings["half_life_years"], settings["playoff_weight"])
        * len(SNAPSHOT_SECONDS) / rows.groupby("game_id").game_id.transform("size").to_numpy(),
        len(STRIKE_OFFSETS),
    )
    monotonic = [0] * len(features)
    needed_feature = "points_needed_centered" if "points_needed_centered" in features else "points_needed"
    monotonic[features.index(needed_feature)] = -1
    model = HistGradientBoostingClassifier(
        learning_rate=0.06, max_iter=settings["max_iter"],
        max_leaf_nodes=settings["max_leaf_nodes"], min_samples_leaf=settings.get("min_samples_leaf", 140),
        l2_regularization=8.0, monotonic_cst=monotonic,
        early_stopping=False, random_state=17,
    )
    model.fit(train_x, train_y, sample_weight=weights)
    model.wnba_features_ = features
    return model


def tip_matrix(rows: pd.DataFrame, features: list[str] | None = None) -> np.ndarray:
    tip = as_matrix(rows)
    for name in ("second", "home_score", "away_score", "score_total", "score_gap",
                 "points_last_300", "home_last_300", "away_last_300", "points_period"):
        tip[:, FEATURES.index(name)] = 0
    tip[:, FEATURES.index("remaining_seconds")] = REGULATION_SECONDS
    tip[:, FEATURES.index("period")] = 1
    tip[:, FEATURES.index("pace_to_now")] = np.nan
    tip[:, FEATURES.index("pace_last_300")] = np.nan
    tip[:, FEATURES.index("points_needed")] = tip[:, FEATURES.index("anchor_line")]
    frame = centered_features(pd.DataFrame(tip, columns=FEATURES))
    return frame[FEATURES if features is None else features].to_numpy(dtype=np.float32)


def anchored_probabilities(model, rows: pd.DataFrame,
                           target_matrix: np.ndarray | None = None) -> np.ndarray:
    if target_matrix is None:
        target_matrix = as_matrix(rows, features=getattr(model, "wnba_features_", FEATURES))
    features = getattr(model, "wnba_features_", FEATURES)
    if len(rows) == 0 or len(target_matrix) % len(rows) or target_matrix.shape[1] != len(features):
        raise ValueError("Prediction matrix is not aligned to its game states")
    needed_feature = "points_needed_centered" if "points_needed_centered" in features else "points_needed"
    needed_column = features.index(needed_feature)
    repetitions = len(target_matrix) // len(rows)
    anchor_needed = np.repeat(
        (rows.anchor_line - rows.score_total).to_numpy(dtype=np.float32), repetitions,
    )
    if needed_feature == "points_needed_centered":
        anchor_needed -= np.repeat((rows.anchor_line * (REGULATION_SECONDS - rows.second) / REGULATION_SECONDS).to_numpy(dtype=np.float32), repetitions)
    offset = target_matrix[:, needed_column] - anchor_needed
    left_offset = np.floor(offset / 4) * 4
    left = target_matrix.copy()
    right = target_matrix.copy()
    left[:, needed_column] = anchor_needed + left_offset
    right[:, needed_column] = left[:, needed_column] + 4
    left_p = model.predict_proba(left)[:, 1]
    right_p = model.predict_proba(right)[:, 1]
    raw = left_p + (right_p - left_p) * (offset - left_offset) / 4
    tip = model.predict_proba(tip_matrix(rows, features))[:, 1]
    tip = np.repeat(tip, repetitions)
    remaining_fraction = np.repeat(
        (REGULATION_SECONDS - rows.second.to_numpy()) / REGULATION_SECONDS,
        repetitions,
    )
    market_p = rows.get("anchor_probability", pd.Series(0.5, index=rows.index)).to_numpy(dtype=float)
    if not np.all(np.isfinite(market_p) & (market_p > 0) & (market_p < 1)):
        raise ValueError("Anchor probabilities must be finite and strictly between 0 and 1")
    probability = expit(logit(np.clip(raw, 1e-6, 1 - 1e-6))
                        + remaining_fraction * (logit(np.repeat(market_p, repetitions))
                                                - logit(np.clip(tip, 1e-6, 1 - 1e-6))))
    # The threshold is already crossed, assuming no subsequent score correction.
    target_needed = offset + np.repeat((rows.anchor_line - rows.score_total).to_numpy(), repetitions)
    return np.where(target_needed < 0, 1.0, probability)


def score_model(model, rows: pd.DataFrame) -> dict:
    rows = rows.loc[rows.second.isin(EVAL_SECONDS)].copy()
    features = getattr(model, "wnba_features_", FEATURES)
    raw_central = model.predict_proba(as_matrix(rows, features=features))[:, 1]
    central = anchored_probabilities(model, rows)
    labels = (rows.total_points.to_numpy() > rows.anchor_line.to_numpy()).astype(int)
    result = {
        "games": int(rows.game_id.nunique()),
        "states": len(rows),
        "raw_brier": float(brier_score_loss(labels, raw_central)),
        "brier": float(brier_score_loss(labels, central)),
        "log_loss": float(log_loss(labels, np.clip(central, 1e-6, 1 - 1e-6))),
    }
    if len(rows.loc[rows.season_type == "postseason"]):
        playoff = rows.season_type.to_numpy() == "postseason"
        result["playoff_games"] = int(rows.loc[playoff, "game_id"].nunique())
        result["playoff_brier"] = float(brier_score_loss(labels[playoff], central[playoff]))
    all_x, all_y = as_matrix(rows, STRIKE_OFFSETS, features)
    all_prob = anchored_probabilities(model, rows, all_x)
    result["nine_strike_brier"] = float(brier_score_loss(all_y, all_prob))
    return result


def train() -> dict:
    from model_review import review_and_train
    return review_and_train()


def future_row(args) -> pd.DataFrame:
    games = load_games()
    date = pd.Timestamp(args.date)
    last_date = games.game_date.max()
    if date <= last_date:
        raise ValueError("Manual predictions require a date after the uploaded games")
    registry = {}
    for game in games.itertuples():
        registry[game.home_team] = game.home_team_id
        registry[game.away_team] = game.away_team_id
    if args.home not in registry or args.away not in registry:
        raise ValueError("Team names must match games.csv")
    if args.home == args.away:
        raise ValueError("Home and away teams must differ")
    ratings = pd.read_csv(RATINGS_PATH).set_index("team")
    if args.home not in ratings.index or args.away not in ratings.index:
        raise ValueError("Current team ratings are missing")
    row = {
        "game_id": "future-input", "game_date": date, "season_end": args.season,
        "season_type": "postseason" if args.playoff else "regular",
        "home_team": args.home, "away_team": args.away,
        "home_team_id": registry[args.home], "away_team_id": registry[args.away],
        "total_points": np.nan, "ou_line": args.anchor_line,
        "neutral_site": args.neutral_site,
        "second": args.second, "home_score": args.home_score,
        "away_score": args.away_score,
        "home_prev_300": args.home_prev_300,
        "away_prev_300": args.away_prev_300,
        "home_period_start": args.home_period_start,
        "away_period_start": args.away_period_start,
    }
    for side, team in (("home", args.home), ("away", args.away)):
        rating = ratings.loc[team]
        if pd.Timestamp(rating.as_of_date) >= date or int(rating.season_end) != args.season:
            raise ValueError("Team ratings must be from an earlier date in the requested season")
        for name in ("off_rating_20", "def_rating_20"):
            row[f"{side}_{name}"] = rating[name]
        row[f"{side}_elo_pre"] = rating.elo
    home_elo = row["home_elo_pre"]
    away_elo = row["away_elo_pre"]
    home_advantage = 0 if args.neutral_site else 65
    row["home_elo_expected"] = 1 / (1 + 10 ** ((away_elo - home_elo - home_advantage) / 400))
    row["away_elo_expected"] = 1 - row["home_elo_expected"]
    all_games = pd.concat([games, pd.DataFrame([row])], ignore_index=True)
    form = pregame_features(all_games)
    row.update(form.loc[form.game_id == "future-input"].iloc[0].to_dict())
    return state_features(pd.DataFrame([row]))


def predict(args) -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Run 'train' before 'predict'")
    if args.second < 0 or args.second > MAX_SUPPORTED_SECOND:
        raise ValueError(f"Elapsed second must be 0..{MAX_SUPPORTED_SECOND}; later states were not trained")
    if not math.isfinite(args.anchor_line) or args.anchor_line <= 0 or not math.isfinite(args.line):
        raise ValueError("Anchor and target totals must be finite; anchor must be positive")
    anchor_probability = getattr(args, "anchor_probability", 0.5)
    if not math.isfinite(anchor_probability) or not 0 < anchor_probability < 1:
        raise ValueError("Anchor probability must be strictly between zero and one")
    if args.season != pd.Timestamp(args.date).year:
        raise ValueError("Season must match the game-date year")
    if min(args.home_score, args.away_score) < 0:
        raise ValueError("Scores must be nonnegative")
    if args.line % 1 != 0.5:
        raise ValueError("Target total must end in .5")
    if abs(args.line - args.anchor_line) > 24:
        raise ValueError("Target line must be within 24 points of the anchor")
    if args.second < 300:
        args.home_prev_300 = 0
        args.away_prev_300 = 0
    elif args.home_prev_300 is None or args.away_prev_300 is None:
        raise ValueError("Supply both scores from 300 seconds earlier")
    if args.home_prev_300 < 0 or args.away_prev_300 < 0:
        raise ValueError("Earlier scores must be nonnegative")
    if args.second % 600 == 0:
        args.home_period_start = args.home_score
        args.away_period_start = args.away_score
    elif args.second < 600:
        args.home_period_start = 0
        args.away_period_start = 0
    elif args.home_period_start is None or args.away_period_start is None:
        raise ValueError("Supply both scores at the start of the current period")
    if args.home_period_start < 0 or args.away_period_start < 0:
        raise ValueError("Period-start scores must be nonnegative")
    bundle = joblib.load(MODEL_PATH)
    if bundle.get("schema_version") != MODEL_SCHEMA_VERSION or bundle["features"] != getattr(bundle["model"], "wnba_features_", None):
        raise ValueError("Model schema is incompatible; retrain with the current code")
    for path in (GAMES_PATH, RATINGS_PATH):
        expected = bundle["report"]["source_hashes"][str(path.relative_to(ROOT))]
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Game or ratings inputs changed since training; retrain before prediction")
    row = future_row(args)
    row["points_needed"] = args.line - row.score_total
    row["anchor_probability"] = anchor_probability
    row = centered_features(row)
    probability = float(anchored_probabilities(bundle["model"], row)[0])
    warnings = ["Research model: no prospective validation or live-execution evaluation.",
                "History stops at the reported date; input hashes cannot detect games not uploaded."]
    if args.second not in SNAPSHOT_SECONDS:
        warnings.append("Between-snapshot prediction: off-grid validation underperformed the empirical benchmark.")
    return {
        "research_only": True,
        "warnings": warnings,
        "probability_over": probability,
        "probability_under": 1 - probability,
        "target_line": args.line,
        "anchor_line": args.anchor_line,
        "anchor_probability": anchor_probability,
        "game_date": args.date,
        "history_latest_game": bundle["report"]["history_latest_game"],
        "model_training_through": bundle["report"]["training_through"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("train")
    live = commands.add_parser("predict")
    live.add_argument("--date", required=True)
    live.add_argument("--season", type=int, required=True)
    live.add_argument("--home", required=True)
    live.add_argument("--away", required=True)
    live.add_argument("--second", type=int, required=True)
    live.add_argument("--home-score", type=int, required=True)
    live.add_argument("--away-score", type=int, required=True)
    live.add_argument("--home-prev-300", type=int)
    live.add_argument("--away-prev-300", type=int)
    live.add_argument("--home-period-start", type=int)
    live.add_argument("--away-period-start", type=int)
    live.add_argument("--anchor-line", type=float, required=True)
    live.add_argument("--anchor-probability", type=float, default=0.5,
                      help="Fair over probability at the anchor; default is a true 50/50 midpoint")
    live.add_argument("--line", type=float, required=True)
    live.add_argument("--playoff", action="store_true")
    live.add_argument("--neutral-site", action="store_true")
    args = parser.parse_args()
    output = train() if args.command == "train" else predict(args)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
