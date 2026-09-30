"""Limited chronological model review. 2026 is previously inspected, not a fresh test."""

from __future__ import annotations

import gzip
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.special import expit, logit

import independent_wnba_model as m


LIVE_SECONDS = [300, 600, 900, 1200, 1500, 1800, 2100, 2250]
OFF_GRID_SECONDS = [375, 825, 1275, 1725, 2175]
CANDIDATES = [
    dict(name="full", features="full", half_life_years=3.0, playoff_weight=1.0,
         max_iter=180, max_leaf_nodes=15, min_samples_leaf=140),
    dict(name="compact", features="centered", half_life_years=3.0, playoff_weight=1.0,
         max_iter=120, max_leaf_nodes=7, min_samples_leaf=500),
    dict(name="compact_playoff", features="centered", half_life_years=3.0, playoff_weight=2.5,
         max_iter=120, max_leaf_nodes=7, min_samples_leaf=500),
    dict(name="compact_recent", features="centered", half_life_years=1.5, playoff_weight=1.0,
         max_iter=120, max_leaf_nodes=7, min_samples_leaf=500),
]
PROTOCOL = {
    "selection_years": [2024, 2025],
    "primary": "equal-game mean Brier over 8 live snapshots at the listed closing line",
    "selection_gate": "improve both years and 9-strike Brier, with paired game-bootstrap upper 95% bound below zero",
    "weight_selection": "extra weighting also must pass the same gate against compact; otherwise use compact",
    "2026_status": "previously inspected exploratory replay; never used in this pass's selection",
    "candidate_settings": CANDIDATES,
    "live_seconds": LIVE_SECONDS,
    "bootstrap_seed": 91030,
    "bootstrap_replicates": 3000,
}


def evaluate(model, rows: pd.DataFrame, candidate: str, fold: str) -> pd.DataFrame:
    features = model.wnba_features_
    probability = m.anchored_probabilities(model, rows)
    matrix, labels = m.as_matrix(rows, m.STRIKE_OFFSETS, features)
    alternative = m.anchored_probabilities(model, rows, matrix)
    alt_loss = ((alternative - labels) ** 2).reshape(len(rows), -1).mean(axis=1)
    out = rows[["game_id", "game_date", "season_end", "season_type", "second"]].copy()
    out["candidate"], out["fold"] = candidate, fold
    out["probability"] = probability
    out["actual_over"] = (rows.total_points.to_numpy() > rows.anchor_line.to_numpy()).astype(int)
    out["brier"] = (probability - out.actual_over) ** 2
    p = np.clip(probability, 1e-6, 1 - 1e-6)
    out["log_loss"] = -(out.actual_over * np.log(p) + (1 - out.actual_over) * np.log(1 - p))
    out["nine_strike_brier"] = alt_loss
    return out


def metrics(predictions: pd.DataFrame) -> dict:
    per_game = predictions.groupby("game_id")[["brier", "log_loss", "nine_strike_brier"]].mean()
    result = {"games": len(per_game), "states": len(predictions),
              **{key: float(value) if pd.notna(value) else None for key, value in per_game.mean().items()}}
    playoff = predictions.loc[predictions.season_type == "postseason"]
    if len(playoff):
        result["playoff_games"] = int(playoff.game_id.nunique())
        result["playoff_brier"] = float(playoff.groupby("game_id").brier.mean().mean())
    result["by_second"] = {str(t): float(loss) for t, loss in predictions.groupby("second").brier.mean().items()}
    calibrated = predictions.assign(bin=np.minimum((predictions.probability * 10).astype(int), 9))
    result["calibration"] = calibrated.groupby("bin").agg(
        states=("actual_over", "size"), probability=("probability", "mean"), frequency=("actual_over", "mean")
    ).reset_index().to_dict("records")
    return result


def paired_interval(left: pd.DataFrame, right: pd.DataFrame) -> dict:
    # All states and synthetic strikes from a game stay together.
    paired = pd.concat([
        left.groupby("game_id").brier.mean().rename("left"),
        right.groupby("game_id").brier.mean().rename("right"),
    ], axis=1)
    if paired.isna().any().any():
        raise ValueError("Comparisons must use exactly the same games")
    difference = (paired.left - paired.right).to_numpy()
    rng = np.random.default_rng(PROTOCOL["bootstrap_seed"])
    boot = difference[rng.integers(0, len(difference), (PROTOCOL["bootstrap_replicates"], len(difference)))].mean(axis=1)
    return {"games": len(difference), "mean_brier_difference": float(difference.mean()),
            "ci95": np.quantile(boot, [0.025, 0.975]).tolist()}


def eligible(rows: pd.DataFrame, seconds: list[int]) -> pd.DataFrame:
    return rows.loc[rows.second.isin(seconds) & (rows.ou_line_status == "explicit_close")
                    & np.isclose(rows.ou_line.mod(1), 0.5)].copy()


def empirical_replay(past: pd.DataFrame, targets: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    if (past.game_date >= cutoff).any() or (targets.game_date < cutoff).any():
        raise ValueError("Empirical benchmark violates the shared cutoff")
    out = targets[["game_id", "game_date", "season_end", "season_type", "second"]].copy()
    probabilities = []
    for target in targets.itertuples():
        sample = past.loc[past.second == target.second]
        this_year = sample.loc[sample.season_end == target.season_end]
        if len(this_year) >= 80:
            sample, weights = this_year, np.ones(len(this_year))
        else:
            weights = np.exp2(-(cutoff - sample.game_date).dt.days.to_numpy() / (3 * 365.25))
        final = sample.total_points.to_numpy()
        order = np.argsort(final)
        median = final[order[np.searchsorted(np.cumsum(weights[order]), weights.sum() / 2)]]
        remaining = final - sample.score_total.to_numpy()
        decay = (2400 - target.second) / 2400
        shift = (target.anchor_line - median) * decay
        raw = np.average(np.clip(remaining - target.points_needed + shift + 0.5, 0, 1), weights=weights)
        tip = np.average(np.clip(final - median + 0.5, 0, 1), weights=weights)
        probability = expit(logit(np.clip(raw, 1e-6, 1 - 1e-6))
                            + decay * (logit(target.anchor_probability) - logit(np.clip(tip, 1e-6, 1 - 1e-6))))
        probabilities.append(1.0 if target.points_needed < 0 else probability)
    out["probability"] = probabilities
    out["actual_over"] = (targets.total_points.to_numpy() > targets.anchor_line.to_numpy()).astype(int)
    out["brier"] = (out.probability - out.actual_over) ** 2
    p = out.probability.clip(1e-6, 1 - 1e-6)
    out["log_loss"] = -(out.actual_over * np.log(p) + (1 - out.actual_over) * np.log(1 - p))
    out["nine_strike_brier"] = np.nan
    out["candidate"], out["fold"] = "empirical", str(cutoff.date())
    return out


def data_checks() -> dict:
    games = m.load_games()
    ratings = pd.read_csv(m.ROOT / "data/processed/wnba_history/team_game_ratings.csv",
                          dtype={"game_id": str, "team_id": str})
    rating_errors = []
    for _, team in ratings.sort_values(["game_date", "game_id"]).groupby("team_id"):
        for game in team.itertuples():
            previous = team.loc[team.game_date < game.game_date].tail(20)
            if len(previous) == 20 and game.season_end >= 2018:
                for column, points in (("off_rating_20", "points_for"), ("def_rating_20", "points_against")):
                    rating_errors.append(abs(getattr(game, column) - 100 * previous[points].sum() / previous.possessions.sum()))
    ordered = ratings.sort_values(["team_id", "game_date", "game_id"])
    prior = ordered.groupby("team_id")[["season_end", "game_date", "elo_post"]].shift()
    continuous = (ordered.season_end == prior.season_end) & (ordered.game_date > prior.game_date)
    elo_error = float((ordered.loc[continuous, "elo_pre"] - prior.loc[continuous, "elo_post"]).abs().max())
    joined_errors = 0
    for side in ("home", "away"):
        joined = games.merge(ratings, left_on=["game_id", f"{side}_team_id"],
                             right_on=["game_id", "team_id"], validate="one_to_one", suffixes=("", "_rating"))
        for column in ("off_rating_20", "def_rating_20", "elo_pre", "elo_expected"):
            joined_errors += int((~np.isclose(joined[f"{side}_{column}"], joined[column], equal_nan=True)).sum())
    states = m.load_states()
    checks, mismatches = 0, []
    # Deterministic, season-stratified raw-play sample, independent of model loss.
    samples = games.loc[games.timeline_available].groupby("season_end", group_keys=False).apply(
        lambda frame: frame.iloc[np.linspace(0, len(frame) - 1, 6).astype(int)], include_groups=False
    )
    for game in samples.itertuples():
        path = m.ROOT / "data/raw/wnba_history/summaries" / f"{game.game_id}.json.gz"
        with gzip.open(path, "rt") as stream:
            source = json.load(stream)
        plays = []
        for play in source.get("plays", []):
            period = play.get("period", {}).get("number", 0)
            clock = play.get("clock", {}).get("displayValue", "")
            if not 1 <= period <= 4 or not clock:
                continue
            elapsed = period * 600 - clock_seconds(clock)
            plays.append((elapsed, int(play.get("sequenceNumber", 0)), play["homeScore"], play["awayScore"]))
        plays.sort()
        for state in states.loc[states.game_id == str(game.game_id)].itertuples():
            previous = [p for p in plays if p[0] <= state.second]
            expected = previous[-1][2:] if previous else (0, 0)
            checks += 1
            if expected != (state.home_score, state.away_score):
                mismatches.append({"game_id": str(game.game_id), "second": state.second,
                                   "raw": list(expected), "processed": [state.home_score, state.away_score]})
    return {
        "completed_games": len(games), "timeline_games": int(games.timeline_available.sum()),
        "missing_timelines_retained_for_form": int((~games.timeline_available).sum()),
        "score_sum_mismatches": int((games.total_points != games.home_score + games.away_score).sum()),
        "closing_odds_probability_games": int((games.anchor_probability_source == "closing_odds").sum()),
        "declared_close_games": int((games.ou_line_status == "explicit_close").sum()),
        "prior_rating_values_checked": len(rating_errors), "max_prior_rating_error": float(max(rating_errors)),
        "same_season_elo_transitions_checked": int(continuous.sum()), "max_elo_transition_error": elo_error,
        "joined_rating_mismatches": joined_errors,
        "non_prior_rating_end_dates": int((ratings.rating_end_date >= ratings.game_date).sum()),
        "raw_timeline_games_sampled": len(samples), "raw_timeline_states_checked": checks,
        "raw_timeline_mismatches": mismatches,
        "limitations": ["current_snapshot lines have no verified pregame capture time",
                        "retrospective play-by-play can contain corrections unavailable to a live feed",
                        "multiple events at one game-clock second share the final state at that clock"],
    }


def clock_seconds(clock: str) -> float:
    parts = clock.split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        return 60 * float(parts[0]) + float(parts[1])
    raise ValueError(f"Invalid game clock: {clock}")


def review_and_train() -> dict:
    m.MODEL_PATH.parent.mkdir(exist_ok=True)
    (m.MODEL_PATH.parent / "review_protocol.json").write_text(json.dumps(PROTOCOL, indent=2) + "\n")
    checks = data_checks()
    print(f"Data checks: {checks['raw_timeline_states_checked']} raw timeline states, "
          f"{len(checks['raw_timeline_mismatches'])} mismatches", flush=True)
    if (checks["raw_timeline_mismatches"] or checks["score_sum_mismatches"]
            or checks["joined_rating_mismatches"] or checks["non_prior_rating_end_dates"]
            or checks["max_prior_rating_error"] > 1e-8 or checks["max_elo_transition_error"] > 1e-8):
        raise ValueError("Data validation failed; inspect the uploaded history before training")
    history = m.build_history()
    predictions, fitted, fold_scores = {}, {}, {}
    for settings in CANDIDATES:
        parts, fold_scores[settings["name"]] = [], {}
        for year in PROTOCOL["selection_years"]:
            cutoff = pd.Timestamp(f"{year}-01-01")
            training = history.loc[history.game_date < cutoff]
            validation = eligible(history.loc[history.season_end == year], LIVE_SECONDS)
            model = m.fit_model(training, cutoff, settings)
            frame = evaluate(model, validation, settings["name"], str(year))
            parts.append(frame)
            fold_scores[settings["name"]][str(year)] = metrics(frame)
            if year == 2025:
                fitted[settings["name"]] = model
            print(f"{settings['name']} {year}: Brier {frame.brier.mean():.6f}", flush=True)
        predictions[settings["name"]] = pd.concat(parts, ignore_index=True)

    comparisons = {}
    def passes(candidate, control):
        interval = paired_interval(predictions[candidate], predictions[control])
        comparisons[f"{candidate}_minus_{control}"] = interval
        each_year = all(fold_scores[candidate][str(y)]["brier"] < fold_scores[control][str(y)]["brier"]
                        for y in PROTOCOL["selection_years"])
        alt_better = metrics(predictions[candidate])["nine_strike_brier"] < metrics(predictions[control])["nine_strike_brier"]
        return each_year and alt_better and interval["ci95"][1] < 0

    compact_ok = passes("compact", "full")
    weights_ok = {name: passes(name, "compact") for name in ("compact_playoff", "compact_recent")}
    selected_name = "full"
    if compact_ok:
        selected_name = "compact"
        for name in ("compact_playoff", "compact_recent"):
            if weights_ok[name] and metrics(predictions[name])["brier"] < metrics(predictions[selected_name])["brier"]:
                selected_name = name
    selected = next(item for item in CANDIDATES if item["name"] == selected_name)
    print(f"Selection frozen from 2024/25: {selected_name}", flush=True)
    (m.MODEL_PATH.parent / "review_selection.json").write_text(json.dumps({"settings": selected, "comparisons": comparisons}, indent=2) + "\n")
    pd.concat(list(predictions.values()), ignore_index=True).to_csv(m.MODEL_PATH.parent / "review_validation_predictions.csv", index=False)

    # Sensitivities below cannot change the selection.
    trusted = history.loc[(history.game_date < "2025-01-01") & (history.ou_line_status == "explicit_close")]
    trustworthy_model = m.fit_model(trusted, pd.Timestamp("2025-01-01"), selected)
    trusted_check = metrics(evaluate(trustworthy_model, eligible(history.loc[history.season_end == 2025], LIVE_SECONDS), "trusted_only", "2025"))
    off_grid = m.build_history(OFF_GRID_SECONDS)
    off_grid_targets = eligible(off_grid.loc[off_grid.season_end == 2025], OFF_GRID_SECONDS)
    off_grid_prediction = evaluate(fitted[selected_name], off_grid_targets, selected_name, "2025_off_grid")
    off_grid_check = metrics(off_grid_prediction)
    off_grid_benchmark = empirical_replay(off_grid.loc[off_grid.game_date < "2025-01-01"],
                                         off_grid_targets, pd.Timestamp("2025-01-01"))
    off_grid_check["empirical_same_cutoff"] = metrics(off_grid_benchmark)
    off_grid_check["paired"] = paired_interval(off_grid_prediction, off_grid_benchmark)
    replays, replay_predictions = [], []
    for start, end in (("2026-01-01", "2027-01-01"), ("2026-07-01", "2026-08-15"), ("2026-08-15", "2026-09-25")):
        cutoff = pd.Timestamp(start)
        past = history.loc[history.game_date < cutoff]
        targets = eligible(history.loc[(history.game_date >= cutoff) & (history.game_date < end)], LIVE_SECONDS)
        model = m.fit_model(past, cutoff, selected)
        pred = evaluate(model, targets, selected_name, start)
        benchmark = empirical_replay(past, targets, cutoff)
        replay_predictions.extend([pred, benchmark])
        replays.append({"train_before": start, "test_before": end, "selected": metrics(pred),
                        "empirical_same_cutoff": metrics(benchmark), "paired": paired_interval(pred, benchmark)})
        print(f"Replay {start}: model {pred.brier.mean():.6f}; empirical {benchmark.brier.mean():.6f}", flush=True)
    pd.concat(replay_predictions, ignore_index=True).to_csv(m.MODEL_PATH.parent / "review_replay_predictions.csv", index=False)
    final_model = m.fit_model(history, history.game_date.max() + pd.Timedelta(days=1), selected)
    source_paths = [m.GAMES_PATH, m.RATINGS_PATH, Path(m.__file__), Path(__file__)]
    source_paths += sorted(m.SECONDS_DIR.glob("season_end=*/part-0.parquet"))
    source_paths += [m.ROOT / "data/processed/wnba_history/team_game_ratings.csv"]
    sources = {str(p.relative_to(m.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    odds_manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted((m.ROOT / "data/raw/wnba_history/odds").glob("*.json.gz"))}
    report = {
        "schema_version": m.MODEL_SCHEMA_VERSION, "protocol": PROTOCOL, "data_checks": checks,
        "validation_status": "research_only; previously inspected seasons, no prospective test",
        "training_through": str(history.game_date.max().date()), "history_latest_game": str(history.game_date.max().date()),
        "training_games": int(history.game_id.nunique()), "feature_names": final_model.wnba_features_,
        "selected_settings": selected, "validation": fold_scores, "paired_comparisons": comparisons,
        "trusted_close_only_2025_sensitivity": trusted_check, "off_grid_2025_sensitivity": off_grid_check,
        "exploratory_2026_replays": replays,
        "source_hashes": sources,
        "raw_odds_manifest_sha256": hashlib.sha256(json.dumps(odds_manifest, sort_keys=True).encode()).hexdigest(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
    }
    joblib.dump({"schema_version": m.MODEL_SCHEMA_VERSION, "model": final_model,
                 "features": final_model.wnba_features_, "report": report}, m.MODEL_PATH)
    m.REPORT_PATH.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    review_and_train()
