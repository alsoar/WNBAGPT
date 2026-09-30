import unittest
import gzip
import json
import tempfile
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.special import expit

import independent_wnba_model as model
import model_review as review


class IndependentModelTests(unittest.TestCase):
    def row(self, **values):
        row = {name: 0.0 for name in model.FEATURES}
        row.update(anchor_line=160.5, points_needed=160.5, remaining_seconds=2400, period=1)
        row.update(values)
        return pd.DataFrame([row])

    def arguments(self, **values):
        args = dict(date="2026-09-29", season=2026, home="Home", away="Away", second=0,
                    home_score=0, away_score=0, home_prev_300=None, away_prev_300=None,
                    home_period_start=None, away_period_start=None, anchor_line=160.5,
                    line=160.5, anchor_probability=0.5, playoff=False, neutral_site=False)
        args.update(values)
        return Namespace(**args)

    def test_recent_scoring_uses_only_earlier_dates(self):
        games = pd.DataFrame({
            "game_id": ["1", "2", "3", "4"],
            "game_date": pd.to_datetime(["2025-01-01", "2025-01-02",
                                         "2025-01-03", "2025-01-03"]),
            "season_end": [2025] * 4,
            "home_team_id": ["A"] * 4, "away_team_id": ["B"] * 4,
            "home_score": [40, 50, 80, 5], "away_score": [60, 70, 120, 5],
            "total_points": [100, 120, 200, 10],
        })
        features = model.pregame_features(games).set_index("game_id")
        self.assertEqual(features.loc["3", "home_total_5"], 110)
        self.assertEqual(features.loc["4", "home_total_5"], 110)
        self.assertEqual(features.loc["3", "league_recent_50"], 110)
        self.assertEqual(features.loc["4", "league_recent_50"], 110)

    def test_last_five_completed_games_follow_teams_across_home_away_roles(self):
        records = []
        a_scores = [20, 70, 80, 90, 100, 110, 900]
        b_scores = [40, 60, 65, 70, 75, 80, 800]
        for index, (a, b) in enumerate(zip(a_scores, b_scores)):
            a_home = index % 2 == 0
            records.append({
                "game_id": str(index), "game_date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=index),
                "season_end": 2025, "home_team_id": "A" if a_home else "B",
                "away_team_id": "B" if a_home else "A", "home_score": a if a_home else b,
                "away_score": b if a_home else a, "total_points": a + b,
            })
        form = model.pregame_features(pd.DataFrame(records)).set_index("game_id").loc["6"]
        expected = {
            "home_total_5": 160, "home_scored_5": 90, "home_allowed_5": 70,
            "away_total_5": 160, "away_scored_5": 70, "away_allowed_5": 90,
        }
        for name, value in expected.items():
            self.assertIn(name, model.FEATURES)
            self.assertEqual(form[name], value)
        self.assertEqual(form.home_games_prior, 5)
        self.assertEqual(form.away_games_prior, 5)

    def test_integer_archive_line_creates_half_point_strikes(self):
        row = {name: 0.0 for name in model.FEATURES}
        row.update({"anchor_line": 160.0, "score_total": 40.0, "points_needed": 120.0})
        frame = pd.DataFrame([{**row, "total_points": 161.0}])
        matrix, labels = model.as_matrix(frame, np.array([-4, 0, 4], dtype=np.float32))
        self.assertEqual(matrix[:, model.FEATURES.index("points_needed")].tolist(),
                         [116.5, 120.5, 124.5])
        self.assertEqual(labels.tolist(), [1, 1, 0])

    def test_anchor_is_half_at_tip_and_prices_fall_with_line(self):
        class SmoothFakeModel:
            def predict_proba(self, matrix):
                needed = matrix[:, model.FEATURES.index("points_needed")]
                yes = expit((170 - needed) / 10)
                return np.column_stack([1 - yes, yes])

        row = {name: 0.0 for name in model.FEATURES}
        row.update({"anchor_line": 173.5, "points_needed": 173.5,
                    "remaining_seconds": 2400, "period": 1})
        frame = pd.DataFrame([row])
        fake = SmoothFakeModel()
        probabilities = []
        for line in (169.5, 171.5, 173.5, 175.5, 177.5):
            frame["points_needed"] = line
            probabilities.append(model.anchored_probabilities(fake, frame)[0])
        self.assertAlmostEqual(probabilities[2], 0.5)
        self.assertTrue(all(a > b for a, b in zip(probabilities, probabilities[1:])))

    def test_explicit_market_probability_at_tip_for_both_feature_sets(self):
        for features, needed in ((model.FEATURES, "points_needed"),
                                 (model.CENTERED_FEATURES, "points_needed_centered")):
            class FakeModel:
                wnba_features_ = features

                def predict_proba(self, matrix):
                    yes = expit(-matrix[:, features.index(needed)] / 30)
                    return np.column_stack([1 - yes, yes])

            row = model.centered_features(self.row(anchor_probability=0.46))
            self.assertAlmostEqual(model.anchored_probabilities(FakeModel(), row)[0], 0.46)
            row["points_needed"] = 164.5
            row = model.centered_features(row)
            self.assertLess(model.anchored_probabilities(FakeModel(), row)[0], 0.46)
            row["anchor_probability"] = np.nan
            with self.assertRaisesRegex(ValueError, "Anchor probabilities"):
                model.anchored_probabilities(FakeModel(), row)

    def test_centered_strikes_match_full_strikes(self):
        rows = model.centered_features(self.row(second=1200, remaining_seconds=1200,
                                                score_total=72, points_needed=88.5, total_points=171))
        full, full_y = model.as_matrix(rows, model.STRIKE_OFFSETS)
        centered, centered_y = model.as_matrix(rows, model.STRIKE_OFFSETS, model.CENTERED_FEATURES)
        np.testing.assert_array_equal(full_y, centered_y)
        np.testing.assert_allclose(centered[:, model.CENTERED_FEATURES.index("points_needed_centered")],
                                   full[:, model.FEATURES.index("points_needed")] - 80.25)

    def test_crossed_threshold_is_certain_in_both_feature_sets(self):
        for features in (model.FEATURES, model.CENTERED_FEATURES):
            class FakeModel:
                wnba_features_ = features

                def predict_proba(self, matrix):
                    return np.tile([0.1, 0.9], (len(matrix), 1))

            row = model.centered_features(self.row(second=2100, remaining_seconds=300,
                                                   score_total=161, points_needed=-0.5))
            self.assertEqual(model.anchored_probabilities(FakeModel(), row)[0], 1)
            row["points_needed"] = 0.5
            row = model.centered_features(row)
            self.assertLess(model.anchored_probabilities(FakeModel(), row)[0], 1)

    def test_training_cutoff_rejects_current_and_future_games(self):
        rows = pd.DataFrame({"game_date": pd.to_datetime(["2025-01-01"]),
                             "ou_line_status": ["explicit_close"], "season_type": ["regular"]})
        for cutoff in ("2025-01-01", "2024-12-31"):
            with self.assertRaisesRegex(ValueError, "cutoff"):
                model.row_weights(rows, pd.Timestamp(cutoff), 3, 1)
        self.assertGreater(model.row_weights(rows, pd.Timestamp("2025-01-02"), 3, 1)[0], 0)

    def test_missing_timeline_does_not_remove_valid_final_from_form(self):
        games = pd.DataFrame({
            "game_id": ["a", "b"], "game_date": ["2025-01-01", "2025-01-02"],
            "season_end": [2025, 2025], "timeline_available": [False, True],
            "home_team_id": ["1", "1"], "away_team_id": ["2", "2"],
            "home_score": [90, 70], "away_score": [80, 60], "total_points": [170, 130],
        })
        with patch.object(model.pd, "read_csv", return_value=games), \
                patch.object(model, "closing_probabilities", return_value={}):
            loaded = model.load_games()
        self.assertEqual(len(loaded), 2)
        features = model.pregame_features(loaded).set_index("game_id")
        self.assertEqual(features.loc["b", "home_total_5"], 170)

    def test_closing_odds_require_matching_provider_and_line(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            odds = root / "data/raw/wnba_history/odds"
            odds.mkdir(parents=True)
            def item(provider, line, over, under):
                return {"provider": {"name": provider}, "close": {
                    "total": {"american": line}, "over": {"american": over}, "under": {"american": under}}}
            with gzip.open(odds / "123.json.gz", "wt") as stream:
                json.dump({"items": [item("Wrong", 160.5, -110, -110), item("Right", 170.5, -110, -110),
                                     item("Right", 160.5, 120, -140)]}, stream)
            games = pd.DataFrame([dict(game_id="123", ou_provider="Right", ou_line=160.5,
                                       ou_line_status="explicit_close")])
            with patch.object(model, "ROOT", root):
                result = model.closing_probabilities(games)
            expected = (100 / 220) / (100 / 220 + 140 / 240)
            self.assertAlmostEqual(result["123"], expected)

    def test_prediction_rejects_unsupported_time_and_nonfinite_prices(self):
        with patch.object(model, "MODEL_PATH") as path:
            path.exists.return_value = True
            for second in (-1, 2251, 2400):
                with self.assertRaisesRegex(ValueError, "Elapsed second"):
                    model.predict(Namespace(second=second))
            for anchor, line in ((np.nan, 160.5), (160.5, np.inf), (0, 160.5)):
                with self.assertRaisesRegex(ValueError, "finite"):
                    model.predict(Namespace(second=0, anchor_line=anchor, line=line))

    def test_old_model_and_changed_inputs_fail_before_prediction(self):
        with patch.object(model, "MODEL_PATH") as path, patch.object(model.joblib, "load") as load:
            path.exists.return_value = True
            load.return_value = {"schema_version": 1}
            with self.assertRaisesRegex(ValueError, "schema"):
                model.predict(self.arguments())
            load.return_value = {
                "schema_version": model.MODEL_SCHEMA_VERSION, "features": model.FEATURES,
                "model": SimpleNamespace(wnba_features_=model.FEATURES),
                "report": {"source_hashes": {str(model.GAMES_PATH.relative_to(model.ROOT)): "changed"}},
            }
            with patch.object(model, "future_row") as future, \
                    patch.object(Path, "read_bytes", return_value=b"changed input fixture"):
                with self.assertRaisesRegex(ValueError, "changed since training"):
                    model.predict(self.arguments())
                future.assert_not_called()

    def test_future_ratings_must_precede_prediction_in_same_season(self):
        games = pd.DataFrame([dict(game_date=pd.Timestamp("2026-09-24"), home_team="Home",
                                   away_team="Away", home_team_id="1", away_team_id="2")])
        for season, date in ((2026, "2026-09-29"), (2025, "2025-09-24")):
            ratings = pd.DataFrame([dict(team=team, season_end=season, as_of_date=date)
                                    for team in ("Home", "Away")])
            with patch.object(model, "load_games", return_value=games), \
                    patch.object(model.pd, "read_csv", return_value=ratings):
                with self.assertRaisesRegex(ValueError, "earlier date"):
                    model.future_row(self.arguments())

    def test_off_grid_states_use_exact_previous_scores(self):
        raw = pd.DataFrame({"game_id": ["g"] * 3, "second": [375, 600, 675],
                            "home_score": [12, 20, 23], "away_score": [11, 19, 25]})
        with patch.object(model, "SECONDS_DIR") as directory, \
                patch.object(model.pd, "read_parquet", return_value=raw):
            directory.glob.return_value = [Path("fake.parquet")]
            state = model.load_states([675]).iloc[0]
        self.assertEqual(state.home_prev_300, 12)
        self.assertEqual(state.away_prev_300, 11)
        self.assertEqual(state.home_period_start, 20)
        self.assertEqual(state.away_period_start, 19)

    def test_raw_clock_supports_fractional_seconds_without_minutes(self):
        self.assertEqual(review.clock_seconds("10:00"), 600)
        self.assertEqual(review.clock_seconds("1:05.5"), 65.5)
        self.assertEqual(review.clock_seconds("37.0"), 37)
        self.assertEqual(review.clock_seconds("0.0"), 0)

    def test_uncertainty_pairs_games_not_individual_snapshots(self):
        left = pd.DataFrame({"game_id": ["a", "a", "b", "b"], "brier": [0.1, 0.1, 0.2, 0.2]})
        right = left.assign(brier=left.brier + 0.01)
        result = review.paired_interval(left, right)
        self.assertEqual(result["games"], 2)
        self.assertAlmostEqual(result["mean_brier_difference"], -0.01)
        with self.assertRaisesRegex(ValueError, "same games"):
            review.paired_interval(left, right.loc[right.game_id == "a"])


if __name__ == "__main__":
    unittest.main()
