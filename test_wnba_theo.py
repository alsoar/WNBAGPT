import unittest

import numpy as np
import pandas as pd

from wnba_theo import form_features, over_probability, predict_from_sample, sample_for_state


class TheoTests(unittest.TestCase):
    def test_fractional_anchor_moves_probability_between_integer_outcomes(self):
        remaining = np.array([166, 167, 168])
        weights = np.ones(3)
        self.assertAlmostEqual(over_probability(remaining, 166.5, 0, weights), 2 / 3)
        self.assertAlmostEqual(over_probability(remaining, 166.5, 1 / 3, weights), 7 / 9)

    def test_team_form_excludes_games_on_the_same_date(self):
        games = pd.DataFrame({
            "game_id": ["1", "2", "3", "4", "5"],
            "game_date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03",
                                         "2025-01-04", "2025-01-04"]),
            "home_team_id": ["A"] * 5, "away_team_id": ["B"] * 5,
            "total_points": [100, 120, 140, 200, 10],
            "ou_line": [130.5] * 5,
        })
        features = form_features(games).set_index("game_id")
        self.assertTrue(np.isnan(features.loc["3", "recent_total"]))
        self.assertEqual(features.loc["4", "recent_total"], 120)
        self.assertEqual(features.loc["5", "recent_total"], 120)

    def test_historical_sample_uses_only_earlier_games(self):
        games = pd.DataFrame({
            "game_id": ["old", "same_day", "future"],
            "game_date": pd.to_datetime(["2025-09-01", "2025-09-02", "2025-09-03"]),
            "season_end": [2025, 2025, 2026],
        })
        states = pd.DataFrame({"game_id": games.game_id, "second": [600] * 3, "score": [40] * 3})
        sample = sample_for_state(games, states, pd.Timestamp("2025-09-02"), 600, 2025, True)
        self.assertEqual(sample.game_id.tolist(), ["old"])

    def test_integer_line_is_rejected(self):
        sample = pd.DataFrame({
            "total_points": [160] * 30, "score": [40] * 30,
            "game_date": pd.to_datetime(["2025-01-01"] * 30),
            "season_end": [2025] * 30, "season_type": ["regular"] * 30,
        })
        with self.assertRaisesRegex(ValueError, "integer lines can push"):
            predict_from_sample(sample, asof=pd.Timestamp("2025-02-01"), second=600,
                                score=40, line=160.0, market_median=160.0)


if __name__ == "__main__":
    unittest.main()
