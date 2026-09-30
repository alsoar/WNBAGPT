# Private input contract

This document describes the inputs expected by the research code. It contains schema information only. Data collection, licensing, preprocessing, and access to the actual records are outside this repository.

All paths are relative to the source directory. The local layout is:

```text
data/
  processed/
    wnba_history/
      games.csv
      team_game_ratings.csv
      latest_team_ratings.csv
      seconds/
        season_end=YYYY/
          part-0.parquet
    wnba_history_2026/
      ou_differential_over_table.csv       # optional legacy audit check only
  raw/
    wnba_history/
      summaries/
        <game_id>.json.gz
      odds/
        <game_id>.json.gz
wnba-trade-audit.sqlite3                   # optional legacy audit check only
models/                                   # generated locally by training
```

The independent training/review workflow needs processed history, rating tables, raw summaries for its consistency checks, and archived odds when available. The empirical predictor/backtest only needs game metadata and timelines. The optional `audit-check` needs the two additional legacy inputs and is not part of independent training.

## Completed games: `games.csv`

One row per completed game. Extra source columns may remain locally; the code consumes these groups:

| Columns | Meaning / convention |
| --- | --- |
| `game_id` | Unique identifier; read as a string |
| `game_date` | ISO date used for chronological ordering and strict prior-date cutoffs |
| `season_end` | Integer season year |
| `season_type` | `postseason` for playoffs; ordinary regular games use `regular` |
| `home_team_id`, `away_team_id` | Stable team identifiers, read as strings |
| `home_team`, `away_team` | Team names used by the manual prediction CLI |
| `home_score`, `away_score`, `total_points` | Final scores, including overtime; total must equal their sum |
| `timeline_available` | Boolean indicating usable second-by-second history |
| `neutral_site` | Boolean site indicator |
| `ou_line` | Recorded pregame total reference, numeric or missing |
| `ou_line_status` | `explicit_close` or `current_snapshot` for eligible training lines; other statuses excluded |
| `ou_provider` | Provider name used to match a declared closing line to archived prices |
| `home_off_rating_20`, `away_off_rating_20` | Pregame rolling offense rating |
| `home_def_rating_20`, `away_def_rating_20` | Pregame rolling defense rating |
| `home_elo_pre`, `away_elo_pre` | Pregame Elo |
| `home_elo_expected`, `away_elo_expected` | Pregame win expectations consistent with those Elo values |

The independent model computes last-five-game scoring and recent league totals itself; do not prefill those with end-of-season or current-game information. Its form history retains completed games even when `timeline_available` is false. State-level training requires an eligible line, a timeline, and season 2018 or later.

The date convention must be consistent across all inputs. Features use dates strictly less than the prediction date, not within-day game ordering. If source timestamps include timezones, normalize them in the upstream preparation process before making these daily files.

## Timelines: `seconds/season_end=YYYY/part-0.parquet`

Required columns are `game_id`, `second`, `home_score`, and `away_score`. Each `(game_id, second)` must be unique. Scores are the state at that elapsed regulation second, not points scored during the second or future quarter-final scores. `second = 0` is the start of regulation; quarter boundaries are 600, 1200, and 1800.

Provide a dense second-by-second table for regulation. Independent training requests its 150-second grid plus exact 300-second lookbacks and quarter starts. Diagnostic checks request other times. The empirical CLI can request any regulation second through 2399. Final-game overtime points come from `games.csv`, not an overtime timeline input.

Official score corrections can lower a recorded score. Multiple events at one clock time use the prepared file's final state at that clock; this convention does not reproduce event-by-event live latency.

## Historical ratings: `team_game_ratings.csv`

One row per team per game. Review checks use:

```text
game_id, game_date, season_end, team_id,
points_for, points_against, possessions,
off_rating_20, def_rating_20, rating_end_date,
elo_pre, elo_post, elo_expected
```

`off_rating_20 = 100 * sum(prior-20 points_for) / sum(prior-20 possessions)`, and defense uses `points_against`. These are ratios of sums, not unweighted means of each game's efficiency. Prior games must precede the current game's date. Historical pregame values must match the side-specific values joined into `games.csv`.

Within a season, a team's next pregame Elo must match its previous postgame Elo. Across-season initializations are upstream inputs, not inferred from future outcomes by this code. Rating end dates must precede the game.

## Latest ratings: `latest_team_ratings.csv`

One row per team, indexed by its name during prediction. Required fields:

```text
team, season_end, as_of_date, off_rating_20, def_rating_20, elo
```

The snapshot must precede the requested prediction date and belong to that season. Latest `elo` and rolling ratings describe the team's history through the snapshot, not the next game's outcome. Inference uses a 65-point home advantage, zero at a neutral site, and a 400-point logistic Elo scale to reproduce the uploaded expectation convention. Refresh ratings together with completed-game history.

## Raw summaries: `summaries/<game_id>.json.gz`

Gzipped UTF-8 JSON with a `plays` array. Sampled verification reads these fields from each play:

```text
period.number
clock.displayValue
sequenceNumber
homeScore
awayScore
```

Periods 1 through 4 count as regulation. Clocks are remaining time in the period, supporting minute/second strings such as `1:05.5` and seconds-only strings such as `37.0`. Numeric sequence numbers order events with the same clock. Summaries must exist for games selected by the deterministic season-stratified check.

## Raw odds: `odds/<game_id>.json.gz`

Gzipped UTF-8 JSON with an `items` array. Closing-price extraction uses:

```text
items[].provider.name
items[].close.total.american
items[].close.over.american
items[].close.under.american
```

In this archive format, `close.total.american` stores the numeric total line, while `close.over.american` and `close.under.american` store signed American odds. Provider and total must match `games.csv`. For positive odds `a`, implied probability is `100 / (100 + a)`; for negative odds it is `-a / (100 - a)`. The over anchor is the over implied probability divided by the sum of the two implied probabilities.

Missing files, unmatched providers/lines, or unavailable valid price pairs fall back to 0.5 and are labeled `assumed_half` in memory. `explicit_close` is an input provenance label, not proof of a pregame capture timestamp. A `current_snapshot` line has weaker timing assurance.

## Optional legacy audit inputs

`wnba_theo.py audit-check` opens `wnba-trade-audit.sqlite3` read-only and reads `fills.payload_json`. It examines each JSON object's `submission_context` for `elapsed_second`, `differential`, `raw_theo`, `remaining_adjustment`, and `adjusted_theo`.

The companion `ou_differential_over_table.csv` is the original archived remaining-score lookup table, indexed by elapsed second with differential columns. The comparator assumes the supplied archive's exact column order and differential grid; it is not a generic table importer. Do not substitute another layout without adapting the comparison code.

Neither file is needed or read by independent model training. They must remain private and are excluded from Git.

## Local outputs

Training creates `models/wnba_independent.joblib`, `models/wnba_independent_report.json`, `models/review_protocol.json`, `models/review_selection.json`, and validation/replay prediction CSVs. They contain learned state or data-derived results and stay local. A clone intentionally has none of these artifacts; generate them from your authorized local history.
