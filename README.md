# WNBAGPT

Research tools for estimating WNBA full-game over/under probabilities from a pregame market anchor, the current score and clock, historical game states, and team scoring in **previous completed games**.

This repository contains source code, tests, and methodology only. **Training data, trade-audit records, databases, credentials, trained model binaries, and generated prediction files are deliberately excluded.** A fresh clone can run the unit tests, but training and real-data prediction require separately supplied local inputs.

## Status and scope

This is a probability research project, not a trading bot. It has no live score feed, exchange connection, account access, order placement, or maker/taker execution loop. The original algorithm creator's trading parameters were context for the research; they are not implemented exchange controls in this repository.

The independently trained model learns actual game outcomes, not audit theos. It remains **research-only**: validation has not established an improvement over the empirical benchmark, and performance between training snapshot times is a particular concern. Prediction output says `research_only: true`.

There are two separate workflows:

1. **Historical reconstruction:** `wnba_theo.py` recreates an empirical remaining-points pricing rule and can compare that rule with a locally supplied audit. This preserves the original investigation.
2. **Independent modeling:** `independent_wnba_model.py` and `model_review.py` train from game outcomes and review alternatives chronologically. They never read the audit database, logged theos, or the reconstructed pricing table.

## Files

| File | Purpose |
| --- | --- |
| `independent_wnba_model.py` | Historical features, classifier training, probability adjustment, and manual prediction CLI |
| `model_review.py` | Data checks, chronological candidate comparison, uncertainty intervals, benchmark replays, and final refit |
| `wnba_theo.py` | Original empirical reconstruction, team-form variants, walk-forward backtest, and optional read-only audit comparison |
| `test_independent_wnba_model.py` | Independent-model tests using synthetic fixtures and mocked local inputs |
| `test_wnba_theo.py` | Empirical probability and chronological-sampling tests |
| [INDEPENDENT_MODEL.md](INDEPENDENT_MODEL.md) | Detailed independent-model findings, assumptions, and local review results |
| [MODEL.md](MODEL.md) | Historical reconstruction notes; these are not the independent model's current validation protocol |
| [DATA_SCHEMA.md](DATA_SCHEMA.md) | Required private input paths, fields, and timing conventions; no data records |
| `requirements.txt` | Pinned dependencies from the reviewed environment |
| `.gitignore` | Excludes private inputs, credentials, archives, and generated artifacts |

## End-to-end process

### 1. Supply and validate local history

Inputs are completed-game metadata and final scores, second-by-second regulation score timelines, pregame ratings, archived odds, and raw play-by-play summaries. They must be prepared separately in the layout documented in [DATA_SCHEMA.md](DATA_SCHEMA.md). This repository does not include the original collection/ETL pipeline or download the data automatically.

The local review used 2,642 completed games from 2016 through September 24, 2026. Of these, 2,629 had usable timelines. Independent training used 2,189 games from 2018 onward with eligible recorded totals. These are descriptive counts from that private snapshot, not bundled inputs.

Validation checks final-score sums, duplicate IDs, prior-game rating calculations and joins, rating dates, within-season Elo transitions, and a season-stratified sample of raw play-by-play against processed scores. The review refuses to train when its checked score/rating invariants fail. A sample check is not certification of every event.

Completed games without usable play-by-play still count toward team form in the independent model; only state-level training requires a timeline. The original reconstruction script is preserved separately and retains its earlier timeline-filtered history behavior.

### 2. Compute team form from previous games

For each team, the independent model takes the **last five completed games before the current game's date**, regardless of whether that team played at home or away. It computes six explicit input features:

| Feature | Meaning |
| --- | --- |
| `home_total_5`, `away_total_5` | Mean combined final score in each team's last five games |
| `home_scored_5`, `away_scored_5` | Mean points scored by each team in those games |
| `home_allowed_5`, `away_allowed_5` | Mean points conceded by each team in those games |

These are averages of **past games**, not the last five minutes of the current game. They are actual inputs to the saved full model, not merely diagnostic columns. The model also receives the number of available prior games, rest days, the league's recent-50-game scoring average, prior-20-game offense/defense ratings, and pregame Elo.

Same-date outcomes are withheld until all that day's pregame features have been built. With fewer than five prior games, the available games are averaged and their count is supplied; no history yields missing values. Form can carry over an offseason. Final-score form includes overtime, and there is no explicit roster or injury adjustment.

### 3. Construct current game states

The live state supplies elapsed regulation time and each team's current score. These determine remaining time, combined score, lead size, and points needed to exceed a target line. Regulation is represented as 2,400 seconds.

The full model also retains optional-in-concept but currently required-by-the-CLI pace inputs: scores 300 seconds earlier and at the current quarter's start. These produce recent-minute scoring, period scoring, and pace features. **They supplement prior-game form; they do not replace it.**

Training samples elapsed seconds 0, 150, 300, ..., 2250. The model does not support the final 150 seconds of regulation or overtime input states. Final outcomes include overtime, so a target contract must use compatible settlement rules.

### 4. Define targets and recency weights

Each sampled state is expanded to nine half-point total strikes around its recorded line, at offsets `[-24, -16, -8, -4, 0, 4, 8, 16, 24]`. Integer archive lines are shifted to a half-point center to avoid pushes. The label is whether the actual final combined score exceeded the synthetic strike.

All rows from one game stay on the same chronological side of evaluation. Synthetic strikes and multiple states do not become independent games for uncertainty calculations. Each game receives equal total snapshot weight before further weighting.

Base training weights favor recent seasons through exponential decay:

```text
age_weight = 2 ** (-age_in_years / half_life_years)
weight = age_weight * odds_source_weight * playoff_multiplier
```

The retained full model uses a three-year half-life and no extra playoff multiplier. Explicit closing lines have source weight 1.0; older `current_snapshot` lines have weight 0.6. Downweighting uncertain odds does not eliminate possible look-ahead bias from unknown capture times.

### 5. Fit and anchor probabilities

The independent estimator is a `HistGradientBoostingClassifier` with a monotonic constraint: requiring more points cannot increase the over probability while other inputs are fixed. The retained full configuration uses 38 features, learning rate 0.06, 180 iterations, at most 15 leaves, minimum leaf size 140 augmented rows, L2 regularization 8, and a fixed random seed. Early stopping is disabled to avoid a random internal validation split across correlated game states.

The model smooths strike prices between four-point grid values. It then adjusts log-odds toward the pregame market anchor:

```text
p = sigmoid(logit(p_raw)
            + remaining_fraction * (logit(p_market_anchor) - logit(p_model_at_tip)))
```

A true 50/50 market midpoint uses `p_market_anchor = 0.5`. A listed closing line is not necessarily 50/50: historical evaluation matches archived closing prices by provider and line, converts American odds to implied probabilities, and proportionally normalizes over/under probabilities. Unavailable price pairs fall back to 0.5. This is an approximation to removing margin, not a guarantee of fair odds.

The adjustment matches the anchor probability at a zero-score tip and decays linearly with remaining regulation time. It does not guarantee calibration later in the game. If the current score has already exceeded the target, over probability is 1, assuming no subsequent official score correction.

### 6. Evaluate without fitting to audits

`model_review.py` records a fixed candidate protocol before running fits:

- Full 38-feature model, three-year recency half-life, no playoff multiplier.
- Compact 19-feature model with market-centered scoring/form, fewer/smaller trees, and a larger minimum leaf size.
- The same compact model with playoff weight 2.5 instead of 1.0.
- The same compact model with a 1.5-year recency half-life instead of three years.

Recency and playoff changes are isolated within the compact family; the compact architecture itself is a joint simplification, not a one-feature ablation. Candidate selection uses 2024 and 2025, with each year's training cutoff at January 1. Pregame feature values for evaluation games can incorporate games completed before their prediction date, even while model parameters remain frozen.

The primary metric is equal-game mean **Brier loss** across eight live times at the listed closing half-point line; lower is better. Tip is excluded because anchoring forces a market match there. Nine-strike Brier, log loss, calibration bins, and playoff-only results are also reported.

A replacement must improve both years, pooled nine-strike Brier, and have a paired game-bootstrap 95% upper bound below zero. Resampling keeps all a game's states together. Weighting variants must also beat the compact control. The small playoff sample and shared-team dependence limit certainty.

2026 had already been inspected before an earlier anchoring change, so **2026 is exploratory replay, not an untouched holdout**. This pass does not select candidates from those results. Earlier development seasons are not pristine independent tests either.

### 7. Check sensitivities and compare a simpler benchmark

After selection is frozen, the review checks training on declared closing lines only and evaluation at nontraining times (375, 825, 1275, 1725, 2175 seconds). These diagnostics do not change the selection.

The empirical comparison uses historical remaining points at the same elapsed second, a decaying market adjustment, and the same training cutoff as the fitted model. It uses current-season history when at least 80 prior games exist, otherwise recency-weighted pooled seasons. It is calculated from game outcomes, not audit targets.

For the original reconstruction in `wnba_theo.py`, the central idea is:

```text
points_needed = target_total - current_score
remaining_points = historical_final_score - historical_score_at_same_time
anchor_shift = (pregame_50_50_total - historical_median_final_total)
               * remaining_fraction
probability_over = weighted_fraction(remaining_points + anchor_shift > points_needed)
```

The implementation interpolates fractional point shifts. Its `team`, `pooled`, and `playoff` variants add prior-game form and different historical weights. `audit-check` is an optional, read-only comparison for that reconstruction only. The independent pipeline never invokes it.

### 8. Refit and save local artifacts

After validation, the selected configuration is refit on all eligible uploaded history. Training writes the estimator, schema/feature metadata, aggregate report, candidate protocol, frozen selection, and per-state diagnostic predictions under `models/`. Versions and hashes record the local sources used. **All of these generated files remain excluded from Git.**

The prediction CLI rejects incompatible model schemas, changed game/ratings inputs, unsupported times, invalid probabilities, integer target totals, and targets outside the supported range. Hash checks cannot detect games that were never uploaded. Refresh history and ratings, then retrain, before calling later-game predictions current. Load only model artifacts you trust; `joblib` files are not a safe interchange format for untrusted inputs.

## Results and interpretation

The last local review retained the full model because the compact alternative's average gain was not decisive under the fixed replacement rule. This does not prove the full model is superior.

| Candidate | 2024 Brier | 2025 Brier |
| --- | ---: | ---: |
| Full | 0.16984 | 0.15364 |
| Compact | 0.16551 | 0.15175 |
| Compact + playoff weighting | 0.16637 | 0.15128 |
| Compact + faster recency weighting | 0.16643 | 0.15162 |

Compact minus full had mean loss difference about -0.0030, with paired game-bootstrap 95% interval approximately [-0.0062, +0.00034]. The interval includes no improvement.

On the 329-game exploratory 2026 replay trained before 2026, full scored 0.14713 versus empirical 0.14451. On the matching 307-game **2025 off-grid diagnostic**, full scored **0.17471 versus empirical 0.15717**; the loss-difference interval was [+0.00617, +0.02857]. The off-grid weakness is a reason not to treat the richer model as ready for arbitrary-second live use. Results do not establish profitable trading and are not comparable to earlier reports using a different time mix/protocol.

See [INDEPENDENT_MODEL.md](INDEPENDENT_MODEL.md) for additional comparisons and [MODEL.md](MODEL.md) for the separate reconstruction findings. Numerical summaries are documentation of local experiments; no underlying records are published.

## Setup and tests

Use Python 3.9 or later. The reviewed local environment used Python 3.9.6 with the dependency versions in `requirements.txt`.

```sh
git clone https://github.com/alsoar/WNBAGPT.git
cd WNBAGPT
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -v
```

The tests are data-free. They cover prior-date and last-five-game form, home/away role changes, missing timelines, closing-odds matching, target monotonicity and anchoring, already-crossed totals, input/schema checks, exact time lookbacks, raw clock parsing, and game-level uncertainty pairing.

## Training and manual prediction

First supply the private files described in [DATA_SCHEMA.md](DATA_SCHEMA.md). Training is deterministic under the recorded versions/seeds but can take several minutes. It regenerates the review and selected local artifact:

```sh
LOKY_MAX_CPU_COUNT=4 OMP_NUM_THREADS=4 python independent_wnba_model.py train
```

`python model_review.py` runs the same workflow without printing the entire final report to the terminal. This is not an unattended retraining service.

Illustrative pregame prediction (not a claim about a scheduled game):

```sh
python independent_wnba_model.py predict \
  --date 2026-09-30 --season 2026 \
  --home 'Las Vegas Aces' --away 'Indiana Fever' \
  --second 0 --home-score 0 --away-score 0 \
  --anchor-line 183.5 --anchor-probability 0.5 --line 185.5 --playoff
```

An illustrative in-game state with the pace inputs retained:

```sh
python independent_wnba_model.py predict \
  --date 2026-09-30 --season 2026 \
  --home 'Las Vegas Aces' --away 'Indiana Fever' \
  --second 675 --home-score 26 --away-score 25 \
  --home-prev-300 12 --away-prev-300 13 \
  --home-period-start 21 --away-period-start 20 \
  --anchor-line 160.5 --anchor-probability 0.5 --line 162.5 --playoff
```

Team form is loaded from prior games automatically; it is not supplied through those minute-lookback arguments. The anchor must describe the **pregame** line, not the live in-game market. Times are elapsed seconds in 0..2250. Target lines must end in .5 and lie within 24 points of the anchor. Names must match the local history. Dates must follow the uploaded games, and ratings must be from an earlier date in the requested season. The example data cutoff was September 24, 2026; the example does not fill the intervening gap.

Original empirical workflow, with appropriate local inputs:

```sh
python wnba_theo.py backtest
python wnba_theo.py predict --date 2026-09-30 --season 2026 \
  --second 600 --score 42 --line 166.5 --market-median 173.5 \
  --home 'Minnesota Lynx' --away 'New York Liberty'
# Requires the private audit database and archived pricing table:
python wnba_theo.py audit-check
```

## Limitations and next evidence

- Unverified timestamps on older odds can cause leakage; reduced weight is not a cure.
- Retrospective score corrections and same-clock event aggregation are not as-received live feeds.
- Last-five form can span offseasons and does not explicitly model injuries, roster changes, or current possessions.
- Market anchoring, linear decay, strike interpolation, and source weights remain assumptions.
- Fixed training times may contribute to poor between-snapshot behavior; the diagnostic does not prove causation.
- Model selection seasons have already been inspected. A fixed prospective evaluation on genuinely unseen games is needed.
- Fills, adverse selection, latency, fees, liquidity, inventory, and settlement compatibility are outside these tests.

One justified next experiment is predetermined stratified training-time sampling, followed by prospective validation rather than repeatedly tuning against the same diagnostic games.

## Keeping data private

Do not commit `data/`, `models/`, `runs/`, audit files, database files, archives, credentials, or generated per-game/per-state predictions. The ignore rules are guardrails, not access controls; `git add -f` can bypass them. Stage named source files and inspect `git diff --cached --stat` and `git diff --cached --name-only` before pushing. This repository intentionally contains no training data or pretrained model download.
