# Independently trained WNBA totals model

The model learns game outcomes, not audit probabilities. Neither `independent_wnba_model.py` nor its training/review harness reads the trading audit or the old pricing table. This is an offline research model, not a live trading integration.

## Review conclusion

The second pass corrected data handling and evaluation problems. It did **not** establish a reliable predictive improvement from a new architecture or additional weighting. The saved artifact retains the full 38-feature model with a three-year recency half-life and no extra playoff multiplier, after the correctness fixes below. This is conservative retention, not proof that the full model is superior.

The smaller alternative improved average validation Brier by about 0.0030, but its paired game-bootstrap 95% interval was approximately [-0.0062, +0.00034]. Because that interval includes no improvement, it did not satisfy the replacement rule. More playoff weight and faster recency decay did not consistently improve the smaller model.

The richer model still has worse average Brier than the empirical benchmark in the exploratory 2026 comparisons. These losses measure probability accuracy, not trading profitability.

## What was corrected

- **Validation claims:** the previous pass inspected 2026 before adding market anchoring. Accordingly, 2026 is an exploratory chronological replay, not an untouched holdout. It was not used to select candidates in this pass. The 2024/25 comparisons are development validation, not a new independent test either.
- **Team form:** all completed games now contribute to prior-game form, including 13 games with valid final scores but unavailable timelines. Only state-level training requires a valid timeline.
- **Closing probabilities:** a listed closing total is not always priced 50/50. Where provider and line match, archived American prices are converted to implied probabilities and proportionally normalized. This is approximate margin removal, not knowledge of the true fair probability.
- **Benchmark cutoffs:** the empirical benchmark and fitted model now use the same training cutoff, games, and evaluation times. The old comparison let the empirical distribution update during the model's frozen window. Current-game inputs and prior-date team form remain available as of each prediction date.
- **Weight attribution:** recency and playoff variants change one factor at a time within the compact family. The old comparison changed both simultaneously.
- **Early pace:** scoring pace before five minutes now uses actual elapsed time rather than a five-minute denominator.
- **Logical/input bounds:** an already-crossed half-point total has over probability 1, assuming no later score correction. Prediction rejects unsupported late times, invalid probabilities/prices, incompatible artifacts, and game/ratings files changed since training.
- **Reproducibility:** training saves the protocol, selection comparisons, per-state predictions, versions, input/code hashes, and a raw-odds manifest hash. Each game has equal total snapshot weight before recency/source/playoff weights.

## Data and leakage checks

Uploaded history contains 2,642 completed games from 2016 through September 24, 2026, with 2,629 timelines. The fitted sample contains 2,189 games with usable recorded lines from 2018 onward: 35,024 sampled states before synthetic strikes.

- No duplicate game IDs or final-score sum mismatches.
- Team last-five and league recent-50 scoring use strictly earlier dates. Same-day games are withheld until all that day's features are computed.
- 8,692 rolling offense/defense rating values match prior-20-game points divided by possessions to numerical precision. Recorded rating end dates precede the game.
- 5,148 within-season Elo transitions exactly match the previous game's postgame Elo. Joined game ratings match the rating table. The inference Elo expectation formula matches the uploaded values.
- 1,056 states across 66 season-stratified games match raw play-by-play, including quarter boundaries. The checker handles both `1:05` and `37.0` clocks. This is a sample, not an exhaustive certification of every play.
- Closing probabilities match for 909 of 911 declared closing-line games. Missing pairs default to 0.5 and are labeled `assumed_half` in loaded data.

These checks do not prove live availability. Older `current_snapshot` odds have **no verified pregame capture time**. Downweighting them to 60% does not remove potential leakage. A closing-line-only training sensitivity scored approximately 0.1636 Brier in 2025 versus 0.1536 using broader history, but it also had far fewer training games. That cannot prove older odds are safe or isolate the cause of the difference. Retrospective play-by-play can contain later corrections, and same-clock events share their final score state.

## Features and probabilities

The full model uses live score, absolute lead, recent five-minute scoring, period scoring, elapsed/remaining time, and pace. Pregame features include the anchor line, each team's last-five totals and points for/against, prior-date offense/defense ratings, Elo, rest, recent league scoring, season, and postseason status. Form can carry across an offseason; roster and injury adjustments are absent.

Training samples every 150 seconds from 0 through 2250 elapsed seconds. Binary labels indicate finishing above nine half-point strikes around each recorded total. Synthetic strikes and snapshots are correlated observations, not extra independent games. Recency weights halve every three years. Recent team/league scoring and the market anchor also adapt to scoring-level changes.

Final-score labels include overtime; input states are regulation only. This assumes the target contract includes overtime in settlement. The model does not support quoting in overtime or the final 150 seconds of regulation.

Probability is constrained to decrease as the target line rises. Four-point strike interpolation smooths tree plateaus. A log-odds adjustment matches the anchor's market probability at a zero-score tip, then decays linearly with remaining regulation time. A true 50/50 midpoint uses 0.5; a listed line with skewed prices should use an explicit anchor probability.

Linear decay, four-point interpolation, last-five form, and the 60% uncertain-source weight remain modeling assumptions, not quantities optimized on the 2026 replay. Anchoring at tip does not guarantee live calibration.

## Selection protocol

`model_review.py` writes `models/review_protocol.json` before training. Four fixed candidates train strictly before January 1 of each validation year:

| Candidate | Features / tree limits | Recency half-life | Playoff weight | 2024 Brier | 2025 Brier |
| --- | --- | ---: | ---: | ---: | ---: |
| Full | 38 features; 180 iterations, 15 leaves | 3 years | 1.0 | 0.16984 | 0.15364 |
| Compact | 19 centered features; 120 iterations, 7 leaves | 3 years | 1.0 | 0.16551 | 0.15175 |
| Compact + playoffs | Same compact architecture | 3 years | 2.5 | 0.16637 | 0.15128 |
| Compact + recent | Same compact architecture | 1.5 years | 1.0 | 0.16643 | 0.15162 |

Compact expresses scoring/form relative to the market anchor, removes redundant raw features, and raises minimum leaf size from 140 to 500 augmented rows. This is a joint simplification, not an isolated feature test.

Primary Brier is equal-game average loss at elapsed seconds 300, 600, 900, 1200, 1500, 1800, 2100, and 2250, at the listed closing half-point line. Validation contains 260 games in 2024 and 307 in 2025. Tip is excluded because forcing the anchor there dilutes differences in live pricing. Nine-strike and playoff-only scores are also reported.

A replacement must improve each year, pooled nine-strike Brier, and have a paired game-bootstrap upper 95% bound below zero. Weighting variants must also pass against compact. All snapshots from a game stay together in the 3,000 bootstrap resamples. Compact failed the uncertainty requirement, so full remains selected. Only 22 and 23 playoff games occur in the validation years; playoff conclusions are weak. Bootstrap intervals do not fully account for shared-team/season dependence or all prior model experimentation.

## Exploratory replay

The final artifact is refit on all eligible uploaded games only after selection. These scores use separate fits with cutoffs preceding the replay games, not the final artifact's own training set:

| Replay window | Training before | Games | Full model Brier | Empirical Brier |
| --- | --- | ---: | ---: | ---: |
| Available 2026 season | Jan 1, 2026 | 329 | 0.14713 | 0.14451 |
| Jul 1-Aug 14, 2026 | Jul 1, 2026 | 111 | 0.13942 | 0.13662 |
| Aug 15-Sep 24, 2026 | Aug 15, 2026 | 75 | 0.16545 | 0.16302 |

All three paired intervals include zero. Windows overlap and must not be pooled as independent tests. The benchmark is an empirical remaining-score distribution with a decaying market adjustment. It uses current-season history when at least 80 prior games exist, otherwise recency-weighted pooled seasons. It is not an audit-matching target.

These scores are **not directly comparable to the old report**: snapshot mix, anchoring, and benchmark protocol changed. The older artifact/report are preserved as `models/wnba_independent_v1.joblib` and `models/wnba_independent_report_v1.json`.

An additional 2025 diagnostic evaluates nontraining times 375, 825, 1275, 1725, and 2175 seconds. On those same times and 307 games, full scored **0.17471 versus empirical 0.15717**, a loss difference of +0.01754 with game-bootstrap 95% interval [+0.00617, +0.02857]. This is a material warning against arbitrary-second live use. It is consistent with possible fixed-time-grid effects, but does not by itself establish their cause. The diagnostic cannot change selection, and comparing its average with the main grid's average would confound time mix. The next justified experiment is predetermined, stratified sampling of training times, followed by prospective validation, not repeated tuning against this diagnostic.

## Use

```sh
LOKY_MAX_CPU_COUNT=4 OMP_NUM_THREADS=4 python3 independent_wnba_model.py train
python3 -m unittest test_independent_wnba_model.py test_wnba_theo.py
python3 independent_wnba_model.py predict \
  --date 2026-09-30 --season 2026 \
  --home 'Las Vegas Aces' --away 'Indiana Fever' \
  --second 0 --home-score 0 --away-score 0 \
  --anchor-line 183.5 --anchor-probability 0.5 --line 185.5 --playoff
```

This input is illustrative, not an assertion that the game is scheduled. `--anchor-probability` defaults to 0.5 and must describe the same pregame total as `--anchor-line`. Do not substitute a live in-game total for the pregame anchor.

At elapsed times of at least 300 seconds, supply both scores from 300 seconds earlier. During later quarters, supply scores at that quarter's start; exact quarter boundaries use current scores. Official downward corrections are possible, so historical scores need not be monotonically increasing.

Supported times are 0..2250. Targets must end in .5 and be within 24 points of the anchor. Team names must match the data. Ratings must precede the requested date and belong to that season. Hash checks detect edited inputs but cannot detect games never uploaded: history still ends September 24, 2026. Refresh history/ratings and retrain before treating later predictions as current.

The artifact is `models/wnba_independent.joblib`; details are in `models/wnba_independent_report.json` and `models/review_*`. Prediction JSON is explicitly marked `research_only`, with an additional warning for between-snapshot times. The tool reads no live feed and places no orders. Stronger evidence now requires prospective predictions fixed before unseen games, verified pregame odds, and as-received live states. Fees, fills, latency, settlement rules, and inventory risk are outside these probability tests.
