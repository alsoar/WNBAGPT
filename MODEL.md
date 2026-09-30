# WNBA live totals theo (research tool)

For the separately trained model that does not use the trade audit, see [INDEPENDENT_MODEL.md](INDEPENDENT_MODEL.md).

`wnba_theo.py` uses only the files in this folder. It reads historical games and second-by-second scores; it never fetches live data or sends an order. Python dependencies are `numpy`, `pandas`, and `pyarrow`.

## Reconstructed pricing rule

For a total line `L`, current combined score `S`, and elapsed regulation second `t`, the differential is `L - S`. At the same second of prior games, the archived final score minus the score then gives each game's *remaining points*. The raw probability is the fraction whose remaining points exceed the differential. Overtime is included in the archived final score.

The audit's market adjustment is approximately:

`shift = (pregame market 50/50 total - historical median final total) * (2400 - t) / 2400`

The tool shifts the historical remaining-points distribution by that many points and interpolates fractional shifts between adjacent integer outcomes. Thus the market anchor has full effect at tip and zero effect at the end of regulation. `replica` uses earlier games from the same season, equally weighted. For the September 27 audit, its pregame probability for NY-MIN over 166.5 is **0.6494428**, exactly the logged adjusted theo.

`audit-check` verifies that all **1,342** logged raw theos exactly match the uploaded 2026 table. The interpolated adjusted theo matches **1,083/1,342** fills exactly; mean absolute difference is **0.00282** and the largest is **0.05162**. The original source code is absent, so the remaining interpolation differences cannot be resolved from the audit alone.

## Variants

| Model | Remaining-points sample | Extra adjustment |
| --- | --- | --- |
| `replica` | Prior games in the current season, equal weight | None |
| `team` | Same as replica | Previous five combined scores for each team, averaged; a past-only ridge regression on verified closing lines learns how much this signal changes totals relative to the market line |
| `pooled` | All prior years, 1.5-year weight half-life and 3x current-season weight | Same team adjustment |
| `playoff` | Same as pooled, with 3x weight for postseason games from the last five seasons | Same team adjustment, refit with postseason weights |

The team adjustment also decays linearly to zero over regulation. A negative fitted team coefficient is possible: recent high-scoring games can be followed by totals closer to the market line. The weights and five-game window are exploratory choices, not tuned on the reported test games.

## Walk-forward check

Run `python3 wnba_theo.py backtest`. Each 2024-26 test game uses only earlier dates, starts once 80 current-season games are available, and has an archived `explicit_close` total. The closing line stands in for the pregame market 50/50 total. We score at tip and after 10, 20, 30, and 35 regulation minutes. All scored lines end in `.5` to avoid pushes. There are **656** test games, including **45** postseason games. Smaller Brier loss is better.

| Model | Mean Brier over five states | Postseason games only |
| --- | ---: | ---: |
| `replica` | **0.16320** | **0.16343** |
| `team` | 0.16332 | 0.16498 |
| `pooled` | 0.16355 | 0.16550 |
| `playoff` | 0.16357 | 0.16528 |

The playoff weighting improves the pooled model slightly on postseason games, but neither new variant beats the replica in this test. Game-level bootstrap intervals for differences from the replica cross zero. The postseason sample is small, and a historical closing line may not have been priced at precisely 50/50.

## Example

```sh
python3 wnba_theo.py audit-check
python3 wnba_theo.py predict --date 2026-09-27 --season 2026 \
  --second 0 --score 0 --line 166.5 --market-median 173.33333333333334 \
  --home 'Minnesota Lynx' --away 'New York Liberty'
python3 -m unittest test_wnba_theo.py
```

`predict` accepts elapsed regulation seconds and the combined live score supplied by the caller. Team names must match `games.csv`. The upload ends on **September 24, 2026**, so subsequent games are absent from team form and historical samples. The model supports half-point lines and regulation only. The [Kalshi KXWNBATOTAL rules](https://kalshi.com/markets/kxwnbatotal/x/kxwnbatotal-26sep29lvind) explicitly include overtime; check the particular Polymarket US contract terms before applying this result there. These probability checks do not include trading fees, latency, or the exchange order book.
