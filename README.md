# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Kazutor/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Kazutor/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Leonida Rarity Forecast predicts, for every GTA VI PlayStation 5 trophy, the share of Exophase-tracked players who will hold it at 2026-11-26T00:00:00Z, as five quantiles per trophy. It works on the logit scale against log-time since release: Exophase refreshes a game's counts in steps, so the history is first collapsed into plateaus of identical reads, then each trophy's logit rate is fitted with a robust Theil–Sen slope, shrunk toward its tier's mean slope, floored at a small positive growth, and extrapolated to the target with the move capped so a short-window trend cannot run away over 4.5 days. The spread comes from the fit residuals plus the refresh jumps seen in the history. When the history is completely flat — no refresh observed yet, as on launch night — the forecast is a two-state mixture: either no refresh lands before the target, or one does, in which case a big cohort of fresh players joins and dilutes every rate but the first trophy's; the mixture's quantiles are computed on a grid, with the refresh hazard rising the longer the page has sat unchanged.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=5v28zqbf&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
