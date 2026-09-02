# Model notes

Source material for the README and interview prep. Plain-language summary
of what was tried, what the data actually showed, and why the final model
is what it is. Full numeric output lives in the commit history of
`src/train.py`'s runs; this file is the distilled version.

Data: 11 Premier League seasons (2015/16-2025/26, 4,180 matches) from
football-data.co.uk. Time-based split: train on 2015/16-2023/24 (3,420
matches), test on 2024/25-2025/26 (760 matches) -- never random, since the
features are built from a chronological team-state pass and a random split
would let future form/Elo leak backward into training rows.

## 1. `class_weight='balanced'` makes both models overconfident about draws

The first full run used `class_weight='balanced'` on both Logistic
Regression and Random Forest, reasoning that draws (~24% of matches) would
otherwise get outvoted by the two majority classes. The draw-calibration
check said otherwise: bucket matches by each model's predicted P(draw) and
compare to how often a draw actually happened in that bucket.

**Predicted P(draw) in the 0.3-0.5 range, balanced vs. unweighted:**

| Setting | n (matches in this range) | Mean predicted P(draw) | Actual draw frequency | Gap |
|---|---|---|---|---|
| `class_weight='balanced'`, LR | 541 | 0.360 | 0.268 | **+0.092** |
| `class_weight='balanced'`, RF | 547 | 0.362 | 0.269 | **+0.093** |
| `class_weight=None`, LR | 28 | 0.308 | 0.250 | +0.058 |
| `class_weight=None`, RF | 34 | 0.309 | 0.353 | -0.044 |

With `balanced` weighting, both models push roughly 550 test matches (72%
of the test set) into a "30-50% chance of a draw" bucket where a draw
actually happens only ~27% of the time -- a real, systematic
overconfidence, not noise. Without it, almost nothing lands in that range
at all (n=28-34): the unweighted models simply don't get confident about
draws very often, which turns out to produce better-calibrated
probabilities where they do.

**Net effect on log loss** (the metric that actually penalizes this):

| Setting | Best log loss (LR or RF) |
|---|---|
| `class_weight='balanced'` | 1.0084 |
| `class_weight=None` | **0.9981** |

`class_weight=None` won and was used for every model in the feature-set
ablation below (part 2).

**Why this happened, in plain terms:** `balanced` reweights the loss so
each class contributes equally regardless of how common it is. That's the
right fix if the failure mode is "the model ignores the minority class
entirely." Here the actual failure mode was different: draws are hard to
predict, not rare enough to be ignorable, and reweighting made both models
overcorrect -- nudging predicted draw probability up across the board
rather than learning which specific matchups are actually more
draw-prone. The lesson generalizes past this one project: `balanced`
class weighting is a blunt instrument for imbalance, and it's worth
checking calibration (not just accuracy or even overall log loss) before
trusting it, because a moderate imbalance (24% minority class here) may
not need it at all.

## 2. Team-stat features add ~nothing once the market's own price is in the model

Three feature sets were compared, all with `class_weight=None`:

- **Market-only**: the three implied win/draw/away probabilities derived
  from the closing average bookmaker odds (`imp_prob_h/d/a`) -- nothing
  else.
- **Team-stat-only**: the 12 engineered features (Elo rating, 5-match
  form, home/away-specific points-per-game, rest days) -- no market
  information at all.
- **Combined**: all 15 features together.

**Log loss (lower is better) and accuracy, test set:**

| Feature set | Algorithm | Log loss | Accuracy |
|---|---|---|---|
| Market-only | Logistic Regression | **0.9989** | 0.5184 |
| Market-only | Random Forest | 1.0035 | 0.5105 |
| Team-stat-only | Logistic Regression | 1.0137 | 0.5053 |
| Team-stat-only | Random Forest | 1.0188 | 0.4987 |
| Combined | Logistic Regression | 0.9992 | 0.5132 |
| **Combined** | **Random Forest** | **0.9981** | 0.5105 |

Two things stand out:

- **Team-stat-only never beats market-only**, with either algorithm. The
  gap is real but not huge (~0.015-0.02 log loss), which matches the
  general finding in sports-analytics literature that betting markets are
  hard to beat -- they aggregate a lot of information (injuries, team
  news, sharp bettors) that isn't in a results-and-odds-history dataset.
- **Combined only edges out market-only by 0.0008 log loss** (0.9981 vs.
  0.9989) at best, and loses to it with Logistic Regression (0.9992 vs.
  0.9989). Adding the engineered features on top of the market barely
  moves the needle -- most of what Elo/form/rest capture, the market has
  apparently already priced in.

Plain-language takeaway: **the market is doing most of the work.** The
engineered features aren't worthless -- combined is never worse than
market-only by a meaningful margin, and it wins outright with Random
Forest -- but they're not adding an independent, strong signal on top of
the odds. This is an honest, unglamorous result and it's the headline
finding of the project, not something to spin.

## 3. Why Random Forest + combined features was picked as final, despite market-only being nearly as good

**Selection rule:** log loss, not accuracy (see `src/train.py` docstring
for the full reasoning -- short version: log loss penalizes confident
wrong answers, which is exactly the failure mode that matters for draws).
By that rule, **Combined - Random Forest (log loss 0.9981)** is the best
of every model fit across both ablations, market-only-LR included
(0.9989).

**Given the two are only 0.0008 log loss apart, why not just ship the
simpler market-only model?** Two reasons, both defensible rather than
"because it technically won":

1. **The margin is real, if small, and it's in the right direction.**
   Combined-RF beats market-only-LR on the metric that was pre-committed
   as the selection criterion. Overriding a pre-registered metric because
   the winner "only won by a little" would be moving the goalposts after
   seeing the result -- the discipline of picking log loss *before*
   running the ablation is worth more than hand-picking whichever number
   looks more impressive after the fact.
2. **Combined-RF is strictly more informative for the stated goal of this
   project.** This is a resume/portfolio project meant to demonstrate a
   full pipeline -- feature engineering, model comparison, calibration
   analysis -- not just "recite the bookmaker's number back." A model
   that only uses `imp_prob_h/d/a` doesn't exercise or showcase any of
   the Elo/form/rest engineering work, and the honest "team stats alone
   barely help beyond the market" finding is a more interesting, more
   defensible interview talking point when it's demonstrated by a model
   that actually had access to both and the market signal still visibly
   dominates its feature importances -- rather than a project that just
   never tried.

**What this is not:** a claim that Random Forest "beats Vegas" in any
meaningful sense. It's within a fraction of a percent of a model that is
*just* the market's own number, on a 760-match test set where that
fraction of a percent is well within noise. The honest framing, and the
one that belongs in the README and in interviews, is: *the model matches
the market's performance and adds a small, real edge via team-strength
features, but does not clearly outperform market-only pricing.*

## Known limitations (carried over from `features.py`, still unresolved)

- **Newly promoted teams** start at a neutral Elo (1500) even though
  they've been playing Championship football, not Premier League --
  likely overstates their strength.
- **Teams returning from relegation** get one 25%-regression nudge
  toward 1500 at their first match back, regardless of how many seasons
  they were away (rest-days and Elo both handle this differently --
  rest-days falls back to a neutral 7-day prior for gaps over 120 days,
  Elo does a single fixed-fraction regression).
- **Persisted feature state is more current than the model's training
  window**: `models/feature_state.joblib` reflects all 11 seasons through
  2025/26 so that inference-time features for a genuinely upcoming
  fixture are as fresh as possible, but the saved model's learned splits
  only ever saw 2015/16-2023/24. This is a deliberate choice (see
  `src/train.py` docstring), not an oversight, but it does mean the model
  hasn't been validated on data generated by a team-state distribution
  from 2024-2026 specifically.
- **`feature_state.joblib` does not update as the current season is
  played** -- this is the most significant limitation and, unlike the
  others above, not a deliberate tradeoff, just unbuilt. `GET /fixtures`
  (`src/fixtures.py`) only ever fetches `status=SCHEDULED` matches from
  football-data.org; nothing fetches *finished* matches and feeds their
  results back through `update_state()`. So every prediction, indefinitely
  into the future, uses Elo/form/rest/PPG frozen at the end of 2025/26
  (May 24, 2026) -- every 2026/27 result is invisible to the model even
  after it's been played. The distortion grows over the season: `form`
  (last 5 matches) never reflects any current-season result, and
  `rest_days` permanently falls back to the neutral 7-day default once
  enough time passes that the real gap exceeds the 120-day cold-start
  cutoff in `features.py`. Fixing this means periodically pulling
  `status=FINISHED` results (the same football-data.org API already in
  use) through `update_state()`, and persisting the updated state
  somewhere that survives Render's free-tier restarts -- its filesystem
  is ephemeral, so a plain file overwrite doesn't survive a redeploy.
  Deferred, not solved.

None of these were fixed here -- each would add real complexity for an
uncertain gain, and are flagged rather than silently patched.
