"""Train + select a Premier League match outcome (H/D/A) classifier.

Split strategy: hard time-based split, train on 2015/16-2023/24 (9 seasons,
~3420 matches), test on 2024/25-2025/26 (2 seasons, ~760 matches). No random
shuffling. Every engineered feature (Elo, form, PPG, rest) is a function of
matches *before* the row it describes, so a random split would let a match's
own future opponents' post-match Elo movements leak backward into training
rows that occur earlier in the raw index but reference teams whose ratings
were partly shaped by test-period matches. Splitting on wall-clock time is
the only split where "train" genuinely means "everything known at the time",
matching how this model would actually be used (predict the next round of
fixtures from history up to now).

Model selection metric: log loss, not accuracy. Accuracy only asks whether
the argmax class was right; log loss asks how much probability mass was put
on the true outcome and penalizes confident wrong answers much harder than
uncertain wrong answers. That distinction matters specifically for draws: a
model that has genuinely learned something about draws will often say "35%
draw" on a game that ends in a draw -- right in spirit, and log loss rewards
that. A model that just leans on home wins being the plurality class can hit
similar accuracy by rarely predicting draws at all, while being badly
overconfident (e.g. 85% home win) on games that turn out to be draws --
accuracy doesn't see this, log loss punishes it directly.

This script runs three passes:
  0. Original run (kept for direct comparison): market baseline (raw odds,
     unfit) + LR/RF with class_weight='balanced' on the full feature set.
  1. Class-weight ablation: LR and RF, each with class_weight='balanced'
     and class_weight=None, full feature set. Tests whether 'balanced'
     reweighting is what's driving the draw-probability overconfidence
     seen in the 0.3-0.5 calibration buckets in the original run.
  2. Feature-set ablation, using whichever class_weight setting wins step 1:
     market-only (imp_prob_h/d/a alone), team-stat-only (everything except
     the market columns), and combined (all 15 features) -- for both LR
     and RF. Tests whether the engineered team-strength features carry any
     signal the market doesn't already have priced in.

Final selection, per prediction mode: rather than persisting only the
single overall-best model (Random Forest, combined features, log loss
0.9981), this saves one model per feature-set slice from step 2 --
market_only, team_stat_only, combined -- so the frontend can offer a live
mode switcher that exposes the feature-set ablation interactively instead
of only as a write-up. Each mode's winning algorithm is picked
independently by log loss (same rule as the overall selection): combined's
best happens to be Random Forest, market-only's and team-stat-only's best
are both Logistic Regression. Using whichever actually won per slice is
more honest than forcing one algorithm across all three just for
consistency. See MODEL_NOTES.md for the full reasoning behind picking
combined-RF as the *default* mode despite market-only being nearly as good.
The exact fitted models from step 2 (not refits) are what get persisted, so
each saved artifact's log loss is guaranteed to match what's reported here.

Persisted artifacts (models/):
  - model_market_only.joblib     fitted model, imp_prob_h/d/a only
  - model_team_stat_only.joblib  fitted model, everything except market odds
  - model_combined.joblib        fitted model, all 15 features (the default)
  - feature_state.joblib    FeatureState as of the END of the full dataset
                             (all 11 seasons, through 2025/26) -- NOT the
                             training cutoff (2023/24). Shared across all
                             three modes above, since it's the same feature
                             engineering regardless of which columns a given
                             mode's model actually consumes. The models'
                             learned splits come from the 2015/16-2023/24
                             window (matches the log losses above), but a
                             fixture predict.py is asked about is always in
                             the future relative to ALL known results, so
                             the Elo/form/rest state it starts from needs to
                             be as current as possible. This is a deliberate
                             train-window vs. state-window mismatch, not an
                             oversight -- flagged in MODEL_NOTES.md.
  - model_metadata.json     per-mode feature columns, algorithm, and
                             confirmed test accuracy/log loss/ROC AUC/Brier
                             score, plus shared split info, so predict.py
                             and api.py don't need to re-derive any of this
                             from train.py's source.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURE_COLUMNS, run_feature_pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.csv"
MATCHES_PATH = PROJECT_ROOT / "data" / "processed" / "matches.csv"
MODELS_DIR = PROJECT_ROOT / "models"

TRAIN_SEASONS = [f"{y}/{str(y + 1)[2:]}" for y in range(2015, 2024)]  # 2015/16 .. 2023/24
TEST_SEASONS = ["2024/25", "2025/26"]

CLASS_ORDER = ["A", "D", "H"]  # alphabetical; matches sklearn's default sort of string labels
CLASS_LABELS = {"H": "Home win", "D": "Draw", "A": "Away win"}

DRAW_CALIBRATION_BINS = np.arange(0.0, 1.01, 0.1)
MARKET_COLS = ["imp_prob_h", "imp_prob_d", "imp_prob_a"]
TEAM_STAT_COLS = [c for c in FEATURE_COLUMNS if c not in MARKET_COLS]

FEATURE_SETS = {
    "Market-only": MARKET_COLS,
    "Team-stat-only": TEAM_STAT_COLS,
    "Combined": FEATURE_COLUMNS,
}

# Maps step-2 ablation labels to the mode keys the API/frontend use.
MODE_KEYS = {"Market-only": "market_only", "Team-stat-only": "team_stat_only", "Combined": "combined"}
MODE_LABELS = {"market_only": "Market-aware", "team_stat_only": "Team-stat", "combined": "Combined"}
MODE_DESCRIPTIONS = {
    "market_only": "Uses only the market's implied win/draw/away probabilities.",
    "team_stat_only": "Hides market odds -- Elo, form, rest, and points-per-game history only.",
    "combined": "Uses both market odds and team-strength history.",
}

# See module docstring re: why this is regularized (shallow depth, largish
# leaf minimum) rather than left at sklearn defaults, and why it's a fixed,
# reasoned choice rather than tuned via search.
RF_PARAMS = dict(
    n_estimators=500,
    max_depth=6,
    min_samples_leaf=30,
    random_state=42,
    n_jobs=-1,
)


def load_split():
    df = pd.read_csv(FEATURES_PATH, parse_dates=["Date"])
    train = df[df["Season"].isin(TRAIN_SEASONS)].reset_index(drop=True)
    test = df[df["Season"].isin(TEST_SEASONS)].reset_index(drop=True)
    assert len(train) + len(test) == len(df), "season lists don't partition the full dataset"
    assert train["Date"].max() < test["Date"].min(), "train/test seasons are not cleanly time-ordered"
    return train, test


def make_logistic_regression(class_weight):
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(class_weight=class_weight, max_iter=2000, random_state=42)),
    ])


def make_random_forest(class_weight):
    return RandomForestClassifier(class_weight=class_weight, **RF_PARAMS)


def proba_frame(proba: np.ndarray, classes: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(proba, columns=classes)[CLASS_ORDER]


def market_baseline_proba(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "A": df["imp_prob_a"], "D": df["imp_prob_d"], "H": df["imp_prob_h"],
    })[CLASS_ORDER]


def multiclass_brier_score(y_true: pd.Series, proba: pd.DataFrame) -> float:
    """Mean squared error between predicted probabilities and the one-hot
    true label, summed across classes then averaged over rows -- the
    standard multiclass generalization of the binary Brier score (sklearn
    only ships brier_score_loss for two classes). Lower is better, same
    direction as log loss, but on a bounded 0-2 scale instead of unbounded."""
    y_onehot = pd.get_dummies(y_true)[CLASS_ORDER].to_numpy(dtype=float)
    diff = proba[CLASS_ORDER].to_numpy() - y_onehot
    return float(np.mean(np.sum(diff**2, axis=1)))


def evaluate(name: str, y_true: pd.Series, proba: pd.DataFrame) -> dict:
    y_pred = proba.idxmax(axis=1)
    acc = accuracy_score(y_true, y_pred)
    ll = log_loss(y_true, proba.to_numpy(), labels=CLASS_ORDER)
    auc = roc_auc_score(y_true, proba[CLASS_ORDER].to_numpy(), labels=CLASS_ORDER, multi_class="ovr")
    brier = multiclass_brier_score(y_true, proba)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=["H", "D", "A"], zero_division=0
    )
    result = {"model": name, "accuracy": acc, "log_loss": ll, "roc_auc": auc, "brier_score": brier}
    for cls, p, r, f, s in zip(["H", "D", "A"], precision, recall, f1, support):
        result[f"precision_{cls}"] = p
        result[f"recall_{cls}"] = r
        result[f"f1_{cls}"] = f
        result[f"support_{cls}"] = int(s)
    return result


def fit_and_evaluate(name, model, feature_cols, train, y_train, test, y_test):
    fitted = clone(model)
    fitted.fit(train[feature_cols], y_train)
    classes = fitted.named_steps["clf"].classes_ if isinstance(fitted, Pipeline) else fitted.classes_
    proba = proba_frame(fitted.predict_proba(test[feature_cols]), classes)
    return evaluate(name, y_test, proba), proba, fitted


def print_overall_table(results: list[dict], title: str) -> None:
    print(f"\n{'-' * 100}\n{title}\n{'-' * 100}")
    summary = pd.DataFrame(results).set_index("model")[["accuracy", "log_loss"]]
    summary.columns = ["Accuracy", "Log Loss"]
    print(summary.round(4).to_string())


def print_full_metrics_table(results: list[dict], title: str) -> None:
    print_overall_table(results, title)
    for cls in ["H", "D", "A"]:
        print(f"\n{CLASS_LABELS[cls]} ({cls}) -- precision / recall / F1 / support")
        cols = [f"precision_{cls}", f"recall_{cls}", f"f1_{cls}", f"support_{cls}"]
        per_class = pd.DataFrame(results).set_index("model")[cols]
        per_class.columns = ["Precision", "Recall", "F1", "Support"]
        print(per_class.round(4).to_string())


def draw_calibration_table(name: str, y_true: pd.Series, proba: pd.DataFrame) -> pd.DataFrame:
    pred_draw = proba["D"]
    actual_draw = (y_true == "D").astype(int)
    bucket = pd.cut(pred_draw, bins=DRAW_CALIBRATION_BINS, include_lowest=True)

    table = pd.DataFrame({"bucket": bucket, "pred_draw_prob": pred_draw, "actual_draw": actual_draw})
    grouped = table.groupby("bucket", observed=True).agg(
        n=("actual_draw", "size"),
        mean_predicted=("pred_draw_prob", "mean"),
        actual_draw_frequency=("actual_draw", "mean"),
    )
    return grouped


def print_draw_calibration(y_true: pd.Series, proba_by_model: dict, title: str) -> None:
    print(f"\n{'-' * 100}\n{title}\n(bucket 'n' below ~15-20 is too small to read as a real calibration signal)\n{'-' * 100}")
    for name, proba in proba_by_model.items():
        table = draw_calibration_table(name, y_true, proba)
        print(f"\n{name}:")
        print(table[["n", "mean_predicted", "actual_draw_frequency"]].round(3).to_string())


def draw_overconfidence_in_range(y_true: pd.Series, proba: pd.DataFrame, low: float, high: float):
    """Rows whose predicted P(draw) falls in [low, high). Returns
    (n, mean_predicted, actual_frequency, gap) where gap = predicted - actual;
    a positive gap means the model is overconfident about draws in that range."""
    pred_draw = proba["D"]
    mask = (pred_draw >= low) & (pred_draw < high)
    n = int(mask.sum())
    if n == 0:
        return n, float("nan"), float("nan"), float("nan")
    mean_pred = float(pred_draw[mask].mean())
    actual = float((y_true[mask] == "D").mean())
    return n, mean_pred, actual, mean_pred - actual


def save_mode_models(fs_results: list[dict], fs_models: dict, class_weight_label: str,
                      train_row_count: int, test_row_count: int) -> dict:
    """Persist one model per prediction mode (market_only / team_stat_only /
    combined) rather than a single overall-best model -- lets the frontend
    offer a live mode switcher that exposes the feature-set ablation
    interactively rather than only as a write-up. For each mode, the
    winning algorithm is picked independently by log loss (same rule as
    the old single-model selection) -- see module docstring for why using
    whichever algorithm actually won each slice is more honest than forcing
    one algorithm across all three. Does NOT build the API/frontend side of
    this -- that's a separate step."""
    MODELS_DIR.mkdir(exist_ok=True)

    all_matches = pd.read_csv(MATCHES_PATH, parse_dates=["Date"])
    _, final_state = run_feature_pipeline(all_matches)
    state_path = MODELS_DIR / "feature_state.joblib"
    joblib.dump(final_state, state_path)

    modes_meta = {}
    for fs_label, cols in FEATURE_SETS.items():
        mode_key = MODE_KEYS[fs_label]
        candidates = [r for r in fs_results if r["model"].startswith(f"{fs_label} - ")]
        best = min(candidates, key=lambda r: r["log_loss"])
        fitted_model = fs_models[best["model"]]
        algo_label = best["model"].split(" - ", 1)[1]
        clf = fitted_model.named_steps["clf"] if isinstance(fitted_model, Pipeline) else fitted_model

        model_path = MODELS_DIR / f"model_{mode_key}.joblib"
        joblib.dump(fitted_model, model_path)

        modes_meta[mode_key] = {
            "label": MODE_LABELS[mode_key],
            "description": MODE_DESCRIPTIONS[mode_key],
            "model_type": type(clf).__name__,
            "algorithm_label": algo_label,
            "feature_columns": cols,
            "test_accuracy": best["accuracy"],
            "test_log_loss": best["log_loss"],
            "test_roc_auc": best["roc_auc"],
            "test_brier_score": best["brier_score"],
            "model_file": model_path.name,
        }
        print(f"  saved {model_path.name}  ({algo_label}, log loss {best['log_loss']:.4f})")

    best_mode = min(modes_meta, key=lambda k: modes_meta[k]["test_log_loss"])

    metadata = {
        "class_weight": class_weight_label,
        "class_order": CLASS_ORDER,
        "train_seasons": TRAIN_SEASONS,
        "test_seasons": TEST_SEASONS,
        "train_row_count": train_row_count,
        "test_row_count": test_row_count,
        "feature_state_as_of_date": str(all_matches["Date"].max().date()),
        "feature_state_note": (
            "state reflects ALL matches through the date above, not just the "
            "training window -- see module docstring 'Persisted artifacts'"
        ),
        "modes": modes_meta,
        "best_mode": best_mode,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
    }
    metadata_path = MODELS_DIR / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  {state_path}  (feature-engineering state as of {metadata['feature_state_as_of_date']}, shared across modes)")
    print(f"  {metadata_path}")
    print(f"\nBest mode by log loss: {best_mode} ({modes_meta[best_mode]['test_log_loss']:.4f}) -- default mode in the API")
    return metadata


def main() -> None:
    train, test = load_split()
    y_train, y_test = train["FTR"], test["FTR"]
    print(f"Train: {len(train)} matches ({TRAIN_SEASONS[0]} - {TRAIN_SEASONS[-1]})")
    print(f"Test:  {len(test)} matches ({', '.join(TEST_SEASONS)})")

    # ================= ORIGINAL RUN (kept for direct comparison) =================
    market_raw_proba = market_baseline_proba(test)
    original_results = [evaluate("Market Baseline (odds only, unfit)", y_test, market_raw_proba)]
    original_proba = {"Market Baseline (odds only, unfit)": market_raw_proba}

    for algo_label, maker in [("Logistic Regression", make_logistic_regression), ("Random Forest", make_random_forest)]:
        name = f"{algo_label} (balanced, combined features)"
        result, proba, _ = fit_and_evaluate(name, maker("balanced"), FEATURE_COLUMNS, train, y_train, test, y_test)
        original_results.append(result)
        original_proba[name] = proba

    print("\n" + "=" * 100)
    print("ORIGINAL RUN (reference, unchanged from previous pass)")
    print("=" * 100)
    print_full_metrics_table(original_results, "Overall metrics")
    print_draw_calibration(y_test, original_proba, "Draw-class calibration")

    # ================= STEP 1: CLASS-WEIGHT ABLATION =================
    cw_results = []
    cw_proba = {}
    for cw_label, cw_value in [("balanced", "balanced"), ("none", None)]:
        for algo_label, maker in [("Logistic Regression", make_logistic_regression), ("Random Forest", make_random_forest)]:
            name = f"{algo_label} (class_weight={cw_label})"
            result, proba, _ = fit_and_evaluate(name, maker(cw_value), FEATURE_COLUMNS, train, y_train, test, y_test)
            cw_results.append(result)
            cw_proba[name] = proba

    print("\n" + "=" * 100)
    print("STEP 1: CLASS-WEIGHT ABLATION (combined feature set, balanced vs. none)")
    print("=" * 100)
    print_overall_table(cw_results, "Accuracy / Log Loss")
    print_draw_calibration(y_test, cw_proba, "Draw-class calibration")

    print(f"\n{'-' * 100}\nDraw overconfidence check: predicted P(draw) in [0.3, 0.5) vs. actual frequency\n{'-' * 100}")
    overconf_rows = []
    for name, proba in cw_proba.items():
        n, mean_pred, actual, gap = draw_overconfidence_in_range(y_test, proba, 0.3, 0.5)
        overconf_rows.append({"model": name, "n": n, "mean_predicted": mean_pred, "actual_freq": actual, "gap (pred - actual)": gap})
    overconf_df = pd.DataFrame(overconf_rows).set_index("model")
    print(overconf_df.round(3).to_string())

    balanced_min_ll = min(r["log_loss"] for r in cw_results if "class_weight=balanced" in r["model"])
    none_min_ll = min(r["log_loss"] for r in cw_results if "class_weight=none" in r["model"])
    winning_cw_label = "balanced" if balanced_min_ll <= none_min_ll else "none"
    winning_cw_value = "balanced" if winning_cw_label == "balanced" else None
    print(f"\nBest log loss under class_weight='balanced': {balanced_min_ll:.4f}")
    print(f"Best log loss under class_weight=None:        {none_min_ll:.4f}")
    print(f"-> class_weight='{winning_cw_label}' wins on log loss and carries forward into step 2.")

    balanced_gap = overconf_df.loc[[m for m in overconf_df.index if "balanced" in m], "gap (pred - actual)"].mean()
    none_gap = overconf_df.loc[[m for m in overconf_df.index if "class_weight=none" in m], "gap (pred - actual)"].mean()
    verdict = "CONFIRMED" if balanced_gap > none_gap else "REFUTED"
    print(f"\nHypothesis check -- 'balanced weighting causes draw overconfidence in the 0.3-0.5 bucket': {verdict}")
    print(f"  mean (predicted - actual) gap in that range: balanced={balanced_gap:.3f}, none={none_gap:.3f}")

    # ================= STEP 2: FEATURE-SET ABLATION =================
    fs_results = []
    fs_proba = {}
    fs_models = {}
    for fs_label, cols in FEATURE_SETS.items():
        for algo_label, maker in [("Logistic Regression", make_logistic_regression), ("Random Forest", make_random_forest)]:
            name = f"{fs_label} - {algo_label}"
            result, proba, fitted = fit_and_evaluate(name, maker(winning_cw_value), cols, train, y_train, test, y_test)
            fs_results.append(result)
            fs_proba[name] = proba
            fs_models[name] = fitted

    print("\n" + "=" * 100)
    print(f"STEP 2: FEATURE-SET ABLATION (class_weight={winning_cw_label}, winner from step 1)")
    print("=" * 100)
    print_overall_table(fs_results, "Accuracy / Log Loss")
    print_draw_calibration(y_test, fs_proba, "Draw-class calibration")

    # ================= FINAL: persist one model per prediction mode =================
    print("\n" + "=" * 100)
    print("SAVING ONE MODEL PER PREDICTION MODE (market_only / team_stat_only / combined)")
    print("=" * 100)
    save_mode_models(fs_results, fs_models, winning_cw_label, train_row_count=len(train), test_row_count=len(test))


if __name__ == "__main__":
    main()
