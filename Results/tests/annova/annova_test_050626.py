"""
three_way_anova.py
═══════════════════════════════════════════════════════════════════════════════
THREE-WAY ANOVA: Platform × Prompt × Difficulty → Solve Rate

Because the outcome (pass/fail) is binary, a standard OLS ANOVA is run on the
binary outcome as a linear probability model — a well-established approach for
factorial designs with binary outcomes when cell sizes are reasonable.
We additionally run a logistic regression for robustness, and report
effect sizes (eta-squared, partial eta-squared) for every factor and
interaction term.

Factors:
  A = Difficulty  (Easy / Medium / Hard)
  B = Prompt      (P1 / P2 / P3)
  C = Platform    (AtCoder / Codeforces / LeetCode)

Outcome: full_pass = PublicPassed AND HiddenPassed (binary 0/1)

Pipeline:
  1. Merge phase1 + rerun_pass1 + rerun_pass2 into canonical pass/fail matrix
  2. Join platform from problems_dataset_150x3.csv
  3. Full factorial 3-way ANOVA (Type III SS) via statsmodels OLS
  4. Logistic regression with same terms (robustness check)
  5. Pairwise post-hoc tests (Tukey HSD) for each significant main effect
  6. Interaction plots data (saved to CSV for plotting)
  7. Effect sizes: eta² and partial eta² for every term
  8. Assumptions checks: Levene's test, residual normality (Shapiro-Wilk sample)

Outputs:
  anova_3way_results.csv         — full ANOVA table with effect sizes
  anova_logistic_results.csv     — logistic regression coefficients + OR
  anova_posthoc_difficulty.csv   — Tukey HSD for Difficulty
  anova_posthoc_prompt.csv       — Tukey HSD for Prompt
  anova_posthoc_platform.csv     — Tukey HSD for Platform
  anova_cell_means.csv           — cell means for all factor combinations
  anova_assumptions.csv          — Levene + Shapiro results
═══════════════════════════════════════════════════════════════════════════════
"""

import os, warnings
import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations

import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
PHASE1_CSV   = "phase1_llama70b-2-2204-metav2150_450x3.csv"
RERUN_P1_CSV = "rerun_pass1_results.csv"
RERUN_P2_CSV = "rerun_pass2_results.csv"
DATASET_CSV  = "problems_dataset_150x3.csv"

OUT_ANOVA      = "anova_3way_results.csv"
OUT_LOGISTIC   = "anova_logistic_results.csv"
OUT_PH_DIFF    = "anova_posthoc_difficulty.csv"
OUT_PH_PROMPT  = "anova_posthoc_prompt.csv"
OUT_PH_PLAT    = "anova_posthoc_platform.csv"
OUT_CELL_MEANS = "anova_cell_means.csv"
OUT_ASSUMPTIONS= "anova_assumptions.csv"

# ── STEP 1: Build canonical pass/fail dataframe ───────────────────────────────
def load_phase1(path):
    df = pd.read_csv(path)
    df["ProblemID"] = df["ProblemID"].astype(str)
    results = {}
    for _, row in df.iterrows():
        pid, strat = str(row["ProblemID"]), str(row["Prompt"])
        pub = int(row.get("PublicPassed", 0) or 0)
        hid = int(row.get("HiddenPassed", 0) or 0)
        results[(pid, strat)] = {
            "pub": pub, "hid": hid,
            "pass": int(pub == 1 and hid == 1),
            "difficulty": str(row.get("Difficulty", "")),
            "problem_name": str(row.get("ProblemName", "")),
        }
    print(f"  Phase1   : {len(df)} rows from {path}")
    return results


def apply_rerun_p1(results, path):
    if not os.path.exists(path):
        return results
    df = pd.read_csv(path)
    df["ProblemID"] = df["ProblemID"].astype(str)
    n = 0
    for _, row in df.iterrows():
        if str(row.get("Pass1_Status", "skipped")) == "skipped":
            continue
        pid, strat = str(row["ProblemID"]), str(row["Prompt"])
        pub = int(row.get("Pass1_PublicPassed", 0) or 0)
        hid = int(row.get("Pass1_HiddenPassed", 0) or 0)
        key = (pid, strat)
        if key in results:
            results[key].update({"pub": pub, "hid": hid,
                                  "pass": int(pub == 1 and hid == 1)})
            n += 1
    print(f"  rerun_p1 : {n} overrides from {path}")
    return results


def apply_rerun_p2(results, path):
    if not os.path.exists(path):
        return results
    df = pd.read_csv(path)
    df["ProblemID"] = df["ProblemID"].astype(str)
    n = 0
    for _, row in df.iterrows():
        pid, strat = str(row["ProblemID"]), str(row["Prompt"])
        pub = int(row.get("Pass2_PublicPassed", 0) or 0)
        hid = int(row.get("Pass2_HiddenPassed", 0) or 0)
        key = (pid, strat)
        if key in results:
            results[key].update({"pub": pub, "hid": hid,
                                  "pass": int(pub == 1 and hid == 1)})
            n += 1
    print(f"  rerun_p2 : {n} overrides from {path}")
    return results


def build_df(results, dataset_path):
    # Load platform map
    plat_map = {}
    if os.path.exists(dataset_path):
        ds = pd.read_csv(dataset_path)
        ds["ProblemID"] = ds["ProblemID"].astype(str)
        for _, r in ds.iterrows():
            pid = str(r["ProblemID"])
            raw = str(r.get("Platform", "unknown")).strip().title()
            plat_map[pid] = raw
    else:
        print(f"  WARNING: {dataset_path} not found — platform will be 'unknown'")

    rows = []
    for (pid, strat), v in results.items():
        rows.append({
            "ProblemID":  pid,
            "Prompt":     strat,
            "Difficulty": v.get("difficulty", ""),
            "Platform":   plat_map.get(pid, "unknown"),
            "full_pass":  v["pass"],
        })
    df = pd.DataFrame(rows)

    # Encode ordered difficulty
    diff_order = {"Easy": 0, "Medium": 1, "Hard": 2}
    df["Difficulty_num"] = df["Difficulty"].map(diff_order)

    print(f"  Dataset  : {len(df)} rows | "
          f"platforms: {df['Platform'].value_counts().to_dict()}")
    return df


# ── STEP 2: Three-way ANOVA (Type III SS, OLS) ───────────────────────────────
def run_anova(df):
    """
    Full factorial model:
      full_pass ~ C(Difficulty) + C(Prompt) + C(Platform)
                + C(Difficulty):C(Prompt)
                + C(Difficulty):C(Platform)
                + C(Prompt):C(Platform)
                + C(Difficulty):C(Prompt):C(Platform)

    Type III SS used so each term is tested controlling for all others.
    Reference cells: Difficulty=Easy, Prompt=P1, Platform=Atcoder
    (alphabetically first, statsmodels default).
    """
    formula = (
        "full_pass ~ C(Difficulty, Treatment('Easy')) "
        "+ C(Prompt, Treatment('P1')) "
        "+ C(Platform, Treatment('Atcoder')) "
        "+ C(Difficulty, Treatment('Easy')):C(Prompt, Treatment('P1')) "
        "+ C(Difficulty, Treatment('Easy')):C(Platform, Treatment('Atcoder')) "
        "+ C(Prompt, Treatment('P1')):C(Platform, Treatment('Atcoder')) "
        "+ C(Difficulty, Treatment('Easy')):C(Prompt, Treatment('P1')):C(Platform, Treatment('Atcoder'))"
    )

    model = smf.ols(formula, data=df).fit()
    anova_table = anova_lm(model, typ=3)

    # ── Effect sizes ─────────────────────────────────────────────────────────
    # eta² = SS_effect / SS_total
    # partial eta² = SS_effect / (SS_effect + SS_residual)
    ss_total    = anova_table["sum_sq"].sum()
    ss_residual = anova_table.loc["Residual", "sum_sq"]

    anova_table["eta_sq"]         = anova_table["sum_sq"] / ss_total
    anova_table["partial_eta_sq"] = (
        anova_table["sum_sq"] / (anova_table["sum_sq"] + ss_residual)
    )
    # Residual has no eta² interpretation
    anova_table.loc["Residual", ["eta_sq", "partial_eta_sq"]] = np.nan

    # Clean up term names for readability
    rename = {
        "C(Difficulty, Treatment('Easy'))":                      "Difficulty",
        "C(Prompt, Treatment('P1'))":                            "Prompt",
        "C(Platform, Treatment('Atcoder'))":                     "Platform",
        "C(Difficulty, Treatment('Easy')):C(Prompt, Treatment('P1'))":
                                                                 "Difficulty × Prompt",
        "C(Difficulty, Treatment('Easy')):C(Platform, Treatment('Atcoder'))":
                                                                 "Difficulty × Platform",
        "C(Prompt, Treatment('P1')):C(Platform, Treatment('Atcoder'))":
                                                                 "Prompt × Platform",
        "C(Difficulty, Treatment('Easy')):C(Prompt, Treatment('P1')):C(Platform, Treatment('Atcoder'))":
                                                                 "Difficulty × Prompt × Platform",
        "Intercept": "Intercept",
        "Residual":  "Residual",
    }
    anova_table.index = [rename.get(i, i) for i in anova_table.index]

    return anova_table, model


# ── STEP 3: Logistic regression (robustness check) ───────────────────────────
def run_logistic(df):
    formula = (
        "full_pass ~ C(Difficulty, Treatment('Easy')) "
        "+ C(Prompt, Treatment('P1')) "
        "+ C(Platform, Treatment('Atcoder')) "
        "+ C(Difficulty, Treatment('Easy')):C(Prompt, Treatment('P1')) "
        "+ C(Difficulty, Treatment('Easy')):C(Platform, Treatment('Atcoder')) "
        "+ C(Prompt, Treatment('P1')):C(Platform, Treatment('Atcoder'))"
    )
    logit = smf.logit(formula, data=df).fit(disp=0)
    summary = logit.summary2().tables[1].copy()
    summary["OR"]        = np.exp(summary["Coef."])
    summary["OR_CI_low"] = np.exp(summary["Coef."] - 1.96 * summary["Std.Err."])
    summary["OR_CI_hi"]  = np.exp(summary["Coef."] + 1.96 * summary["Std.Err."])
    summary = summary.round(4)
    return summary, logit


# ── STEP 4: Post-hoc Tukey HSD ───────────────────────────────────────────────
def posthoc_tukey(df, factor_col):
    tukey = pairwise_tukeyhsd(
        endog=df["full_pass"],
        groups=df[factor_col],
        alpha=0.05
    )
    ph_df = pd.DataFrame(
        data=tukey._results_table.data[1:],
        columns=tukey._results_table.data[0]
    )
    # Add mean solve rate per group for context
    means = df.groupby(factor_col)["full_pass"].mean().round(4).to_dict()
    ph_df["group1_mean"] = ph_df["group1"].map(means)
    ph_df["group2_mean"] = ph_df["group2"].map(means)
    ph_df["mean_diff_pp"] = ((ph_df["group1_mean"] - ph_df["group2_mean"]) * 100).round(2)
    return ph_df


# ── STEP 5: Cell means ───────────────────────────────────────────────────────
def cell_means(df):
    # All combinations: Difficulty × Prompt × Platform
    full = (
        df.groupby(["Difficulty", "Prompt", "Platform"])
        .agg(
            n           = ("full_pass", "count"),
            solve_rate  = ("full_pass", "mean"),
            n_solved    = ("full_pass", "sum"),
        )
        .reset_index()
    )
    full["solve_rate"] = full["solve_rate"].round(4)
    full["solve_pct"]  = (full["solve_rate"] * 100).round(1)
    return full


# ── STEP 6: Assumption checks ─────────────────────────────────────────────────
def check_assumptions(df, model):
    rows = []

    # Levene's test: homogeneity of variance across each factor's groups
    for factor in ["Difficulty", "Prompt", "Platform"]:
        groups = [g["full_pass"].values for _, g in df.groupby(factor)]
        stat, p = stats.levene(*groups)
        rows.append({
            "test":      f"Levene's ({factor})",
            "statistic": round(stat, 4),
            "p_value":   round(p, 6),
            "pass_p05":  p >= 0.05,
            "note": "Homogeneity of variance — want p≥0.05 for ANOVA assumption"
        })

    # Shapiro-Wilk on residuals (sample 500 if large)
    resids = model.resid.values
    sample = resids if len(resids) <= 500 else np.random.choice(resids, 500, replace=False)
    sw_stat, sw_p = stats.shapiro(sample)
    rows.append({
        "test":      "Shapiro-Wilk (residuals)",
        "statistic": round(sw_stat, 6),
        "p_value":   round(sw_p, 6),
        "pass_p05":  sw_p >= 0.05,
        "note": (
            "Normality of residuals — binary outcome will fail this; "
            "see logistic regression for robust inference"
        )
    })

    # Kruskal-Wallis per factor (non-parametric backup)
    for factor in ["Difficulty", "Prompt", "Platform"]:
        groups = [g["full_pass"].values for _, g in df.groupby(factor)]
        kw_stat, kw_p = stats.kruskal(*groups)
        rows.append({
            "test":      f"Kruskal-Wallis ({factor})",
            "statistic": round(kw_stat, 4),
            "p_value":   round(kw_p, 6),
            "pass_p05":  None,
            "note": "Non-parametric backup for main effect — no normality assumption"
        })

    return pd.DataFrame(rows)


# ── STEP 7: Print report ──────────────────────────────────────────────────────
def print_report(anova_table, logit_summary, df):
    print("\n" + "="*72)
    print("  THREE-WAY ANOVA: Difficulty × Prompt × Platform → Solve Rate")
    print("="*72)

    print("\n── ANOVA TABLE (Type III SS, OLS linear probability model) ──────────")
    print(f"{'Term':<34} {'df':>4} {'F':>9} {'p-value':>10} {'η²':>8} {'η²p':>8}  sig")
    print("─"*72)
    sig_terms = []
    for term, row in anova_table.iterrows():
        if term in ("Intercept", "Residual"):
            continue
        f   = row.get("F", np.nan)
        p   = row.get("PR(>F)", np.nan)
        eta = row.get("eta_sq", np.nan)
        pet = row.get("partial_eta_sq", np.nan)
        df_ = row.get("df", np.nan)
        if pd.isna(p):
            continue
        stars = ("***" if p < 0.001 else
                 "**"  if p < 0.01  else
                 "*"   if p < 0.05  else "ns")
        if p < 0.05:
            sig_terms.append(term)
        print(f"  {term:<32} {int(df_):>4} {f:>9.3f} {p:>10.4f} "
              f"{eta:>7.4f} {pet:>7.4f}  {stars}")
    print("─"*72)
    print("  η² = eta-squared (variance explained out of total)")
    print("  η²p = partial eta-squared (effect controlling for other terms)")
    print("  *** p<.001  ** p<.01  * p<.05  ns = not significant")

    # ── Cell means by each factor ─────────────────────────────────────────
    print("\n── MARGINAL MEANS ────────────────────────────────────────────────────")
    for factor in ["Difficulty", "Prompt", "Platform"]:
        print(f"\n  By {factor}:")
        m = df.groupby(factor)["full_pass"].agg(["mean","count","sum"])
        m.columns = ["solve_rate","n_pairs","n_solved"]
        m["solve_pct"] = (m["solve_rate"]*100).round(1)
        for g, r in m.iterrows():
            print(f"    {str(g):<14} {r['n_solved']:>4.0f}/{r['n_pairs']:>4.0f} "
                  f"= {r['solve_pct']:>5.1f}%")

    # ── Significant findings ───────────────────────────────────────────────
    print("\n── SIGNIFICANT EFFECTS ───────────────────────────────────────────────")
    if sig_terms:
        for t in sig_terms:
            print(f"  ✅ {t}")
    else:
        print("  No terms reached p < 0.05")

    # ── Interaction summary ────────────────────────────────────────────────
    print("\n── KEY INTERACTIONS (cell means Difficulty × Platform) ──────────────")
    cross = df.groupby(["Difficulty","Platform"])["full_pass"].mean().unstack()
    cross = (cross * 100).round(1)
    print(cross.to_string())

    print("\n── KEY INTERACTIONS (cell means Difficulty × Prompt) ────────────────")
    cross2 = df.groupby(["Difficulty","Prompt"])["full_pass"].mean().unstack()
    cross2 = (cross2 * 100).round(1)
    print(cross2.to_string())

    print("\n── KEY INTERACTIONS (cell means Prompt × Platform) ──────────────────")
    cross3 = df.groupby(["Prompt","Platform"])["full_pass"].mean().unstack()
    cross3 = (cross3 * 100).round(1)
    print(cross3.to_string())

    print("\n" + "="*72)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*72)
    print("  THREE-WAY ANOVA PIPELINE")
    print("  Factors: Difficulty × Prompt × Platform")
    print("  Outcome: full_pass (PublicPassed AND HiddenPassed = 1)")
    print("="*72 + "\n")

    # ── Load data ────────────────────────────────────────────────────────────
    print("Loading and merging results...")
    results = load_phase1(PHASE1_CSV)
    results = apply_rerun_p1(results, RERUN_P1_CSV)
    results = apply_rerun_p2(results, RERUN_P2_CSV)
    df = build_df(results, DATASET_CSV)

    # ── Filter to known platforms ────────────────────────────────────────────
    df = df[df["Platform"] != "unknown"].copy()
    print(f"\n  Final dataset for ANOVA: {len(df)} observations")
    print(f"  Cells (Difficulty × Prompt × Platform): "
          f"{df.groupby(['Difficulty','Prompt','Platform']).ngroups}")

    # ── Check cell sizes ─────────────────────────────────────────────────────
    cell_sizes = df.groupby(["Difficulty","Prompt","Platform"]).size()
    print(f"\n  Cell sizes — min: {cell_sizes.min()}, "
          f"max: {cell_sizes.max()}, mean: {cell_sizes.mean():.1f}")
    if cell_sizes.min() < 5:
        print("  ⚠️  Some cells have <5 observations — interpret with caution")

    # ── Three-way ANOVA ──────────────────────────────────────────────────────
    print("\nRunning 3-way ANOVA (Type III SS)...")
    anova_table, ols_model = run_anova(df)

    # ── Logistic regression ──────────────────────────────────────────────────
    print("Running logistic regression (robustness check)...")
    logit_summary, logit_model = run_logistic(df)

    # ── Post-hoc Tukey ───────────────────────────────────────────────────────
    print("Running Tukey HSD post-hoc tests...")
    ph_diff   = posthoc_tukey(df, "Difficulty")
    ph_prompt = posthoc_tukey(df, "Prompt")
    ph_plat   = posthoc_tukey(df, "Platform")

    # ── Cell means ───────────────────────────────────────────────────────────
    cells = cell_means(df)

    # ── Assumption checks ────────────────────────────────────────────────────
    print("Checking ANOVA assumptions...")
    assumptions = check_assumptions(df, ols_model)

    # ── Print report ─────────────────────────────────────────────────────────
    print_report(anova_table, logit_summary, df)

    # ── Post-hoc summary ─────────────────────────────────────────────────────
    print("\n── POST-HOC TUKEY HSD ────────────────────────────────────────────────")
    for name, ph in [("Difficulty", ph_diff),
                     ("Prompt",     ph_prompt),
                     ("Platform",   ph_plat)]:
        print(f"\n  {name}:")
        print(f"  {'Group 1':<14} {'Group 2':<14} {'Mean 1':>8} {'Mean 2':>8} "
              f"{'Diff (pp)':>10} {'p-adj':>10} {'Reject H0':>10}")
        print("  " + "─"*74)
        for _, r in ph.iterrows():
            sig = "YES ***" if r["reject"] else "no"
            g1m = float(r["group1_mean"]) if pd.notna(r["group1_mean"]) else 0
            g2m = float(r["group2_mean"]) if pd.notna(r["group2_mean"]) else 0
            diff = float(r["mean_diff_pp"]) if pd.notna(r["mean_diff_pp"]) else 0
            print(f"  {str(r['group1']):<14} {str(r['group2']):<14} "
                  f"{g1m*100:>7.1f}% {g2m*100:>7.1f}% "
                  f"{diff:>+9.1f}pp {float(r['p-adj']):>10.4f}  {sig}")

    # ── Assumptions summary ───────────────────────────────────────────────────
    print("\n── ASSUMPTION CHECKS ─────────────────────────────────────────────────")
    for _, r in assumptions.iterrows():
        result = "✅ PASS" if r["pass_p05"] else ("❌ FAIL" if r["pass_p05"] is False else "ℹ️ INFO")
        print(f"  {r['test']:<35} stat={r['statistic']:>10.4f}  "
              f"p={r['p_value']:.4f}  {result}")
        print(f"    → {r['note']}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    print("\nSaving outputs...")
    anova_table.round(6).to_csv(OUT_ANOVA)
    print(f"  → {OUT_ANOVA}")

    logit_summary.to_csv(OUT_LOGISTIC)
    print(f"  → {OUT_LOGISTIC}")

    ph_diff.to_csv(OUT_PH_DIFF,   index=False)
    ph_prompt.to_csv(OUT_PH_PROMPT, index=False)
    ph_plat.to_csv(OUT_PH_PLAT,   index=False)
    print(f"  → {OUT_PH_DIFF}, {OUT_PH_PROMPT}, {OUT_PH_PLAT}")

    cells.to_csv(OUT_CELL_MEANS, index=False)
    print(f"  → {OUT_CELL_MEANS}")

    assumptions.to_csv(OUT_ASSUMPTIONS, index=False)
    print(f"  → {OUT_ASSUMPTIONS}")

    print(f"\n🎉 Done. {len(df)} observations across "
          f"{df.groupby(['Difficulty','Prompt','Platform']).ngroups} cells.")


if __name__ == "__main__":
    main()