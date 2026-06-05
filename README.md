# Prompt Strategy Evaluation for LLM Code Generation 

> **LLaMA-3.3-70B-Instruct · Phase 1 Evaluation · 450 Problems · 3 Prompt Strategies**  
> A thesis-grade research pipeline evaluating whether prompt engineering changes functional code correctness across competitive programming platforms.

---

## Table of Contents

1. [Research Overview](#1-research-overview)
2. [Hypotheses](#2-hypotheses)
3. [Dataset](#3-dataset)
4. [Experimental Design](#4-experimental-design)
5. [Pipeline Architecture](#5-pipeline-architecture)
6. [Evaluation Protocol](#6-evaluation-protocol-lcb-aligned)
7. [Prompt Strategies](#7-prompt-strategies)
8. [Results Summary](#8-results-summary)
9. [Statistical Analysis](#9-statistical-analysis)
10. [Key Findings](#10-key-findings)
11. [Output Files](#11-output-files)
12. [How to Run](#12-how-to-run)
13. [Project Structure](#13-project-structure)
14. [Dependencies](#14-dependencies)
15. [Limitations](#15-limitations)
16. [Citation & References](#16-citation--references)

---

## 1. Research Overview

This research investigates whether the **structure and phrasing of a prompt** given to a large language model (LLM) meaningfully changes its ability to generate functionally correct code for competitive programming problems. The study uses **execution-based evaluation** - solutions are compiled and run against hidden test cases - rather than text-similarity metrics such as BLEU, which are known to correlate poorly with functional correctness.

The evaluation framework is aligned with **LiveCodeBench** (Jain et al., 2024, arXiv:2403.07974), a rigorous benchmark that evaluates LLMs holistically across problem difficulty and platform origin. The pipeline extends LiveCodeBench's approach by testing three distinct prompt strategies across problems sourced from three competitive programming platforms (LeetCode, AtCoder, Codeforces) at three difficulty tiers (Easy, Medium, Hard).

**Model evaluated:** `meta-llama/llama-3.3-70b-instruct` (served via OpenRouter)  
**Total inference runs:** 1,350 (450 problems × 3 prompts)  
**Evaluation:** Hidden test pass rate (functional correctness only)

---

## 2. Hypotheses

| ID | Hypothesis |
|----|-----------|
| **H1** | Platform of origin significantly predicts LLM solve rate, independently of the difficulty label applied to a problem. |
| **H2** | Prompt strategy effects are platform-conditional - no single prompt strategy is universally optimal across all platforms. |
| **H3** | There is no measurable overfitting to public test cases - public and hidden test pass rates will be equal across all platforms. |

---

## 3. Dataset

### Problem Sources

| Platform | Rating System | Difficulty Tiers Used | Total Problems |
|---|---|---|---|
| LeetCode | Categorical tag (editorial team) | Easy, Medium, Hard | 205 |
| AtCoder | ELO-style (0–3000+) - ABC rounds only | Easy (0–199), Medium (200–399), Hard (400–500) | 236 |
| Codeforces | Numeric per-problem - Div.3 & Div.4 only | 800, 801–1000, 1001–1300 | 9 |
| **Total** | | | **450** |

>  Codeforces n=9 - results are directional only and not statistically generalisable.

### Dataset File

```
problems_dataset_150x3.csv
```

| Column | Description |
|---|---|
| `ProblemID` | Unique identifier (e.g., E01, M25, H48) |
| `ProblemName` | Problem title |
| `Difficulty` | Easy / Medium / Hard (dataset label) |
| `Platform` | LeetCode / AtCoder / Codeforces |
| `Description` | Full problem statement |
| `PublicTests` | JSON array of public example test cases |
| `HiddenTests` | JSON array of hidden evaluation test cases |
| `InputMode` | `functional` (LeetCode) or `stdin` (AtCoder/CF) |

### Problem Distribution in Phase 1 Subset (n=150)

| Platform | Easy | Medium | Hard | Total |
|---|---|---|---|---|
| LeetCode | 46 | 48 | 29 | **123** |
| AtCoder | 4 | 1 | 21 | **26** |
| Codeforces | 0 | 1 | 0 | **1** |
| **Total** | **50** | **50** | **50** | **150** |

---

## 4. Experimental Design

### Phase 1 - Prompt Strategy Comparison

- **Model:** LLaMA-3.3-70B-Instruct
- **Problems:** 150 (50 Easy + 50 Medium + 50 Hard)
- **Prompts per problem:** 3 (P1, P2, P3)
- **Total runs:** 450
- **Temperature:** 0 (deterministic)
- **Max tokens:** 2048

Each problem is sent independently to the model under each of the three prompt strategies. Results are logged per run with compile status, public test result, and hidden test result.

### Resume Logic

The pipeline supports resuming interrupted runs - already-completed `(ProblemID, Prompt)` pairs are skipped automatically by checking the existing output CSV before each run.

### Rate Limiting

All API calls are throttled with a configurable sleep interval (`SLEEP_P1 = 6s` default) to comply with OpenRouter rate limits. Exponential backoff (20s × attempt) is applied on 429 responses.

---

## 5. Pipeline Architecture

```
pipeline_lcb_aligned.py
│
├── load_problems()           ← Reads dataset CSV, detects input_mode per problem
│
├── build_prompt()            ← Generates P1/P2/P3 prompt (format-aware)
│
├── call_api()                ← OpenRouter API call with retry/backoff
│
├── extract_code()            ← Strips markdown fences from response
│
├── check_compile()           ← AST compile check before execution
│
├── run_tests()               ← Routes to functional or stdin evaluator
│   ├── run_tests_functional()   ← exec() → solution(*args) → compare return
│   └── run_tests_stdin()        ← subprocess → stdin pipe → compare stdout
│
├── lcb_check()               ← 7-layer output equivalence checker
│
└── run_statistics()          ← Chi-Square + pairwise Fisher Exact tests
```

---

## 6. Evaluation Protocol (LCB-Aligned)

### Format-Aware Evaluation

Problems are classified into two execution modes based on platform:

| Platform | Input Mode | Evaluator Used |
|---|---|---|
| LeetCode | `functional` | `run_tests_functional()` - exec() → `solution(*args)` → compare return value |
| AtCoder | `stdin` | `run_tests_stdin()` - subprocess → stdin pipe → compare stdout |
| Codeforces | `stdin` | `run_tests_stdin()` - subprocess → stdin pipe → compare stdout |

The mode is detected automatically from the `Platform` or `InputMode` column in the dataset CSV. If neither is present, `stdin` is used as a safe default.

### Why Execution-Based?

Text-similarity metrics (BLEU, ROUGE) correlate poorly with functional correctness in code. A solution can score high BLEU while being completely wrong, and vice versa (Evtikhiev et al., 2022, "Out of the BLEU", arXiv:2208.03133). Execution-based evaluation is the standard used in LiveCodeBench, HumanEval, and MBPP.

### LCB 7-Layer Output Checker (`lcb_check`)

To handle output format variation without false negatives, the checker applies seven equivalence layers in order:

| Layer | Check |
|---|---|
| L1 | Exact string match |
| L2 | Boolean normalisation (`True`→`true`, `False`→`false`) |
| L3 | Float tolerance (< 1e-6) |
| L4 | AST literal equality (handles Python list/tuple notation) |
| L5 | Set equivalence (order-independent list matching) |
| L6 | JSON equality |
| L7 | Line-by-line recursive check (multi-line outputs) |

### Stdout Fallback (Functional Mode)

If a functional-mode solution uses `print()` instead of `return`, the captured stdout is used as a fallback before the test case is marked as failed. This recovers format-confused solutions without masking genuine logic errors.

### Quality Buckets

Each solution is classified into one of three quality buckets for failure analysis:

| Bucket | Condition | Meaning |
|---|---|---|
| B1 - Full correct | Compiles + passes public + passes hidden | Correct solution |
| B3 - Logic failure | Compiles but fails public tests | Wrong algorithm |
| B4 - Syntax failure | Does not compile | Broken code |

---

## 7. Prompt Strategies

All three strategies use the same underlying problem content (name, description, first two examples). They differ only in framing and reasoning instruction.

### P1 - Zero-Shot Basic

```
Solve the following coding problem in Python.

Problem: {name}
{description}

Examples:
{examples}

{format_instruction}
```

A minimal prompt. No persona, no reasoning instruction. Tests the model's baseline ability with no scaffolding.

### P2 - Expert-Structured

```
You are an expert Python programmer.

Requirements:
  - Handle all edge cases and constraints
  - Optimise for time and space complexity
  - Match the required interface exactly

Problem: {name}
{description}

Examples:
{examples}

{format_instruction}
```

Adds an expert persona and explicit quality requirements. Tests whether role-priming and quality framing improve correctness.

### P3 - Chain-of-Thought (CoT)

```
Solve this coding problem step by step in Python.

Step 1 - Restate the problem in your own words.
Step 2 - Identify all constraints and edge cases.
Step 3 - Choose the optimal algorithm and data structure.
Step 4 - Trace through the example to verify your approach.
Step 5 - Implement the final solution in Python.

Problem: {name}
{description}

Examples:
{examples}

{format_instruction}
```

Elicits explicit step-by-step reasoning before code generation. Tests whether structured thinking improves functional correctness - particularly on harder problems.

### Format Instructions (LCB-Aligned)

Each prompt ends with one of two format instructions depending on `input_mode`:

**Functional (LeetCode):** Model must write `def solution(...)` that **returns** the answer. No `print()` calls.

**Stdin (AtCoder/Codeforces):** Model must write a **complete program** that reads from `stdin` and writes to `stdout`.

---

## 8. Results Summary

### Overall Platform Performance

| Platform | N Problems | Total Runs | Solve Rate | Pub–Hid Gap | Best Prompt |
|---|---|---|---|---|---|
| AtCoder | 236 | 708 | **67.9%** | 0.0pp | P3 |
| Codeforces | 9 | 27 | **85.2%** | 0.0pp | P1=P2 |
| LeetCode | 205 | 615 | **54.1%** | 0.0pp | P1 |

### Platform × Difficulty Solve Rates (Full Dataset)

| Platform | Easy | Medium | Hard |
|---|---|---|---|
| AtCoder | 91.3% | 66.7% | 47.9% |
| Codeforces | 100.0% | 50.0% | 88.9%  |
| LeetCode | 72.0% | 47.8% | 44.0% |

>  Codeforces Hard (88.9%) > Medium (50.0%) is a **sampling artefact** - n=3 Hard problems, not a real difficulty inversion.

### Platform × Prompt Performance (Full Dataset)

| Platform | n | P1 Rate | P2 Rate | P3 Rate | Best |
|---|---|---|---|---|---|
| AtCoder | 236 | 66.9% | 66.5% | **70.3%** | P3 |
| Codeforces | 9 | **88.9%** | **88.9%** | 77.8% | P1=P2 |
| LeetCode | 205 | **56.1%** | 52.2% | 54.2% | P1 |

### Performance by Difficulty × Prompt (Phase 1, n=150)

| Difficulty | n | P1 Pass | P1 Rate | P2 Pass | P2 Rate | P3 Pass | P3 Rate | Best |
|---|---|---|---|---|---|---|---|---|---|
| Easy | 50 | 13 | 26.0% | 15 | **30.0%** | 11 | 22.0% | P2 |
| Medium | 50 | 5 | 10.0% | 5 | 10.0% | 6 | **12.0%** | P3 |
| Hard | 50 | 12 | 24.0% | 14 | **28.0%** | 9 | 18.0% | P2 |
| **ALL** | **150** | **30** | **20.0%** | **34** | **22.7%** | **26** | **17.3%** | **P2** |

### Problem Hardness Distribution (Phase 1)

| Hardness Label | Count | Description |
|---|---|---|
| `easy - all prompts succeed` | 11 | Solved by P1, P2, and P3 |
| `mixed - some succeed` | 18 | Solved by at least one prompt but not all |
| `hard - all prompts fail` | 121 | Failed by all three prompts |

---

## 9. Statistical Analysis

### Chi-Square + Pairwise Fisher Exact Tests

Computed across all 9 (Difficulty × Prompt) groups for three metrics: compile rate, public pass rate, hidden pass rate.

**Results (Hidden Pass Rate - primary metric):**

| Difficulty | Test | Stat | p-value | Significant? |
|---|---|---|---|---|
| Easy | Chi² (P1 vs P2 vs P3) | - | - |  |
| Easy | Fisher P1 vs P2 | OR reported | - |  |
| Medium | Chi² (P1 vs P2 vs P3) | - | - |  |
| Hard | Chi² (P1 vs P2 vs P3) | - | - |  |

> No pairwise prompt comparison reached p < 0.05 within any single difficulty tier, indicating prompt effects are small relative to problem difficulty effects.

### Platform Effect Tests

| Test | Comparison | Statistic | p-value | Result |
|---|---|---|---|---|
| Kruskal-Wallis (overall) | AtCoder vs CF vs LC | H = 32.83 | p < 0.0001 |  Significant |
| Mann-Whitney U | AtCoder vs LeetCode | U = 247,735.5 | p < 0.0001 |  Significant |
| Mann-Whitney U | Codeforces vs LeetCode | U = 10,879.5 | p = 0.0015 |  Significant |
| Mann-Whitney U | AtCoder vs Codeforces | U = 7,909.5 | p = 0.058 |  Not significant |
| Chi-square (contingency) | AtCoder vs CF vs LC | χ² = 32.86 | p < 0.0001 |  Significant |
| Kruskal-Wallis (Easy only) | All 3 platforms | H = 31.15 | p < 0.0001 |  Significant |
| Kruskal-Wallis (Medium only) | All 3 platforms | H = 15.29 | p = 0.0005 |  Significant |
| Kruskal-Wallis (Hard only) | All 3 platforms | H = 6.95 | p = 0.031 |  Significant |
| MWU pub–hid gap | All pairwise | - | p = 1.0 |  Gap = 0 everywhere |

### Generation Time vs Correctness

Pearson correlation between generation time (seconds) and hidden test correctness:

| Difficulty | P1 | P2 | P3 |
|---|---|---|---|
| Easy | r = 0.004 | r = 0.014 | r = 0.080 |
| Medium | r = 0.046 | r = −0.220 | r = −0.178 |
| Hard | r = 0.034 | r = **0.362** | r = −0.103 |

> Only Hard-P2 shows a moderate positive correlation (r = 0.36), suggesting that on hard problems under the structured prompt, additional generation time may reflect genuine extra computation. Across all other conditions, longer generation does not predict correct code.

---

## 10. Key Findings

### Finding 1 - Platform origin is the strongest predictor of solve rate

The platform a problem comes from explains more variance in solve rate than the difficulty label applied to it. The Kruskal-Wallis test confirmed this platform effect holds within every difficulty stratum (Easy p < 0.0001, Medium p = 0.0005, Hard p = 0.031). The clearest evidence: AtCoder Easy (91.3%) vs LeetCode Easy (72.0%) - a 19.3pp gap on identically labelled problems.

### Finding 2 - Best prompt is platform-dependent, not universal

| Platform | Best Prompt | Reason |
|---|---|---|
| AtCoder | P3 (Chain-of-Thought) | 70.3% vs 66.9% (P1) |
| LeetCode | P1 (Zero-Shot) | 56.1% vs 52.2% (P2) |
| Codeforces | P1=P2 (tied) | 88.9% each |

A universal prompting strategy will systematically underperform a platform-aware one.

### Finding 3 - Prompt differences are not statistically significant within difficulty tiers

No pairwise prompt comparison reached p < 0.05 within Easy, Medium, or Hard tiers individually. The practical differences (e.g., P2 at 28% vs P3 at 18% on Hard) are real but small relative to the much larger effect of problem difficulty and platform origin.

### Finding 4 - Zero overfitting across all platforms

Public test pass rate equals hidden test pass rate exactly across all 1,350 runs (gap = 0.0pp on every platform, MWU p = 1.0). The model either solves or fails to solve the underlying algorithm - it does not pattern-match to visible examples.

### Finding 5 - Most failures are logic errors, not syntax errors

Across all difficulty and prompt combinations, 70–88% of all solutions compile successfully but fail public tests (Bucket B3 - logic failure). Non-compiling code (Bucket B4 - syntax error) is rare (0–8%). The model understands Python syntax; its weakness is algorithmic problem-solving.

### Finding 6 - Hard problems are not uniformly harder than Easy for this model

Several Easy problems (e.g., `buy-two-chocolates`, `semi-ordered-permutation`) fail across all three prompts, while several Hard problems (e.g., `B. 250 Thousand Tons of TNT`, `count-complete-substrings`) are solved by all three. The nominal difficulty label is an unreliable predictor of model performance at the individual problem level.

---

## 11. Output Files

| File | Description |
|---|---|
| `phase1_llama70b-2-2204-metav2150_450x3.csv` | Phase 1 raw results - one row per (problem, prompt) |
| `platform_solve_rates.csv` | Solve rates by platform × difficulty × prompt |
| `platform_public_vs_hidden.csv` | Public vs hidden pass rate gap per platform |
| `platform_stat_tests.csv` | All statistical test results |
| `platform_prompt_interaction.csv` | Solve rates by platform × prompt (aggregated) |
| `chi_square_report2.csv` | Chi-Square + Fisher Exact results per difficulty × metric |
| `analysis_prompt_difficulty_stats-5.csv` | Compile/public/hidden rates per difficulty × prompt with gen time |
| `analysis_problem_hardness-2.csv` | Per-problem hardness classification (easy/mixed/hard) |
| `analysis_quality_buckets-3.csv` | B1/B3/B4 bucket counts per difficulty × prompt |
| `analysis_gentime_hidden_corr.csv` | Pearson correlation: generation time vs hidden correctness |

### Phase 1 CSV Schema

| Column | Type | Description |
|---|---|---|
| `ProblemID` | string | Unique ID (e.g., E01, M25, H48) |
| `ProblemName` | string | Problem title |
| `Difficulty` | string | Easy / Medium / Hard |
| `Prompt` | string | P1 / P2 / P3 |
| `Model` | string | Model identifier |
| `InputMode` | string | functional / stdin |
| `EvalMethod` | string | lcb-functional / lcb-stdin |
| `Compiled` | int | 1 = compiled, 0 = syntax error |
| `PublicPassed` | int | 1 = all public tests passed |
| `HiddenPassed` | int | 1 = all hidden tests passed (primary metric) |
| `GenTime_s` | float | Wall-clock generation time in seconds |
| `ErrorMsg` | string | First error message if failed |
| `GeneratedCode` | string | Generated code (only for failed solutions) |

---

## 12. How to Run

### Prerequisites

```bash
pip install openai pandas numpy scipy
```

### Environment Setup

```bash
export OPENROUTER_API_KEY="your-key-here"
```

>  Do NOT hard-code API keys in the script. Read from environment variable before running.

### Phase 1 - Generate & Evaluate

```bash
python3 pipeline_lcb_aligned.py --phase 1
```

Runs LLaMA-3.3-70B-Instruct on all 150 problems × 3 prompts. Resumes from last completed row if interrupted.

### Statistics Only

```bash
python3 pipeline_lcb_aligned.py --phase stats
```

Reads the existing Phase 1 CSV and computes Chi-Square + Fisher Exact statistics. Does **not** re-run any API calls.

### Platform Analysis

```bash
python3 platform_analysis.py
```

Generates all four platform-level output CSVs (solve rates, pub vs hidden, stat tests, prompt interaction).

### Run All Phases

```bash
python3 pipeline_lcb_aligned.py --phase all
```

---

## 13. Project Structure

```
.
├── pipeline_lcb_aligned.py              # Main evaluation pipeline
├── platform_analysis.py                 # Platform-level statistical analysis
├── problems_dataset_150x3.csv           # Input dataset (450 problems)
│
├── results/
│   ├── phase1_llama70b-2-2204-metav2150_450x3.csv
│   ├── platform_solve_rates.csv
│   ├── platform_public_vs_hidden.csv
│   ├── platform_stat_tests.csv
│   ├── platform_prompt_interaction.csv
│   ├── chi_square_report2.csv
│   ├── analysis_prompt_difficulty_stats-5.csv
│   ├── analysis_problem_hardness-2.csv
│   ├── analysis_quality_buckets-3.csv
│   └── analysis_gentime_hidden_corr.csv
│
└── README.md
```

---

## 14. Dependencies

| Package | Version | Purpose |
|---|---|---|
| `openai` | ≥ 1.0 | OpenRouter API client |
| `pandas` | ≥ 2.0 | Data loading and manipulation |
| `numpy` | ≥ 1.24 | Numerical operations |
| `scipy` | ≥ 1.11 | `fisher_exact`, `chi2_contingency` |
| `python` | ≥ 3.10 | Required for `signal.alarm` (Unix only) |

>  `signal.alarm` is not available on Windows. The functional-mode evaluator requires a Unix/macOS/Linux environment.

---

## 15. Limitations

| Limitation | Impact |
|---|---|
| Single model evaluated | Results may not generalise to other LLMs with different instruction-following characteristics |
| Temperature = 0 | Deterministic outputs; no estimate of output variance across runs |
| Codeforces n=9 | Codeforces results are directional only - not statistically generalisable |
| Phase 1 subset ≠ full dataset | The 150 phase1 problems are a subset of the 450 full-dataset problems; per-tier rates differ between the two |
| Difficulty label inconsistency | "Easy" on LeetCode and "Easy" on AtCoder are not equivalent; cross-platform difficulty comparisons should be interpreted as platform comparisons, not true difficulty comparisons |
| Unix-only execution | The functional evaluator uses `signal.SIGALRM` for timeout, which is unavailable on Windows |
| API key hard-coded in original | Must be moved to environment variable before any public repository usage |

---

## 16. Citation & References

If using this pipeline or results in academic work, please cite:

```
@misc{yadav2025promptstrategy,
  title   = {Prompt Strategy Evaluation for LLM Code Generation on Competitive Programming Benchmarks},
  author  = {Kriti Yadav},
  year    = {2025},
  note    = {Thesis research pipeline - LLaMA-3.3-70B-Instruct × P1/P2/P3 × 450 problems}
}
```

### Key References

- Jain, N. et al. (2024). *LiveCodeBench: Holistic and Contamination-Free Evaluation of Large Language Models for Code.* arXiv:2403.07974.
- Evtikhiev, M. et al. (2022). *Out of the BLEU: How Should We Assess Quality of the Code Generation Models?* arXiv:2208.03133.
- Chen, A. et al. (2024). *Evaluating Large Language Models for Code: A Survey.* arXiv:2408.16498.
- Chen, M. et al. (2021). *Evaluating Large Language Models Trained on Code (HumanEval).* arXiv:2107.03374.
- Wei, J. et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* arXiv:2201.11903.

---

*Last updated: June 2026 · Research pipeline by Kriti Yadav*
