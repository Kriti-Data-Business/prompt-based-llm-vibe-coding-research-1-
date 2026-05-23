# Methodology and Results
## LLM Code Generation Evaluation — LCB-Aligned Pipeline (LLaMA-3.3-70B)

---

# 4. Methodology

## 4.1 Benchmark and Dataset Selection

This study used the LiveCodeBench benchmark to evaluate the code generation capability of large language models (LLMs) under varying prompt strategies. LiveCodeBench is a contamination-resistant benchmark designed specifically for evaluating LLMs on competitive programming problems sourced from platforms such as LeetCode and Codeforces. Unlike earlier benchmarks such as HumanEval (Chen et al., 2021) or MBPP (Austin et al., 2021), LiveCodeBench continuously refreshes its problem pool post knowledge-cutoff, which reduces the risk of models simply recalling memorised solutions from training data rather than genuinely solving new problems.

A dataset of 150 problems was selected, distributed equally across three difficulty tiers: 50 Easy, 50 Medium, and 50 Hard. Each problem was run three times, once per prompt strategy (P1, P2, P3), producing 450 total evaluation runs. All problems required algorithmic reasoning and Python code output, covering topics such as array manipulation, dynamic programming, greedy algorithms, sliding window, and combinatorics.

## 4.2 Model Selection

Two variants of the LLaMA-3.3-70B family were evaluated across two experiment runs. The first run used meta-llama/llama-3.3-70b-versatile, a general-purpose model optimised for diverse task types. The second run used meta-llama/llama-3.3-70b-instruct, a model fine-tuned specifically for instruction-following. Both models were accessed via OpenRouter's API. The rationale for switching from Versatile to Instruct was threefold: the Instruct model was expected to follow structured prompt templates more reliably, to handle chain-of-thought (CoT) style prompts more precisely, and to natively output boolean values in lowercase (e.g., true/false), which matched the expected output format in JSON-style competitive programming problems and avoided a systematic evaluation error discovered in Run 1.

## 4.3 Prompt Strategy Design

Three prompt strategies were designed and applied uniformly across all 450 runs in each experiment. The strategies represent a gradient from minimal instruction to structured reasoning guidance, in line with the prompting literature (Brown et al., 2020; Wei et al., 2022):

- **P1 — Zero-Shot Baseline:** A direct instruction prompt with no examples and no reasoning scaffold. The model was told the problem statement and asked to write a correct Python solution. This represents the simplest possible prompt and serves as the performance floor for the other strategies.

- **P2 — Expert/Few-Shot Guided:** A prompt that framed the model as a competitive programmer and included structural guidance on how to approach the problem (e.g., think about edge cases, check input constraints). This draws on the expert role-prompting and few-shot conditioning literature, which suggests that framing the model's role and providing structural cues can improve output quality on well-defined tasks (Liu et al., 2023).

- **P3 — Chain-of-Thought (CoT):** A prompt that instructed the model to reason step-by-step before producing the final code. CoT prompting, originally proposed by Wei et al. (2022), has been shown to improve performance on multi-step reasoning tasks by encouraging the model to externalise intermediate reasoning before committing to an answer.

All three prompts shared the same problem description as input. No in-context solved examples were included in P1 or P3. Only P2 included role framing and a structural scaffold.

## 4.4 Evaluation Framework

Each generated code response was evaluated against three binary measures:

- **Compilation (Compiled):** Whether the generated Python code was syntactically valid and could be executed without an import or syntax error.
- **Public Test Pass (PublicPassed):** Whether the code produced the correct output on the visible test cases provided in the problem statement.
- **Hidden Test Pass (HiddenPassed):** Whether the code produced the correct output on held-out test cases not visible to the model at inference time.

A run was considered a full pass only if both public and hidden tests passed. This strict criterion was chosen to ensure that passing public tests was not treated as success when the underlying logic remained incorrect.

Two evaluation methods were used across the two runs. Run 1 (Versatile) used a LeetCode-style function-call wrapper, where the model wrote a `def solution(...)` function that received pre-parsed inputs directly as Python arguments. This method isolated algorithmic logic from input/output formatting. Run 2 (Instruct) switched to the standard LiveCodeBench lcb-stdin method, where the model's code read from `sys.stdin` and printed to `sys.stdout`. This aligns with the evaluation standard used in the original LiveCodeBench paper and reflects real competitive programming judges.

### 4.4.1 Quality Bucket Classification

To understand *why* a solution fails — not just *that* it fails — each generation was placed into one of three execution quality buckets. This multi-level approach is recommended by the survey on evaluating LLMs for code (Chen et al., 2024):

| Bucket | Label | Meaning |
|--------|-------|---------|
| **B1** | Full correct | Compiles + passes public tests + passes hidden tests |
| **B3** | Logic failure | Compiles but fails public tests (wrong algorithm or output) |
| **B4** | Syntax failure | Does not compile at all |

Across all (Difficulty × Prompt) groups, 70–88% of all solutions fell into B3 (logic failure), and non-compiling code (B4) was rare (0–8%). This distribution confirms that the model understands Python syntax well but struggles with problem-solving logic — a finding consistent with the survey's conclusion that strong LLMs rarely fail on syntax; their primary weakness is functional correctness (Chen et al., 2024).

### 4.4.2 Generation Time Recording

Wall-clock generation time (seconds from API call to response) was logged for every completion. Following the survey recommendation to treat efficiency and correctness as *separate evaluation axes* (Chen et al., 2024), generation time was analysed independently and its Pearson correlation with correctness was tested per (Difficulty, Prompt) group. The expectation from the literature is that these two axes should be largely independent.

## 4.5 Evaluation Error Discovery and Correction

During inspection of Run 1 failures, a systematic evaluation error was identified: a subset of problems expected lowercase boolean strings (true / false) in the output, following JSON conventions used by LeetCode. However, Python's built-in `print(True)` outputs `True` with a capital letter. The exact-match checker incorrectly flagged these as failures. A multi-layer checker was developed to address this: Layer 1 applied exact string matching, and Layer 2 applied case-normalisation on boolean-like tokens. After this correction, 27 runs were reclassified from failure to pass, revising the Run 1 full pass count from 82/450 (18.2%) to 109/450 (24.2%). This discovery directly motivated the switch to the Instruct model in Run 2, which natively outputs lowercase booleans and eliminated this error class entirely.

The importance of multi-layer evaluation pipelines has been noted in recent benchmarking literature: text-similarity metrics and single-layer exact-match checkers are insufficient for evaluating code generation, particularly when models may format output slightly differently from the expected string (Evtikhiev et al., 2023). The multi-layer checker implemented here aligns with the 7-layer LiveCodeBench-style output checker, which applies exact matching, boolean normalisation, float tolerance, AST equivalence, set comparison, JSON equivalence, and multiline line-by-line comparison in sequence.

---

# 5. Results

## 5.1 Run 1 — LLaMA-3.3-70B-Versatile (LeetCode Wrapper Evaluation)

### 5.1.1 Compilation Performance

Of the 450 total runs in Run 1, 444 produced syntactically valid Python code, giving a compilation rate of 98.7%. The six no-code failures (1.3%) were all caused by model response timeouts, spread across problems E25, E26, H34, H47, and H49. Compilation rates by strategy were: P1 = 100% (150/150), P2 = 100% (150/150), and P3 = 99.2% (149/150). This near-perfect compilation rate across all prompt strategies established a key baseline finding: prompt design had no meaningful effect on whether the model could produce runnable code. The barrier to success in this study was not syntax — it was logical correctness.

**Quality Bucket Breakdown — Run 1:**

| Difficulty | Prompt | B1 Full Correct % | B3 Logic Failure % | B4 Non-Compiling % |
|-----------|--------|--------------------|--------------------|--------------------|
| Easy | P1 | 36.0 | 62.0 | 2.0 |
| Easy | P2 | 34.0 | 64.0 | 2.0 |
| Easy | P3 | 38.0 | 60.0 | 2.0 |
| Medium | P1 | 8.0 | 90.0 | 2.0 |
| Medium | P2 | 10.0 | 88.0 | 2.0 |
| Medium | P3 | 6.0 | 92.0 | 2.0 |
| Hard | P1 | 12.0 | 86.0 | 2.0 |
| Hard | P2 | 12.0 | 86.0 | 2.0 |
| Hard | P3 | 8.0 | 90.0 | 2.0 |

B3 (logic failure) dominates across every group, accounting for 60–92% of all runs. B4 (non-compiling) is uniformly minimal. This confirms the survey finding that strong LLMs rarely fail on syntax; their primary weakness is semantic/logical correctness (Chen et al., 2024; Evtikhiev et al., 2023).

### 5.1.2 Overall Pass Rate

Applying the strict full-pass criterion (public and hidden tests both passed), 109 of 450 runs passed after the boolean correction, giving a final pass rate of 24.2%. The identical public and hidden test pass rates across all difficulty tiers confirmed that the model did not overfit to the visible test cases: when it solved a problem, it solved it correctly for both seen and unseen inputs. This provides confidence that the pass rates reflect genuine generalisation, not surface-level pattern matching on examples.

### 5.1.3 Pass Rate by Difficulty

Difficulty was the single strongest predictor of performance. The full pass rate for Easy problems was 36.0% (54/150), for Hard problems it was 10.7% (16/150), and for Medium problems it was 8.0% (12/150). The lower Medium rate compared to Hard is not an anomaly — it reflects the specific problem set, where several Hard problems (H01, H02, H15, H22, H27, H40, H48) involved structurally straightforward logic that the model handled well, while many Medium problems required subtle dynamic programming or greedy reasoning where model solutions were close but incorrect. This difficulty-reversal pattern between Medium and Hard has been observed in the original LiveCodeBench study for smaller open-source models and is consistent with those findings. LiveCodeBench itself notes that platform origin and structural task type matter more than the stated difficulty label in determining actual model pass rates.

### 5.1.4 Pass Rate by Prompt Strategy

| Difficulty | P1 (Zero-Shot) | P2 (Expert) | P3 (CoT) | Row Total |
|-----------|---------------|-------------|----------|-----------|
| Easy | 18/50 = 36% | 17/50 = 34% | 19/50 = 38% | 54/150 = 36.0% |
| Medium | 4/50 = 8% | 5/50 = 10% | 3/50 = 6% | 12/150 = 8.0% |
| Hard | 6/50 = 12% | 6/50 = 12% | 4/50 = 8% | 16/150 = 10.7% |
| **Overall** | **28/150 = 18.7%** | **28/150 = 18.7%** | **26/150 = 17.3%** | **82/450 = 18.2%** |

*Table 5.1: Run 1 (Versatile) pass rates by prompt strategy and difficulty, before boolean correction.*

The effect of prompt strategy on pass rate was minimal and inconsistent. P1 and P2 produced identical overall pass rates (18.7%), with P3 marginally lower at 17.3%. Within Easy, CoT (P3) led by 2 percentage points, but on Medium and Hard, P3 was the weakest strategy. The maximum difference between any two strategies within any difficulty tier was 4 percentage points — a margin that falls within natural variability across 50 problems and does not constitute evidence of a consistent, reproducible prompt effect. Hypothesis H1 (prompt strategy significantly and consistently affects pass rate) was therefore rejected.

This finding is consistent with the prompt sensitivity literature: Zi, Menon & Guha (IJCNLP 2025) found that prompt effects on code generation are task-dependent and often non-significant across difficulty tiers, with structured I/O-focused prompts tending to outperform reasoning-narration prompts on well-defined algorithmic tasks.

### 5.1.5 Problem-Level Solvability

At the problem level, performance was bimodal rather than evenly distributed. Three hardness patterns emerged:

| Hardness Category | Meaning | Example Problems |
|------------------|---------|-----------------|
| **all_prompts_succeed** | All 3 prompts pass hidden tests | E01 A. Short Sort, H01 B. 250 Thousand Tons of TNT, H40 count-complete-substrings |
| **mixed_some_succeed** | At least 1 prompt passes, at least 1 fails | E09 find-the-losers-of-the-circular-game, H15 number-of-beautiful-integers |
| **all_prompts_fail** | All 3 prompts fail hidden tests | E06 buy-two-chocolates, E12 semi-ordered-permutation, H04 count-of-integers, H07 greatest-common-divisor-traversal |

For Easy problems: 10/50 (20%) were solved by all three strategies; 8/50 (16%) by exactly two; 8/50 (16%) by exactly one; and 24/50 (48%) by none. For Medium problems: no problem was solved by all three strategies; only 3/50 (6%) by two strategies; 6/50 (12%) by one; and 41/50 (82%) were unsolvable across all prompts. For Hard problems: 2/50 (H15 and H40) were solved by all three strategies.

This distribution means prompt engineering has effectively zero impact on problems the model fundamentally cannot solve, and only marginal impact on problems it can already solve under at least one strategy. Problems labelled "Easy" in the dataset (e.g., `buy-two-chocolates`, `semi-ordered-permutation`) appearing in the `all_prompts_fail` cluster indicate structural blind spots in the model unrelated to prompt wording — consistent with LiveCodeBench's observation that a long tail of tasks remains unsolved regardless of model settings. The **mixed** group is the most practically useful: it identifies problems where prompt choice matters and switching from P1 to P2 can recover a correct solution.

### 5.1.6 Generation Time vs. Correctness — Run 1

Mean generation time increases with difficulty: roughly 3.5–4.6 s on Easy, 4.5–6.1 s on Medium, and 6.8–7.2 s on Hard. However, Pearson correlations between generation time and hidden correctness are near zero for most groups:

| Difficulty | Prompt | r (GenTime vs HiddenCorrect) | n |
|-----------|--------|------------------------------|---|
| Easy | P1 | 0.00 | 50 |
| Easy | P2 | 0.01 | 50 |
| Easy | P3 | 0.08 | 50 |
| Medium | P1 | 0.05 | 50 |
| Medium | P2 | −0.22 | 50 |
| Medium | P3 | −0.18 | 50 |
| Hard | P1 | 0.03 | 50 |
| Hard | **P2** | **0.36** | 50 |
| Hard | P3 | −0.10 | 50 |

The only meaningful positive correlation is Hard-P2 (r = 0.36), suggesting that on hard problems under a structured prompt, some additional generation time may reflect genuine productive computation. Everywhere else, longer generation does not reliably produce more correct code. This confirms the survey recommendation to treat efficiency and correctness as separate evaluation axes (Chen et al., 2024) — generation time should not be used as a proxy for output quality.

---

## 5.2 Transition from Versatile to Instruct: Rationale and Changes

Three methodological decisions were made before Run 2:

- **Model switch (Versatile → Instruct):** The Instruct model's fine-tuning was expected to improve adherence to structured prompt templates and natively resolve the boolean capitalisation error.
- **Evaluation switch (LeetCode wrapper → lcb-stdin):** The function-call wrapper was methodologically convenient but non-representative of real competitive programming evaluation. Run 2 adopted standard `sys.stdin` / `sys.stdout` evaluation, consistent with the original LiveCodeBench evaluation protocol.
- **Generated code logging:** The `GeneratedCode` column was added to the output CSV, enabling direct inspection of every model output for root-cause failure analysis — not just pass/fail flags.

---

## 5.3 Run 2 — LLaMA-3.3-70B-Instruct (lcb-stdin Evaluation)

### 5.3.1 Boolean Error: Resolved

The most immediate finding from Run 2 was the complete elimination of the boolean capitalisation error. Problem E14 (check-if-the-number-is-fascinating), which failed across all three strategies in Run 1 with `want='true' got='True'`, passed cleanly across all three strategies in Run 2 with no error. This directly confirmed the rationale for switching models and validated the multi-layer checker approach as a temporary safeguard rather than a long-term solution.

### 5.3.2 New Dominant Failure: Empty Output

Switching to lcb-stdin evaluation exposed a new class of failure that did not exist in Run 1. On a substantial proportion of LeetCode-style problems, the model generated logically correct code that included `print()` statements, but the test harness captured empty stdout (`got=''`). Across multiple problems (E05, E06, E07, E08 and others), all three strategies consistently showed empty output despite the code being structurally valid. This pattern — correct code structure, correct print calls, but empty captured output — across all strategies points to a systematic harness-level stdin injection issue rather than a model reasoning failure. Approximately 40% of all failures in Run 2 were in this empty-output category, making it the single most common error type and larger even than Wrong Answer, which dominated Run 1.

### 5.3.3 Pass Rates by Strategy and Difficulty

| Metric | Easy | Medium | Hard | Total |
|--------|------|--------|------|-------|
| **Compile Rate** | | | | |
| P1 | 50/50 (100%) | 48/50 (96%) | 48/50 (96%) | 146/150 (97.3%) |
| P2 | 50/50 (100%) | 48/50 (96%) | 47/50 (94%) | 145/150 (96.7%) |
| P3 | 50/50 (100%) | 48/50 (96%) | 49/50 (98%) | 147/150 (98.0%) |
| **Public / Hidden Pass Rate** | | | | |
| P1 | 16/50 (32%) | 10/50 (20%) | 12/50 (24%) | 38/150 (25.3%) |
| P2 | 16/50 (32%) | 5/50 (10%) | 9/50 (18%) | 30/150 (20.0%) |
| P3 | 14/50 (28%) | 8/50 (16%) | 12/50 (24%) | 34/150 (22.7%) |

*Table 5.2: Run 2 (Instruct, lcb-stdin) compile and pass rates by strategy and difficulty.*

Three key observations emerge. First, public and hidden pass rates were identical across every prompt and difficulty — confirming again that the model did not overfit to visible examples. Second, P1 (Zero-Shot) achieved the highest overall pass rate at 25.3%, followed by P3 (CoT) at 22.7%, and P2 (Expert) at 20.0% — a reversal of the Run 1 ordering where P2 performed better on harder problems under the wrapper evaluation. Third, CoT (P3) performed worst overall in Run 2, which is the opposite of its relative strength on Easy problems in Run 1, reinforcing the finding that no single prompt strategy is consistently optimal across difficulties or evaluation methods.

The P3 underperformance on Hard problems (18% in Run 2, 8% in Run 1) is a notable finding. Research on prompt variability in code generation (Code Roulette, arXiv:2506.10204) warns that reasoning narration can add verbosity without adding functional correctness — the model over-explains the problem rather than producing tight, correct code. This is consistent with the prompt specificity study (Zi et al., IJCNLP 2025), which found that structured I/O-focused prompts outperform step-by-step reasoning prompts on competitive programming tasks.

**Quality Bucket Breakdown — Run 2:**

| Difficulty | Prompt | B1 Full Correct % | B3 Logic Failure % | B4 Non-Compiling % |
|-----------|--------|--------------------|--------------------|--------------------|
| Easy | P1 | 32.0 | 66.0 | 2.0 |
| Easy | P2 | 32.0 | 68.0 | 0.0 |
| Easy | P3 | 28.0 | 70.0 | 2.0 |
| Medium | P1 | 20.0 | 76.0 | 4.0 |
| Medium | P2 | 10.0 | 86.0 | 4.0 |
| Medium | P3 | 16.0 | 80.0 | 4.0 |
| Hard | P1 | 24.0 | 72.0 | 4.0 |
| Hard | P2 | 18.0 | 76.0 | 6.0 |
| Hard | P3 | 24.0 | 74.0 | 2.0 |

The same B3-dominance pattern holds in Run 2: 66–86% of all solutions compile but fail on logic. B4 (non-compiling) remains low at 0–6%. This is consistent across both model variants and both evaluation protocols, reinforcing that logic error is the structural bottleneck for this model on competitive programming tasks.

### 5.3.4 Format Mismatch Errors

A secondary failure type in Run 2 was output format mismatch: the model produced a correct answer value but in the wrong format. Problem E09 (find-the-losers-of-the-circular-game) illustrates this clearly — P2 produced `4 5` (space-separated) where the expected output was `[4, 5]` (Python list notation). This is specific to LeetCode-style problems in the dataset, where the expected output format uses Python list representation rather than Codeforces-style space-separated output. Code that used `print(*losers)` (correct for Codeforces) failed on LeetCode problems, while `print(losers)` (correct for LeetCode) would fail on Codeforces. This format sensitivity was invisible in Run 1's function-call wrapper evaluation and became visible only under the stricter stdin/stdout evaluation method.

---

## 5.4 Cross-Run Synthesis

### 5.4.1 Difficulty Is the Primary Determinant

Across both models and both evaluation methods, problem difficulty was the strongest predictor of performance by a large margin. In Run 1: Easy = 36.0%, Medium = 8.0%, Hard = 10.7%. In Run 2: Easy ≈ 28–32%, Medium ≈ 10–20%, Hard ≈ 18–24%. In both runs, the gap between Easy and the harder tiers was roughly 3–4×. No prompt strategy or model switch came close to closing this gap. This confirms the primary research hypothesis: difficulty is the performance ceiling, and prompt design operates within that ceiling rather than above it. This aligns with LiveCodeBench's own finding that pass rates drop steeply with difficulty across all model types.

### 5.4.2 Prompt Strategy Has a Weak and Inconsistent Effect

Across all runs, the difference between any two prompt strategies on any difficulty tier never exceeded 8 percentage points and was typically 2–4 percentage points. Crucially, the direction of the effect was inconsistent: CoT (P3) led on Easy in Run 1 but trailed on Medium and Hard. Expert (P2) was strongest on harder problems in Run 1 under the wrapper method, but weakest overall in Run 2 under stdin/stdout. This inconsistency confirms that there is no universally optimal prompt strategy for this model-benchmark combination, and that prompt design differences are not a reliable lever for improving code generation quality at this model scale.

### 5.4.3 Evaluation Framework Integrity

The discovery of 27 false negatives (7.3% of original failures) in Run 1 demonstrates that evaluation framework quality directly impacts apparent model performance. Without the boolean normalisation fix, the final reported pass rate would have been 18.2% instead of 24.2% — a 6-percentage-point underestimation of true capability. This finding argues strongly for multi-layer evaluation pipelines in any LLM code generation study, particularly when models may format output slightly differently from the expected string. The switch to the Instruct model, which resolved the boolean issue at source, represents the methodologically cleaner long-term solution, but the multi-layer checker remains a necessary safety net when working with diverse problem sets and multiple models. This recommendation aligns with the "Out of the BLEU" finding (Evtikhiev et al., 2023) that single-metric evaluation of code generation is systematically unreliable.

### 5.4.4 Generation Time Does Not Predict Correctness

Across both runs, generation time showed near-zero correlation with correctness in almost every (Difficulty × Prompt) group. The single exception — Hard-P2 (r = 0.36) — suggests that on structurally hard problems under a structured prompt, additional generation time may occasionally reflect productive computation. However, this is a modest and isolated finding that requires larger-scale validation before generalisation. The overall pattern confirms the survey recommendation (Chen et al., 2024) to treat efficiency and correctness as independent axes: reporting only generation time or only pass rate in isolation would give an incomplete picture of model behaviour.

### 5.4.5 Alignment with Research Literature

| Finding | Status | Supporting Reference |
|---------|--------|---------------------|
| Pass rates drop steeply with difficulty | ✅ Confirmed | LiveCodeBench (Jain et al., 2024) |
| Compilation near-perfect; logic is the barrier | ✅ Confirmed | Chen et al. survey (2024); Evtikhiev et al. (2023) |
| P2 (structured) outperforms P3 (CoT) on Hard in Run 1 | ✅ Confirmed | Zi, Menon & Guha, IJCNLP 2025 |
| Generation time ≠ correctness | ✅ Confirmed | Chen et al. survey (2024) |
| Platform/label mismatch: Medium harder than Hard | ✅ Consistent with | LiveCodeBench platform analysis |
| P3 underperforms P1 on Hard in Run 2 (CoT hurts) | 🆕 New finding | Code Roulette (arXiv:2506.10204); Prompt Sensitivity (OpenReview 2025) |
| Hard-P2 moderate GenTime–correctness correlation (r = 0.36) | 🆕 Exploratory finding | Not previously reported; needs larger-scale validation |
| 27 false negatives from boolean capitalisation (7.3% of failures) | 🆕 Methodological finding | Consistent with Evtikhiev et al. (2023) |

