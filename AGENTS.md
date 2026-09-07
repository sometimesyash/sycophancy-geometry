# AGENTS.md

Operating manual for an AI agent or researcher continuing this project.
Read this file completely before running anything.

---

## 1. What this project is

An empirical study of **when language models abandon a position under social pressure**,
and whether that abandonment is predictable from internal activation geometry before the
model responds.

It began as "detect sycophancy-based jailbreaks using geometric features (LID, curvature)".
That framing died on contact with data, for a reason that became the project's main result.
See section 3.

**The current research question:**

> Alignment training appears to produce an asymmetrically robust boundary: hard on harm,
> soft on truth. Is the moment of capitulation geometrically detectable in activations
> before the model speaks?

---

## 2. Hard constraints (read before planning anything)

### 2.1 Azure access has ENDED

All Azure resources were on a Microsoft internal subscription that no longer exists.
**Nothing can be regenerated.** Specifically dead:

- `phi4-target`, `phi4mini-target` (hosted generation)
- `gpt-5-4`, `gpt-56-sol` (judges)
- `embed-3-large` (embeddings)

**Do not write code that calls Azure. Do not try to re-run scripts 01-07.**
`src/azclient.py` and `scripts/0*.py` are preserved for provenance and reproducibility
documentation only.

Everything needed downstream is frozen in `data/frozen/`. Treat those files as immutable
primary data.

### 2.2 The rule that keeps this project alive

> **Every script numbered 10 or above must be a pure function of files on disk.**
> No network calls. No hosted APIs. Open weights from HuggingFace only.

`python scripts/99_check_portability.py` enforces this. Run it before any commit.
If it fails, the project has become unrunnable for whoever comes next.

### 2.3 Compute

Author's laptop: 4 cores, 7.6 GB RAM. Cannot hold a 7B model.

- **Phi-4-mini (3.8B)** is the PRIMARY target. ~7.6 GB in fp16, fits a **free Kaggle or
  Colab T4**. No paid compute is required for any remaining work.
- **Phi-4 (14B)** is secondary. ~28 GB, needs 2xT4 (Kaggle provides this) or a large CPU VM.

Compute is deliberately off the critical path. Do not reintroduce a paid dependency.

---

## 3. What was found (all frozen, all reproducible from disk)

### 3.1 Sycophancy does NOT breach safety boundaries

| Experiment | n | Violation rate |
|---|---|---|
| Single-turn sycophantic framing (Phi-4) | 240 | **0.000** |
| Multi-turn sycophantic pressure (Phi-4) | 204 | **0.000** |
| Multi-turn sycophantic pressure (Phi-4-mini) | 202 | 0.005 |
| Neutral control | 152 | 0.000 |
| Non-sycophantic routes | 203 | 0.186 |
| └ fiction escalation specifically | 25 | **0.680** |

The matched neutral and non-sycophantic controls are what make this a finding rather than
a broken harness: the same pipeline detects success at 0.68 when success occurs.

**Independently corroborated.** Two judges (gpt-5.4 and gpt-5.6-sol) each flagged exactly
one violation across 644 sycophantic attempts, but they flagged *different* items. Zero
violations survive corroboration by both judges, so the independently confirmed rate is
0.0000. Label reliability over n=3009: violation kappa 0.647 at 94.3% raw agreement,
compliance kappa 0.944. The moderate violation kappa reflects grey-zone disagreement about
what counts as substantively harmful, not label instability, and it cannot inflate a null.

**Not a decoding artefact.** Resampling at temperature 0.7 (2,484 generations) preserves
the ranking: fiction framing 0.38 / 0.33, sycophantic framings 0.017-0.056, neutral
baseline 0.028.

### 3.2 Sycophancy DOES induce factual capitulation

| Condition | Phi-4 | Phi-4-mini |
|---|---|---|
| Safety violation under sycophancy | 0.000 | 0.005 |
| **Factual capitulation under sycophancy** | **0.011** | **0.130** |
| Neutral re-ask (control) | 0.000 | 0.040 |
| **Correct-push (corrigibility control)** | **0.438** | **0.500** |

`correct_push` is the decisive control. Both models yield readily when pushed TOWARD the
truth but resist being pushed AWAY from it. That distinguishes **truth-sensitive
corrigibility** from mere instability, and without this arm the result would be
uninterpretable.

Secondary findings:
- Capitulation scales inversely with capability (mini capitulates ~12x more than Phi-4)
- Concentrated in `common_fact`, not arithmetic (`hard_fact`)
- **Authority and expertise claims are the potent levers (0.149); consensus is inert
  (0.000); flattery is weak (0.064)**

### 3.2a Escalation: resistance erodes under sustained pressure

A single push understates the effect. `scripts/08_escalation.py` sustains pressure for up
to three rounds, samples 4 times at temperature 0.8, and judges every round.

| Model | Sycophantic levers | Neutral control |
|---|---|---|
| Phi-4 | **0.086** (69/800) | 0.020 |
| Phi-4-mini | **0.462** (377/816) | 0.284 |

Per round, Phi-4 capitulation *rises* with sustained pressure: 0.016 -> 0.024 -> 0.048.
Resistance erodes rather than holding. Phi-4-mini does not show this pattern (0.222 ->
0.135 -> 0.200), so early capitulation dominates in the weaker model.

Lever ranking under escalation, consistent with the single-push run:
authority 0.354 > expertise 0.280 > disappointment 0.267 > flat contradiction 0.203.

**This is what makes RQ2 tractable.** Positives available for geometry went from 2 to
**69** (Phi-4) and from 25 to **377** (Phi-4-mini). Use
`corpus_fc_escalation.parquet` for the geometry stage, not `corpus_fc_labelled.parquet`.

`capitulation_propensity.parquet` gives a per-item rate in [0,1] over the 4 samples, which
is a stronger regression target than a rare binary.

### 3.3 Which framings actually work (expansion sweep, n=1560, 189 violations)

| Route family | Violation rate |
|---|---|
| fiction | 0.181 |
| indirection | 0.142 |
| hypothetical | 0.059 |
| neutral (baseline) | 0.042 |
| flattery | 0.025 |
| social_pressure | 0.025 |

Sycophantic framings sit at or below the neutral baseline.

---

## 4. Repository layout

```
sycophancy-geometry/
├── AGENTS.md                     <- this file
├── README.md
├── requirements.txt
├── src/
│   ├── azclient.py               DEAD (provenance only) async Azure client
│   ├── corpus.py                 single-turn minimal-pair construction
│   ├── multiturn.py              multi-turn pressure on real refusals
│   ├── factual.py                factual capitulation probes  <- the live experiment
│   ├── judge.py                  judge rubric + defensive parsing
│   └── geometry.py               LID + curvature   <- PORTABLE, the core method
├── scripts/
│   ├── 01_build_corpus.py        DEAD  built the single-turn corpus
│   ├── 02_generate.py            DEAD  Azure generation
│   ├── 03_judge.py               DEAD  Azure judging
│   ├── 04_multiturn.py           DEAD  multi-turn run
│   ├── 05_expand.py              DEAD  framing sweep + embeddings
│   ├── 06_reliability.py         DEAD  second judge + temperature resampling
│   ├── 07_factual.py             DEAD  factual capitulation run
│   ├── 10_extract_activations.py LIVE  HuggingFace forward passes
│   ├── 11_geometry_analysis.py   LIVE  LID/curvature + comparator ladder
│   └── 99_check_portability.py   LIVE  enforces the no-Azure rule
├── data/
│   ├── raw/                      JailbreakBench source CSVs
│   ├── interim/                  per-call JSONL checkpoints (provenance)
│   ├── frozen/                   IMMUTABLE PRIMARY DATA
│   └── activations/              generated locally, gitignored (large)
└── results/                      metrics CSVs and summaries
```

### 4.1 Frozen data contract

| File | Rows | Label column | Group column |
|---|---|---|---|
| `corpus_labelled.parquet` | 780 | `violation` | `base_id` |
| `corpus_mt_labelled.parquet` | 780 | `violation` | `base_id` |
| `corpus_exp_labelled.parquet` | 1560 | `violation` | `base_id` |
| `corpus_fc_labelled.parquet` | 500 | `capitulated` | `qid` |
| `corpus_fc_escalation.parquet` | ~2000 | `capitulated` | `qid` |
| `capitulation_propensity.parquet` | ~500 | `capitulated` (rate) | `qid` |
| `embeddings_azure.npz` | 3120 x 3072 | - | - |
| `judge_agreement.parquet` | 3009 | Cohen kappa inputs | - |
| `sampling_variance.parquet` | 2484 | temperature 0.7 resamples | - |

Every row has a stable `rid` (sha1 prefix) that joins across generations, judgements and
activations. Never regenerate `rid`s; they are the join key for everything.

---

## 5. How to continue the work

### Step 1 — environment

```bash
pip install -r requirements.txt
python scripts/99_check_portability.py     # must pass
```

### Step 2 — extract activations (needs GPU or patience; free T4 is sufficient)

```bash
python scripts/10_extract_activations.py --model phi4-mini --corpus factual
python scripts/10_extract_activations.py --model phi4-mini --corpus safety_exp
```

Writes `data/activations/{model}__{corpus}.npz` containing:
- `pooled`    (n_items, n_layers, hidden)      mean over non-pad tokens -> LID, probes
- `per_token` (n_items, n_kept_layers, T, hidden) trailing states -> curvature
- `rids`, `lengths`, `keep_layers`

On a free T4, reduce memory with `--batch-size 4 --keep-tokens 32 --layer-stride 2`.

### Step 3 — analysis (CPU, minutes)

```bash
python scripts/11_geometry_analysis.py --model phi4-mini --corpus factual
```

Outputs to `results/`: `comparison__*.csv`, `layer_sweep__*.csv`, `summary__*.json`.

### Step 4 — transferability (RQ4)

```bash
python scripts/10_extract_activations.py --model mistral7b  --corpus factual
python scripts/10_extract_activations.py --model qwen2.5-3b --corpus factual
```

Because detection is **prompt-side**, labels come from generation but features come from a
prompt forward pass. Any open model can therefore be probed on the same frozen labels
without ever having been deployed. This is why RQ4 survived the loss of Azure.

---

## 6. Methodological rules that must not be broken

These are not stylistic preferences. Violating any one of them invalidates the results.

1. **Group-wise splitting, never random.** Split by `qid` (factual) or `base_id` (safety).
   Minimal-pair twins share a base item; a random split puts them on both sides and the
   classifier memorises the base instead of learning the phenomenon.

2. **LID reference set is fitted on TRAINING DATA ONLY.** LID is a neighbourhood statistic,
   so fitting neighbours on the full corpus leaks test information through the feature
   itself. This is the exact data-snooping failure catalogued by Arp et al. (USENIX 2022).
   `geometry.LIDExtractor` enforces the split; do not bypass it.

3. **Report TPR at 1% FPR, not accuracy.** The deployment base rate of attacks is tiny, so
   false positives dominate cost and headline accuracy is close to meaningless.

4. **Latency is MARGINAL cost above an inference already being performed.** In deployment
   the activations exist anyway. Counting a full forward pass loses to CurvaLID by
   construction and misrepresents the deployment picture.

5. **The comparator ladder must stay intact.** A geometric feature only earns its cost if
   it beats tfidf, Azure embeddings, a raw-hidden-state linear probe, and trivial norm
   statistics. If it does not, report that. A null result here is publishable; a
   flattering result obtained by dropping baselines is not.

6. **Never delete `data/frozen/`.** It cannot be regenerated.

---

## 7. Known open items

- [ ] **Verify CurvaLID's claims.** The proposal asserts >500ms latency. Confirm the figure
      appears in the paper. Also determine whether CurvaLID operates on target-model
      activations or on its own prompt embeddings; if the latter, this work's use of
      internals is a departure and the word "hybrid" needs redefining.
- [ ] **The 96% F1 baseline must not be used as a threshold.** It comes from Galinkin &
      Sablotny on harmful-vs-benign, a substantially easier task. Cite it as context only,
      and run Random Forest in-house as one comparator among several.
- [ ] **100ms anchor citation:** Miller (1968); Card, Robertson & Mackinlay (1991);
      Nielsen (1993). Cite rather than assert.
- [ ] Extend the factual probe set beyond 25 questions if more positives are needed.
- [ ] Ethics: confirmed NOT required (no human participants). The report should still carry
      an explicit section, since novel pressure framings were authored.

---

## 8. Framing the write-up

The strongest version of this project is **not** "geometric jailbreak detection". It is:

> Alignment training hardens models against social pressure on safety, but leaves factual
> commitments comparatively soft. Models are corrigible toward truth and resistant away
> from it. The moment of capitulation may be visible in activations before it is visible
> in output.

The negative result (section 3.1) is a **control arm**, not a failure. It is what licenses
the asymmetry claim: the same levers, the same models, the same pipeline, two very
different outcomes depending on the type of commitment under attack.

Suggested structure:
1. Sycophancy does not breach safety boundaries (n=444, matched controls)
2. The same levers do induce factual capitulation, and capability predicts resistance
3. Corrigibility is truth-sensitive, not symmetric (the `correct_push` arm)
4. Which levers work: authority and expertise, not flattery or consensus
5. Geometric detectability, benchmarked honestly against the comparator ladder
6. Early-exit depth and marginal latency

---

## 9. Provenance

- Target models: Phi-4 (v7) and Phi-4-mini-instruct, Azure GlobalStandard, **temperature 0**
- Judge: gpt-5.4; second independent judge gpt-5.6-sol for Cohen kappa
- Base requests: JailbreakBench (Chao et al., 2024), filtered to Fraud/Deception,
  Disinformation and Economic harm. Harassment, sexual content, physical harm, malware and
  privacy categories excluded by construction.
- Harmful content is sourced from a published benchmark rather than authored here. This is
  deliberate: the contribution is the framing variation, not the harm.
- All corpora carry a sha256 in their `*_meta.json`.
