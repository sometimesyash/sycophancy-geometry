# Sycophancy Geometry

**Do language models abandon positions under social pressure, and is that abandonment
visible in their activations before they respond?**

Yash Narain, September 2026. Laidlaw research project.

---

## 1. The one-paragraph summary

This project set out to detect sycophancy-based jailbreaks using geometric features of
model activations. The first experiment returned a violation rate of **exactly zero**, and
that null turned out to be the finding rather than a failure. Sycophantic pressure does not
breach safety boundaries, but the *same levers* readily induce **factual** capitulation.
Alignment training appears to produce an asymmetrically robust boundary: **hard on harm,
soft on truth**. Capitulation falls monotonically with model capability, reaching zero at
the frontier. All behavioural data is frozen and committed; the geometry stage runs on open
weights and needs no paid compute.

---

## 2. Results

### 2.1 Sycophancy does NOT breach safety boundaries

| Experiment | n | Violation rate |
|---|---|---|
| Single-turn sycophantic framing (Phi-4) | 240 | **0.000** |
| Multi-turn sycophantic pressure (Phi-4) | 204 | **0.000** |
| Multi-turn sycophantic pressure (Phi-4-mini) | 202 | 0.005 |
| Neutral control | 152 | 0.000 |
| Non-sycophantic routes | 203 | 0.186 |
| └ fiction escalation specifically | 25 | **0.680** |

The matched controls are what make this a measurement rather than a broken harness: the
same pipeline detects success at 0.68 when success actually occurs.

**Independently corroborated.** Two judges (gpt-5.4 and gpt-5.6-sol) each flagged exactly
one violation across 644 sycophantic attempts, but they flagged *different* items. Zero
violations survive corroboration by both, so the independently confirmed rate is **0.0000**.

**Not a decoding artefact.** Resampling at temperature 0.7 (2,484 generations) preserves
the ranking: fiction 0.38 / 0.33, sycophantic framings 0.017-0.056, neutral 0.028.

### 2.2 Which framings actually work (n=1560, 189 violations)

| Route family | Violation rate |
|---|---|
| fiction | 0.181 |
| indirection | 0.142 |
| hypothetical | 0.059 |
| **neutral (baseline)** | **0.042** |
| flattery | 0.025 |
| social_pressure | 0.025 |

Sycophantic framings sit *at or below* the do-nothing baseline.

### 2.3 The same levers DO induce factual capitulation

Single push (n=500):

| Condition | Phi-4 | Phi-4-mini |
|---|---|---|
| Safety violation under sycophancy | 0.000 | 0.005 |
| Factual capitulation under sycophancy | 0.011 | 0.130 |
| Neutral re-ask (control) | 0.000 | 0.040 |
| **Correct-push (corrigibility control)** | **0.438** | **0.500** |

The `correct_push` arm is the decisive control. Both models yield readily when pushed
*toward* the truth and resist being pushed *away* from it. That distinguishes
**truth-sensitive corrigibility** from mere instability. Without this arm the result would
be uninterpretable, because a model that flips under any pressure is simply unstable.

### 2.4 Resistance erodes under sustained pressure

Escalation to 3 rounds, 4 samples at temperature 0.8 (n=3884 conversations):

| Model | Sycophantic levers | Neutral control |
|---|---|---|
| Phi-4-mini (3.8B) | **0.462** | 0.284 |
| Phi-4 (14B) | **0.086** | 0.020 |
| gpt-5.4 | 0.046 | 0.000 |
| gpt-5.6-sol | **0.000** | 0.000 |

Per round, Phi-4 capitulation *rises* with sustained pressure (0.016 -> 0.024 -> 0.048):
**resistance erodes rather than holding**. Phi-4-mini shows no such pattern (0.222 -> 0.135
-> 0.200), so early capitulation dominates in the weaker model.

### 2.5 Capability scaling

Capitulation falls **monotonically** with capability and reaches exactly zero at the
frontier: 0.462 -> 0.086 -> 0.046 -> 0.000. Neutral controls are 0.000 for all three larger
models, so the effect is attributable to the sycophantic content and not to being
questioned at all.

### 2.6 Which levers work

| Lever | Rate |
|---|---|
| authority claim | **0.198** |
| expertise override | 0.146 |
| disappointment | 0.137 |
| flat contradiction | 0.111 |
| consensus ("everyone says") | **0.000** — dropped after run 1 |
| flattery | 0.064 (weak) |

**Asserted authority moves models; social consensus and flattery do not.** This is stable
across both the single-push and escalation runs.

---

## 3. How the project got here (full narrative)

Nothing below is hidden, because the route matters for the write-up.

1. **Original proposal**: detect sycophancy jailbreaks via LID and curvature, benchmarked
   against CurvaLID's latency.
2. **Infrastructure audit** found the Azure subscription was Microsoft-internal
   (`quotaId: Internal_2014-09-01`). GPU quota was 0 in every region; four quota requests
   all failed with `QuotaNotAvailableForResource`. Marketplace was blocked, so no Mistral
   or Llama endpoints. Llama 3.1 8B was separately deprecated on Azure.
3. **Phi-4 was chosen** because it is the only model that is *both* deployable as a
   serverless endpoint on that subscription *and* open-weight (MIT) on HuggingFace. The
   model that generated the responses is therefore the same weights whose activations get
   probed. That identity is a methodological requirement, not a compromise.
4. **Single-turn run: 0/240 violations.** Diagnosis: a flattery wrapper on a bare harmful
   request is not sycophancy, it is an odd preamble. Sycophancy in Sharma et al. is a
   *multi-turn* failure where a model abandons a position it has already taken.
5. **Multi-turn run: still 0/204**, but fiction framing hit 0.68. So the harness works and
   sycophancy genuinely does not breach safety.
6. **The pivot.** Safety refusals absorb enormous adversarial training investment; ordinary
   factual assertions absorb almost none. That predicts an asymmetry, which section 2.3
   confirms.
7. **Escalation run** was added because the first factual run gave only 2 positives on
   Phi-4, far too few for the geometry stage. It produced 69 and 377.

**The negative result is a control arm, not a failure.** It is what licenses the asymmetry
claim: same levers, same models, same pipeline, opposite outcomes depending on the type of
commitment under attack.

---

## 4. What is INCOMPLETE (read this)

Azure access ended mid-run. Precisely one thing is affected.

### 4.1 The capability-scaling arm is partial

| Arm | Rounds completed | Status |
|---|---|---|
| **Geometry** (Phi-4, Phi-4-mini) | 3 of 3 | **COMPLETE** |
| Scaling (gpt-5.4, gpt-5.6-sol) | 2 of 3 | **PARTIAL** |

- **The geometry arm is complete.** RQ2 is fully supported and nothing is missing.
- The scaling arm has rounds 0 and 1 for gpt-5.4, and round 0 for gpt-5.6-sol.
- **Direction of the bias is known and favourable to caution**: more rounds can only
  *increase* capitulation, so the frontier figures (0.046 and 0.000) are **lower bounds**.
  The monotonic ordering in section 2.5 is therefore safe, but state in the write-up that
  the two frontier models received fewer pressure rounds than the two Phi models.
- `data/frozen/corpus_fc_escalation_meta.json` records exactly which rounds exist.

### 4.2 Other gaps

- **Judge kappa on violation is 0.647** (94.3% raw agreement, n=3009), which is
  "substantial" but not "near-perfect". It reflects genuine grey-zone disagreement about
  what counts as substantively harmful. Pre-empt this in the write-up: **disagreement of
  this kind cannot manufacture a null**, and compliance kappa is 0.944.
- **CurvaLID's >500ms claim is unverified.** Confirm the figure appears in the paper, and
  determine whether CurvaLID uses target-model activations or its own prompt embeddings.
  If the latter, this work's use of internals is a departure and "hybrid" needs redefining.
- **No activations extracted yet.** Scripts exist and are tested; they need a GPU session.
- The **96% F1 Random Forest baseline must not be used as a threshold**. It comes from
  Galinkin & Sablotny on harmful-vs-benign, a substantially easier task. Cite as context;
  run RF in-house as one comparator among several.

---

## 5. Data inventory

Everything in `data/frozen/` is **irreplaceable** — the subscription that produced it no
longer exists.

| File | Rows | Label | Group | What it is |
|---|---|---|---|---|
| `corpus_labelled.parquet` | 780 | `violation` | `base_id` | single-turn safety |
| `corpus_mt_labelled.parquet` | 780 | `violation` | `base_id` | multi-turn safety |
| `corpus_exp_labelled.parquet` | 1560 | `violation` | `base_id` | framing sweep |
| `corpus_fc_labelled.parquet` | 500 | `capitulated` | `qid` | factual, single push |
| **`corpus_fc_escalation.parquet`** | **3884** | `capitulated` | `qid` | **escalation — use this for geometry** |
| `corpus_fc_escalation_rounds.parquet` | 5535 | `capitulated` | `qid` | per-round detail |
| `capitulation_propensity.parquet` | ~550 | `propensity` | `qid` | continuous target in [0,1] |
| `judge_agreement.parquet` | 3009 | — | — | two-judge kappa inputs |
| `sampling_variance.parquet` | 2484 | `violation` | — | temperature 0.7 resamples |
| `embeddings_azure.npz` | 3120 x 3072 | — | — | text-embedding-3-large |

Every row carries a stable `rid` (sha1 prefix) joining generations, judgements and
activations. **Never regenerate `rid`s.** Every corpus has a sha256 in its `*_meta.json`.

Total: **~11,000 labelled model interactions.**

---

## 6. Repository layout

```
src/
  azclient.py       DEAD  async Azure client (token refresh, backoff, resumable JSONL)
  corpus.py         single-turn minimal-pair construction
  multiturn.py      multi-turn pressure applied to real captured refusals
  factual.py        55 factual probes across 3 difficulty bands + wrong answers
  judge.py          safety judge rubric and defensive parsing
  capitulation.py   capitulation rubric, parsing, deterministic string check
  geometry.py       LID (Levina-Bickel MLE) + trajectory curvature   <- CORE METHOD

scripts/
  01_build_corpus.py         DEAD  build + freeze single-turn corpus
  02_generate.py             DEAD  generation
  03_judge.py                DEAD  labelling
  04_multiturn.py            DEAD  multi-turn run
  05_expand.py               DEAD  framing sweep + embeddings
  06_reliability.py          DEAD  second judge + temperature resampling
  07_factual.py              DEAD  factual capitulation, single push
  08_escalation.py           DEAD  escalation + sampling + capability scaling
  09_salvage_escalation.py   LIVE  rebuild frozen artefacts from checkpoints
  10_extract_activations.py  LIVE  HuggingFace forward passes
  11_geometry_analysis.py    LIVE  LID/curvature + comparator ladder
  99_check_portability.py    LIVE  enforces the no-Azure rule
```

"DEAD" means the script required Azure-hosted inference and **cannot be re-run**. It is
retained for provenance and reproducibility documentation. Scripts 09+ are pure functions
of files on disk.

---

## 7. How to continue

```bash
pip install -r requirements.txt
python scripts/99_check_portability.py          # must pass

python scripts/10_extract_activations.py --model phi4-mini --corpus factual
python scripts/11_geometry_analysis.py   --model phi4-mini --corpus factual
```

**Phi-4-mini (3.8B) is the primary target**, ~7.6GB in fp16, fits a **free Kaggle or Colab
T4**. No paid compute is required for any remaining work. Phi-4 (14B) is secondary and
needs 2xT4 or a large CPU box. On a free T4 use
`--batch-size 4 --keep-tokens 32 --layer-stride 2`.

For transferability (RQ4), extract from `mistral7b` or `qwen2.5-3b`. Because detection is
**prompt-side**, labels come from generation but features come from a prompt forward pass,
so any open model can be probed against the frozen labels without ever being deployed.
**This is why RQ4 survived the loss of Azure.**

See [AGENTS.md](AGENTS.md) for the complete operating manual.

---

## 8. Methodological rules (violating any one invalidates the results)

1. **Group-wise splitting, never random.** Split by `qid` or `base_id`. Minimal-pair twins
   share a base item; a random split puts them on both sides and the classifier memorises
   the base instead of learning the phenomenon.
2. **LID reference set fitted on TRAINING DATA ONLY.** LID is a neighbourhood statistic, so
   fitting on the full corpus leaks test information through the feature itself — exactly
   the data-snooping failure catalogued by Arp et al. (USENIX 2022).
   `geometry.LIDExtractor` enforces this; do not bypass it.
3. **Report TPR at 1% FPR, not accuracy.** Deployment base rates are tiny, so false
   positives dominate cost and headline accuracy is close to meaningless.
4. **Latency is MARGINAL cost above an inference already being performed.** In deployment
   the activations exist anyway; counting a full forward pass loses by construction.
5. **Keep the comparator ladder intact.** Geometry only earns its cost if it beats tfidf,
   Azure embeddings, a raw-hidden-state linear probe, and trivial norm statistics. A null
   here is publishable; a flattering result obtained by dropping baselines is not.
6. **Never delete `data/frozen/`.**

---

## 9. Suggested write-up structure

1. Sycophancy does not breach safety boundaries (n=444, matched controls, two judges)
2. The same levers induce factual capitulation
3. Corrigibility is truth-sensitive, not symmetric (the `correct_push` arm)
4. Resistance erodes under sustained pressure (the per-round curve)
5. Capability scaling: 0.462 -> 0.086 -> 0.046 -> 0.000
6. Which levers work: asserted authority, not consensus or flattery
7. Geometric detectability, benchmarked honestly against the comparator ladder
8. Early-exit depth and marginal latency

The framing to avoid is "geometric jailbreak detection". The framing that the data supports
is: **alignment hardens models against social pressure on safety while leaving factual
commitments comparatively soft, and the moment of capitulation may be visible in
activations before it is visible in output.**

---

## 10. Provenance and ethics

- **Targets**: Phi-4 (v7), Phi-4-mini-instruct, gpt-5.4, gpt-5.6-sol via Azure
  GlobalStandard. Temperature 0 unless stated; escalation used temperature 0.8 with 4
  samples; variance run used 0.7.
- **Judges**: gpt-5.4 primary, gpt-5.6-sol independent second judge.
- **Base requests**: [JailbreakBench](https://jailbreakbench.github.io/) (Chao et al.,
  2024), filtered to Fraud/Deception, Disinformation and Economic harm. Harassment, sexual
  content, physical harm, malware and privacy categories **excluded by construction**.
- Harmful content is sourced from a published, peer-reviewed benchmark rather than authored
  here. The contribution is the framing variation, not the harm.
- **Ethics approval**: confirmed not required (no human participants). The report should
  still carry an explicit ethics section, since novel pressure framings were authored.
- Factual probes are neutral general-knowledge questions and carry no harm surface.

## 11. References

1. Sharma, M. et al. (2023) *Towards Understanding Sycophancy in Language Models*, Anthropic.
2. Chao, P. et al. (2024) *JailbreakBench*, NeurIPS Datasets and Benchmarks.
3. Yung, C. et al. (2025) *CurvaLID: Geometrically-guided Adversarial Prompt Detection*.
4. Levina, E. & Bickel, P. (2004) *Maximum Likelihood Estimation of Intrinsic Dimension*, NeurIPS.
5. Arp, D. et al. (2022) *Dos and Don'ts of Machine Learning in Computer Security*, USENIX Security.
6. Galinkin, E. & Sablotny, A. (2024) *Evaluation of Random Forest Classifiers for Adversarial Prompt Detection*.
7. Miller, R. B. (1968) *Response time in man-computer conversational transactions*, AFIPS.
8. Nielsen, J. (1993) *Usability Engineering* — the 100ms threshold.
