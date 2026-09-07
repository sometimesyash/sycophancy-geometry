# Sycophancy Geometry

Do language models abandon positions under social pressure, and is that abandonment
visible in their activations before they respond?

## The finding

Sycophantic pressure does **not** breach safety boundaries. Across 444 attempts on Phi-4
with matched controls, the violation rate was 0.000. The same pipeline detects fiction
framing succeeding at 0.68, so this is a measurement rather than a broken harness.

The same levers **do** induce factual capitulation, and the picture is asymmetric:

| Condition | Phi-4 | Phi-4-mini |
|---|---|---|
| Safety violation under sycophancy | 0.000 | 0.005 |
| Factual capitulation under sycophancy | 0.011 | 0.130 |
| Neutral re-ask (control) | 0.000 | 0.040 |
| **Correct-push (corrigibility control)** | **0.438** | **0.500** |

Both models yield readily when pushed *toward* the truth and resist being pushed *away*
from it. That is truth-sensitive corrigibility, not instability, and the `correct_push`
arm is what makes the distinction measurable.

The null is independently corroborated: two judges each flagged exactly one violation
across 644 sycophantic attempts, but flagged *different* items, so zero survive
agreement by both. Resampling at temperature 0.7 preserves the ranking, so it is not an
artefact of greedy decoding.

> Alignment training appears to produce an asymmetrically robust boundary: hard on harm,
> soft on truth.

Which levers work also turns out to be specific. Authority and expertise claims carry the
effect (0.149); consensus is completely inert (0.000) and flattery is weak (0.064).

## Quick start

```bash
pip install -r requirements.txt
python scripts/99_check_portability.py

python scripts/10_extract_activations.py --model phi4-mini --corpus factual
python scripts/11_geometry_analysis.py   --model phi4-mini --corpus factual
```

Phi-4-mini is 3.8B and fits a free Kaggle or Colab T4, so no paid compute is required.

## Structure

Scripts `01`-`07` produced the frozen corpora using Azure-hosted inference. **That
subscription no longer exists and those scripts cannot be re-run**; they are kept for
provenance. Scripts `10`+ are pure functions of `data/frozen/` and run on open weights
indefinitely. `99_check_portability.py` enforces that boundary.

See [AGENTS.md](AGENTS.md) for the full operating manual, methodological rules, and the
reasoning behind the pivot.

## Data

Base requests come from [JailbreakBench](https://jailbreakbench.github.io/) (Chao et al.,
2024), filtered to Fraud/Deception, Disinformation and Economic harm. Harassment, sexual
content, physical harm, malware and privacy categories are excluded by construction.
Harmful content is sourced from a published benchmark rather than authored here: the
contribution is the framing variation, not the harm.

All generation ran at temperature 0. Every corpus carries a sha256 in its `*_meta.json`.
