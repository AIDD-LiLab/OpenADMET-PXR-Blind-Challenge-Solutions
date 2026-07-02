# OpenADMET PXR (NR1I2) Challenge — Aggressive Variant Report

**Team:** AIDD-LiLab (University of Florida — Yanjun Li Lab)
**Base system:** the Phase-1 winning `MoE_v2 + MOE-multikernel` ensemble (see `../main solution/`).
**Variant prediction file:** `SUBMIT_AGGR_SVMHolo3way440_qHTS.csv` (513 rows).

This variant is a deliberate, higher-variance bet on the **hidden Set 2**. It keeps the winning architecture and weights **unchanged**, and changes only two things: (i) three components are **retrained with 440 additional high-activity molecules**, and (ii) an external **qHTS discriminator** post-hoc down-shift is applied. It is *not* claimed to beat the base system on the reliable validator; it is offered as an active-region-targeted alternative.

---

## 1. Motivation

Two facts drove the variant:

1. **Set 1 → Set 2 is a chemistry shift.** Set 1 (253, now unblinded) and Set 2 (260, hidden final scorer) have median pairwise Tanimoto ≈ 0.26; on the fine-grained top-tier ranking where the leaders sit, Set-1 RAE is a poor predictor of Set-2 RAE. A submission that is neutral-to-slightly-worse on Set 1 can still differ on Set 2.
2. **Set 2 is active-enriched, and 440 new high-activity molecules were released.** Because the base ensemble mildly under-predicts the active tail (Set 1 pEC50>5: bias −0.15), extra high-activity training data plausibly helps the active region that dominates Set 2.

---

## 2. Method (only data changes; architecture identical)

**Retraining (same pipeline, +440 high-activity data, original ensemble ratio).**
Three components are retrained on `train (4,139) + 440` high-activity molecules, using the identical model, folds, and hyper-parameters as the base system:

- **SVM_20K** (embedding multi-kernel SVR) — neutral on Set 1, helps the active-region proxy.
- **Holo** (Uni-Mol cofold binding-pose) — the most active-region-relevant component (binding pose × high-activity), kept at its original ensemble weight.
- **sw3way** (3-way frozen-pretrain SVR) — neutral, included for coverage.

Components deliberately **excluded**: **ConfTTA** (its +440 retrain clearly degrades the reliable validator — it carries the largest weight, which amplifies an unhelpful data effect) and **Gated_5mod** (Set-1-neutral but degrades the active-region proxy).

The retrained components are injected as a **clean within-pipeline delta** — `Δ = ensemble(component=+440) − ensemble(component=base)` — added to the frozen winning prediction, so the pipeline-reproduction offset cancels and only the +440 data effect is applied.

**qHTS discriminator (external, leak-free).**
An ECFP4 classifier trained on the external NCATS PXR qHTS assay (PubChem AID 1346982; ~9,700 compounds, disjoint from train and test) predicts P(active). Molecules the base model calls inactive (pred < 4) **and** the external classifier flags inactive (P_active < 0.3) are shifted down by 0.2 pEC50 — correcting the model's systematic over-prediction of inactives (Set 1 pEC50<4: bias +0.56).

---

## 3. Validation (Set 1 unblinded, official RAE)

| Prediction | Set 1 (253) RAE | MAE | R² | Spearman ρ |
|---|---:|---:|---:|---:|
| Base system (`main solution`) | 0.4875 | 0.3893 | 0.669 | 0.861 |
| **Aggressive variant** (SVM+Holo+3way +440, + qHTS) | **0.4841** | 0.3866 | 0.674 | 0.856 |
| Conservative reference (SVM +440 only, + qHTS) | 0.4779 | 0.3817 | 0.682 | 0.861 |

- Held-out V (51 scaffold-disjoint Set-1 molecules, never in any training): aggressive variant RAE 0.348 vs base 0.371.
- The qHTS down-shift is the robust, transferable part of the gain (external assay; validated leak-free; corrects the +0.56 inactive-over-prediction bias toward truth).

---

## 4. Honest positioning (rigorous)

- The **conservative** SVM+440+qHTS reference (0.4779) is the best submission on the reliable 253-molecule validator; nearly all of its gain comes from the qHTS discriminator, with the +440 SVM retrain contributing ≈0 (it is a small-weight, near-neutral passenger).
- This **aggressive** variant additionally injects the **Holo/3way +440 retrains**. On the reliable Set-1 validator their net contribution is *within the noise floor* (bootstrap σ ≈ 0.028 MAE on 253 points; the aggressive variant is +0.0062 RAE above the conservative reference). On an independent 202-molecule split their effect is ≈neutral.
- The variant is therefore **not** a Set-1-validated improvement over the base system's low-activity correction alone. It is included as an **active-region-targeted bet**: if the Set-1 → Set-2 chemistry shift makes the reliable Set-1 signal uninformative for the active tail, the +440 high-activity retrain (especially the Holo binding-pose component) may help Set 2, at a small, quantified Set-1 cost.

---

## 5. Files

- `SUBMIT_AGGR_SVMHolo3way440_qHTS.csv` — the aggressive variant prediction (513 rows).

*Conservative and base predictions, and the retraining / injection code, accompany this report in the repository.*
