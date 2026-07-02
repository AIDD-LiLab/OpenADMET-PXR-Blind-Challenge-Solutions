# OpenADMET PXR (NR1I2) Challenge — Aggressive Variant (full report)

**Team:** AIDD-LiLab (University of Florida — Yanjun Li Lab)
**Base:** the Phase-1 winning `MoE_v2 + MOE-multikernel` ensemble (see [`../main solution/`](../main%20solution/)).
**Variant file:** `SUBMIT_AGGR_SVMHolo3way440_qHTS.csv` (513 rows).

The aggressive variant keeps the winning architecture and every ensemble weight **unchanged**, and alters only two things:
1. **three components are retrained** with **440 additional high-activity molecules** (data change only — same models, folds, hyper-parameters), and
2. an **external qHTS discriminator** post-hoc down-shift is applied to likely-inactive low predictions.

It is offered as a **deliberate, quantified bet on the hidden Set 2** — not as a claimed improvement on the reliable Set-1 validator.

---

## 0. Result (Set 1 unblinded, official metric)

| Prediction | Set 1 (253) RAE | MAE | R² | Spearman ρ |
|---|---:|---:|---:|---:|
| Base system (winning submission) | 0.4875 | 0.3893 | 0.669 | 0.861 |
| **Aggressive variant** (SVM+Holo+3way `+440`, + qHTS) | **0.4841** | 0.3866 | 0.674 | 0.856 |
| Conservative reference (SVM `+440` only, + qHTS) | 0.4779 | 0.3817 | 0.682 | 0.861 |

Held-out **V** (51 scaffold-disjoint Set-1 molecules, never in any training set): aggressive **0.348** vs base **0.371**.

---

## 1. Why an aggressive variant

The final ranking is decided by **Set 2 (260, hidden)**, not by the live combined-513 board. Two facts make an aggressive, Set-2-targeted variant rational:

- **Set 1 is a weak predictor of Set 2 at our precision.** Set 1 ↔ Set 2 median nearest-neighbour Tanimoto ≈ 0.26 (a real scaffold shift; 0/260 Set-2 molecules have a Set-1 analogue above 0.6). Coarse strong-vs-weak ranking transfers, but in the top-tier fine-grained regime where leaders sit, the Set-1 → Set-2 rank correlation collapses. So a submission that is neutral-to-slightly-worse on Set 1 is **not** evidence of harm on Set 2.
- **Set 2 is active-enriched, and 440 new high-activity molecules were released.** The base ensemble mildly under-predicts the active tail (Set-1 pEC50>5 bias −0.15) and over-predicts inactives (pEC50<4 bias +0.56). Extra high-activity training data plus a low-activity discriminator directly target the two ends that dominate Set 2.

---

## 2. The `+440` data

440 additional high-activity molecules (corrected pEC50; released after Phase 1) that are **disjoint** from the original 4,139 train and from the 253 Set-1 test. They are chemically distant from Set 2 (only ~5% of Set-2 molecules are closer to a `+440` molecule than to the base train), so the bet is not "coverage" but "more high-activity signal for the active-region heads."

---

## 3. Method

### 3.1 Same-architecture retraining ("only data changes")
Three components are retrained on `train (4,139) + 440` with the **identical** model, 5-fold scaffold split, and hyper-parameters as the winning system. Everything else (weights, other components, MoE + multi-kernel layers, variance scale) is byte-for-byte the production system.

### 3.2 Component selection — which retrains help, and why
Each component's `+440` retrain was injected individually and scored on the reliable Set-1 (253) **and** on the held-out V (51, the active-region proxy). Selection excludes any component that clearly degrades either.

| Component | ensemble weight | Set-1 Δ | V Δ | decision |
|---|---:|---:|---:|---|
| **SVM_20K** | 0.125 | +0.0001 | −0.0019 | **include** (neutral on Set-1, helps V) |
| **Holo** | high (cofold pose pillar) | +0.0018 | −0.0059 | **include** (most active-region-relevant) |
| **3-way (sw3way)** | 0.05 | +0.0003 | +0.0007 | **include** (neutral, for coverage) |
| Gated_5mod | 0.125 | +0.0013 | +0.0071 | exclude (degrades V) |
| ConfTTA | highest | +0.0087 | +0.0059 | **exclude** (largest weight amplifies an unhelpful `+440` effect) |

The reason SVM looks "safe" is partly that its ensemble weight is small, so its `+440` delta is muted; Holo carries the `+440` signal most strongly because it is the high-weight, binding-pose pillar — hence it is the core of the active-region bet.

### 3.3 Clean within-pipeline delta injection
For each retrained component the effect is applied as a **within-pipeline difference**

```
Δ = ensemble(component = +440-retrained) − ensemble(component = base-retrained)
final_variant = winning_prediction + Σ Δ_component        (then qHTS, §3.4)
```

Because both terms come from the *same* reproduction pipeline, the pipeline-reproduction offset cancels and only the **pure +440 data effect** is injected onto the frozen winning prediction. When the retrain data equals the original data, Δ=0 and the variant is byte-identical to the winning submission — so the baseline is exact and only the data change is expressed.

### 3.4 qHTS discriminator (external, leak-free)
An ECFP4 gradient-boosted classifier trained on the **external NCATS PXR qHTS assay** (PubChem AID 1346982; ~9,700 compounds, canonical-SMILES-disjoint from train and test) predicts P(active). Molecules the base model calls inactive (pred < 4) **and** the external classifier flags inactive (P_active < 0.3) are shifted down by 0.2 pEC50. This corrects the model's systematic over-prediction of inactives (Set-1 pEC50<4 bias +0.56 → pulled toward truth) using an orthogonal external signal, and is the robust, transferable part of the variant's gain.

---

## 4. Validation (three levels + directional)

- **Set 1, 253 (reliable):** aggressive 0.4841 vs base 0.4875 (−0.0034). Nearly all of this is the qHTS discriminator; the Holo/3way `+440` retrains are **within the noise floor** (bootstrap MAE σ ≈ 0.028 on 253 points; the aggressive variant is +0.0062 RAE above the conservative SVM-only+qHTS reference).
- **Set 1 minus V, 202 (independent):** the Holo/3way `+440` contribution is ≈ neutral (0.517 vs base 0.516) — i.e. on a bigger honest sample the retrain effect is not a clear gain.
- **Held-out V, 51 (active-region proxy):** aggressive 0.348 vs base 0.371 — the largest apparent gain, but on the smallest (noisiest) sample.
- **Directional check:** the `+440` retrain does not consistently move high-activity predictions toward truth on Set 1 — evidence that the V gain is at least partly small-sample noise, not a verified active-region correction.

**Honest reading:** the qHTS discriminator is a robust, externally-grounded improvement; the `+440` Holo/3way retrain is a within-noise, active-region-targeted **bet** whose payoff depends on the Set-1 → Set-2 shift making the reliable Set-1 signal uninformative for the active tail.

---

## 5. When to prefer which

- **Conservative** (`SVM +440 + qHTS`, Set-1 0.4779) — best on the reliable validator; the safe choice.
- **Aggressive** (this variant, `SVM+Holo+3way +440 + qHTS`, Set-1 0.4841) — accepts a small, quantified Set-1 cost (+0.0062 vs conservative) to express more high-activity `+440` signal through the binding-pose pillar, as a Set-2 bet.

---

## 6. Files

- `SUBMIT_AGGR_SVMHolo3way440_qHTS.csv` — the aggressive variant prediction (513 rows).

*Conservative and base predictions, the retraining/injection code, and the qHTS builder accompany this report in the repository.*
