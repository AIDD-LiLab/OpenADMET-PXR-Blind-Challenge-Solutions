# OpenADMET PXR (NR1I2) Challenge — Aggressive Variant (+440 Retrain + qHTS)

**Team:** AIDD-LiLab — University of Florida (Yanjun Li Lab)
**Base:** the winning `MoE_v2 + MOE-multikernel` ensemble (see [`../main solution/README.md`](../main%20solution/README.md)).
**Variant file:** `SUBMIT_AGGR_SVMHolo3way440_qHTS.csv` (513 rows).
**Metric:** `RAE = Σ|ŷ−y| / Σ|y−ȳ|`.

This is a **deliberate, higher-variance bet on the hidden Set 2**. It keeps the winning architecture and weights unchanged and changes only two things:

1. **three components are retrained** with **+440 additional high-activity molecules** (a data change only — identical models, folds, hyper-parameters), injected as a within-pipeline delta onto the frozen winning prediction; and
2. the same external **qHTS discriminator** gate as the main solution.

---

## 0. Result — the bet, and how it resolved

Final challenge leaderboard (**Set 2**, 260 hidden molecules):

| Rank | Team | MAE | **RAE** | R² | Spearman ρ | Kendall τ | Proprietary data |
|---:|---|---:|---:|---:|---:|---:|:--:|
| 1 | matcha-croissant | 0.4061 | 0.5631 | 0.598 | 0.827 | 0.643 | Yes |
| 2 | **AIDD-LiLab** (main: base + qHTS) | 0.4092 | **0.5676** | 0.591 | 0.825 | 0.644 | No |
| 3 | **AIDD-LiLab-Aggressive** (this variant) | 0.4104 | **0.5692** | 0.589 | 0.822 | 0.641 | No |

**How the bet resolved (honestly).** The bet slightly lost: the aggressive variant finished **#3 overall, 0.0016 RAE behind our own main solution** on Set 2 — within the Set-1 noise floor (bootstrap σ ≈ 0.028), so the conservative frozen-ensemble choice was the right call. This is exactly what the pre-submission analysis predicted: the `+440` retrain was always a small, bounded-downside coin-flip, and it landed on the downside. Even so, **both AIDD-LiLab entries finished ahead of every other public-data team**.

We publish this variant in full because the *reasoning* was sound and the outcome is instructive: a well-constructed, honestly-quantified aggressive bet that lost by less than a third of the noise floor.

**Set-1 validator (unblinded, held-out):**

| Prediction | Set 1 (253) RAE | MAE | V(51) RAE |
|---|---:|---:|---:|
| Base ensemble (Phase-1 winner) | 0.4875 | 0.3893 | 0.3706 |
| Main solution (base + qHTS) | 0.4777 | 0.3815 | — |
| **Aggressive (this variant)** | 0.4841 | 0.3866 | 0.3479 |

*(V(51) = a 51-molecule cross-setting held-out subset, see §3; the **qHTS gate** = an external cheap-assay classifier applied only as post-processing to nudge likely-inactive low predictions down, see §5.)*

All Set-1 numbers are **fully held-out**: this variant trains on the `hi` setting (base + `+440`, containing **zero** Set-1 molecules — verified in §3), so all 253 Set-1 molecules are out-of-training, as they are for the base system.

---

## Table of contents
1. [Why an aggressive variant](#1-why-an-aggressive-variant)
2. [The +440 data](#2-the-440-data)
3. [The settings framework](#3-the-settings-framework)
4. [What is retrained — data + model per component](#4-what-is-retrained)
5. [The qHTS discriminator](#5-the-qhts-discriminator)
6. [Assembly — delta-injection onto the frozen winner](#6-assembly)
7. [Validation](#7-validation)
8. [Design findings & negatives](#8-design-findings--negatives)
9. [Reproduction notes](#9-reproduction-notes)

---

## 1. Why an aggressive variant

The final ranking is decided by **Set 2 (260, hidden)**, not the live combined-513 board. Two facts made a Set-2-targeted, higher-variance variant rational:

- **Set 1 is a weak predictor of Set 2 at our precision.** Set 1 ↔ Set 2 median nearest-neighbour Tanimoto ≈ 0.26 (a genuine scaffold shift; 0/260 Set-2 molecules have a Set-1 analogue above 0.6). Coarse ranking transfers, but in the fine-grained top-tier regime the Set-1 → Set-2 correlation collapses, and the ~0.005 gap between leaders is 0.18 σ on Set 1 — invisible. So a submission neutral-to-slightly-different on Set 1 is *not* evidence of Set-2 harm.
- **Set 2 is active-enriched, and 440 new high-activity molecules were released.** The base ensemble under-predicts the active tail (Set-1 pEC50 > 5 bias −0.15). Extra high-activity training data plausibly sharpens the region that dominates Set 2.

The variant hedges its risk by keeping **Set 1 as a clean held-out validator** — it retrains on the `+440` pool only (no Set-1 labels), so its Set-1 RAE is honest, not an in-region fit.

---

## 2. The +440 data

- A HuggingFace drop (commit `ab8903a`, dated **2026-05-28**) **adds 440 new molecules** with corrected pEC50 (crude→corrected ≈ +0.34 log real) — *not* a correction to existing labels (the base train is unchanged). After canonicalization the base training pool is **4,140** (the raw HF TRAIN.csv is 4,139); of the 440 new molecules 5 overlap it, so **435 are net-new** → the augmented pool `train_hi` = 4,140 + 435 = **4,575** (§3).
- Boltz cofold poses for the 435 posable ligands were generated with Boltz 2.2.1 (single-sequence / empty-MSA, 293-aa PXR-LBD, seed 42).
- The `+440` is **too weak to blend directly** as a standalone model (augmented GBR/LGBM/HGBRT/Ridge each reach only ~0.6 MAE, i.e. RAE ≈0.75). Its productive use is as a **retrain-pool addition to the strong real-embedding heads**.
- **It post-dates the winning file** (2026-05-22), so it is absent from every Phase-1 production component — including the MOE multi-kernel active-specialist (that is fit on the top-30% of the *main* train, not the +440; see main solution §8.2).

---

## 3. The settings framework

The retrain pipeline reads pre-built **setting tables** (columns `SMILES, TARGET`), each defining a training pool. Verified row counts and Set-1 content (by canonical SMILES):

| Setting | Rows | Composition | Set-1 in pool |
|---|---:|---|---:|
| `base` | 4,140 | base HF train | **0** |
| `hi` | 4,575 | base + `+440` (435 net-new) | **0** ← *this variant uses it* |
| `set1` | 4,342 | base + 202 Set-1 (51 held out → V) | 202 |
| `set1_hi` | 4,777 | base + 202 Set-1 + `+440` | 202 |
| `V` (holdout) | 51 | Set-1 molecules held out of every `set1*` pool | (all 51 ∈ Set-1) |

Each retrainable head produces per-setting predictions (a train out-of-fold vector, the test-513 vector, and the 51-mol V-holdout vector). "Aggressive" vs "conservative" differ by which heads × settings are selected and re-injected.

---

## 4. What is retrained

The aggressive variant is **SVM_20K + Holo + 3way**, each retrained on the `hi` setting (base + `+440`). Each head keeps its Phase-1 architecture *exactly* (main solution §3–6); only the training pool changes. **ConfTTA and Gated_5mod are held at production** (zero delta) — hence the name.

| Component | Retrained on | Model (unchanged) | Role in the bet |
|---|---|---|---|
| **SVM_20K** | `hi` (base + 440) | RBF-Nyström SVR, D=1786 (MiniMol⊕MoLFormer⊕LF20K⊕MACCS⊕ph4) | the *safe* retrain — real-embedding SVR is neutral-to-positive under +440 |
| **Holo** | `hi` (base + 440) | Uni-Mol v2 84M on the cofold binding pose (**retrained — `holo_hi`, not frozen**) | carries the bulk of the active-region bet (binding pose × new high-activity mols) |
| **3-way** | `hi` (base + 440) | RBF SVR on CLAMP⊕da4mt⊕ChemFM-3B (4608-D) | the most genuinely-external head; sharpens the active region while keeping cross-corpus orthogonality |

> **Holo is retrained, not frozen.** The variant uses the `+440`-retrained Holo predictions; a reconstruction confirms the Holo `+440` delta is a dominant term (removing it collapses the delta-fit R² from ~0.91 to ~0.33). Retraining Holo costs one Uni-Mol run per fold (≈1 h GPU each).

---

## 5. The qHTS discriminator

Identical to the main solution's post-processing gate — the robust, external, disjoint-transferring part of the change:

1. Train on the external NCATS PXR qHTS assay (**PubChem AID 1346982**), filtered to `Active`/`Inactive`, **canonical-SMILES-disjoint from train and test**.
2. **3-class labels** by potency (`pAC50 = −Fit_LogAC50-Replicate_1`, split at the median of the actives): inactive / weak-active / strong-active.
3. `HistGradientBoostingClassifier(max_iter=400, lr=0.05, random_state=0)` on ECFP4 (2048, r=2); `P_active = 1 − P(inactive)`.
4. **Gate:** `(prediction < 4) AND (P_active < 0.3) → prediction − 0.2` — flags **~58 molecules**, essentially the same low-activity set as the main solution's 58 (the classifier is identical; the `+440` delta only marginally shifts which predictions fall below 4).

This corrects the systematic over-prediction of inactives (Set-1 pEC50 < 4 bias +0.56). It is a discrete external-classifier gate — **not** a flat down-shift and **not** a distribution rescale.

---

## 6. Assembly

**In one line:** take the frozen winning prediction, add the small change caused by retraining three components on `+440`, then apply the qHTS gate — nothing else moves.

### 6.1 The delta-injection

The variant is **not** a from-scratch rebuild. It is a within-pipeline **delta injected onto the frozen winning prediction** `P513` (the main solution's ensemble output, before its qHTS gate):

```
v = P513 + Σ_c  W_c · ( component_c[hi] − component_c[base] )      # c ∈ {SVM_20K, Holo, 3way}
flag = (v < 4) & (P_active < 0.3) ;   v[flag] -= 0.2               # qHTS gate (§5)
```

Both terms of each delta come from the *same* reproduction pipeline, so the pipeline-reproduction offset cancels and only the **pure +440 data effect** is injected. When the retrain pool equals the base pool the delta is exactly 0 and `v` is byte-identical to the winner — so the baseline is exact and only the data change is expressed.

### 6.2 Injection weights

**Documented injection weights.** `DIL = 0.5` is the documented dilution of the QUAD base by the downstream MoE / MOE-mk / scale layers; each component's injection weight = its QUAD weight × DIL:

| Component | QUAD weight | Injection weight (× DIL 0.5) | Delta injected? |
|---|---:|---:|:--:|
| ConfTTA | 0.75 · 0.7 = 0.525 | held at base | **no** (→ 0) |
| Holo | 0.75 · 0.3 = 0.225 | **0.1125** | yes |
| SVM_20K | 0.125 | **0.0625** | yes |
| 3-way | 0.05 | **0.025** | yes |

(The deep pillar is `0.7·ConfTTA + 0.3·Holo` at a combined 0.375; only the Holo share carries a delta, since ConfTTA is held at its base array.)

### 6.3 Setting & provenance verification

**Which setting shipped — verified.** Reconstructing the shipped file from the component arrays, the `hi`-setting deltas reproduce it far better than any Set-1-augmented setting (delta-fit correlation ≈ 0.71 for `hi` vs ≈ 0.24 for `set1_hi`). Combined with `train_hi` containing **0** Set-1 molecules, this confirms the shipped variant uses `hi` — base + `+440`, **no Set-1**.

**Honesty on exact weights:** the specific inline builder that wrote `SUBMIT_AGGR_*.csv` is not byte-preserved; the per-component `vm()`-alignment plus ConfTTA/Holo delta collinearity leave a small (~0.02) residual in a linear reconstruction. The *method* (delta-injection + `hi` + qHTS), the *component set*, and the *setting* are all verified; the documented weights are the framework.

---

## 7. Validation

Because the variant uses `hi` (no Set-1), the whole of Set 1 (253) is out-of-training — its Set-1 RAE is honest:

- **Set 1, 253 (held-out):** aggressive **0.4841** vs base **0.4875** (−0.0034); MAE 0.3866 vs 0.3893. Nearly all of the resolvable Set-1 gain is the qHTS gate; the SVM/Holo/3way `+440` retrain is **within the noise floor** (and the aggressive variant sits +0.0064 RAE *above* the main solution's 0.4777).
- **V, 51 (cross-setting disjoint holdout):** aggressive **0.3479** vs base **0.3706** — the largest apparent gain, but on the smallest, noisiest sample.
- **Set 2, 260 (final):** aggressive **0.5692** vs main **0.5676** — the bet lost by 0.0016, within noise. The V(51) gain did *not* generalize to Set 2, exactly the small-sample-noise risk the analysis flagged.

**Honest reading:** the qHTS gate is the robust, transferable improvement; the `+440` Holo/3way retrain was a within-noise, active-region-targeted bet whose payoff depended on the Set-1 → Set-2 shift favouring the active tail — and on Set 2 it did not.

---

## 8. Design findings & negatives

- **Retraining is model-dependent.** `+440` *hurts* a weak ECFP-GBDT (+0.0194) but is **neutral** on real MoLFormer+ChemBERTa SVR heads (−0.0016, P(help)=57%). Retrain only the real-embedding heads.
- **Set-1 augmentation is at the detection floor** (nested-CV shift ≈0 at the best weight; the disjoint-region shift is within noise) — which is *why* this variant uses `hi` (no Set-1): folding in 253 in-region labels gains nothing resolvable and would forfeit Set-1 as a clean validator.
- **The `+440` alone is too weak to blend** (~0.6 MAE ≈ 0.75 RAE standalone); only useful as retrain-pool augmentation of strong heads.
- **Low-pred push-down works and transfers** (`pred<4 & high disagreement → −0.128`: Set-1 −0.0022, 100% disjoint, P=76%) — confirming the *direction* the qHTS gate encodes.
- **Domain-generalization methods fail** (GroupDRO/CORAL/IRM do not beat ERM on the scaffold shift). The hardest-region gap is information-limited (cliffs), not invariance-limited.
- **The outcome confirms the caution:** the aggressive variant that retrained on new data finished 0.0016 behind the conservative main solution on Set 2 — the honest "within-noise bet" framing was correct.

---

## 9. Reproduction notes

The variant is fully specified by this report: (i) the three retrained components — SVM_20K, Holo, 3-way — each trained on the `hi` setting (base + `+440`) with the identical architecture and hyper-parameters of the main solution (§4); (ii) the delta-injection onto the frozen winning ensemble (§6); and (iii) the qHTS gate (§5). All retrains use fixed seeds and 5-fold scaffold cross-validation, so they are deterministic given the setting tables. The exact per-component injection coefficients of the one-off assembly are not byte-preserved, but the method, component set, and setting are all independently verified (§6).

All numbers are re-scored against the unblinded Set-1 labels (253) and the 51-molecule V holdout; the Set-2 figures are the official final leaderboard.

---

*AIDD-LiLab, University of Florida — Yanjun Li Lab. Aggressive `+440` high-activity retrain variant — final rank #3 (Set 2, RAE 0.5692), 0.0016 behind the main solution. This work is described in the group's forthcoming publication.*
