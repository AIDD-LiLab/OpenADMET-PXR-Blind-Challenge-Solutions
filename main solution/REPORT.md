# OpenADMET PXR (NR1I2) Activation Challenge — Main Solution Report

**Team:** AIDD-LiLab (Department of Medicinal Chemistry / College of Pharmacy, University of Florida — Yanjun Li Lab)
**Task:** Predict human PXR (NR1I2) agonist potency (pEC50) for a blinded compound set.
**System name:** `MoE_v2 + MOE-multikernel`
**Final prediction file:** `MoE_v2_plus_MOE_multikernel_a05_w20.csv` (513 rows; columns `SMILES, Molecule Name, pEC50`).

---

## 1. Result

| Evaluation set | MAE | RAE | R² | Spearman ρ |
|---|---:|---:|---:|---:|
| Set 1 (253, Phase-1 live leaderboard) | 0.3899 | **0.4897** | 0.669 | 0.858 |
| Set 1 (253), re-scored locally (mean-centred MAD) | 0.3893 | 0.4875 | 0.669 | 0.861 |
| Combined 513 (Set 1 + Set 2) | 0.400 | **0.528** | 0.636 | 0.843 |

Ranked **#1** on the combined-513 public leaderboard at the close of Phase 1.
(Set 1 was unblinded on 2026-05-26; Set 2 labels remain the hidden final scorer. The two Set-1 RAE values differ only by the mean- vs official-normaliser convention — MAE is identical.)

**Metric.** RAE = Σ|ŷ−y| / Σ|y−ȳ| (relative absolute error; the challenge's primary metric).

---

## 2. Data

- **Training:** 4,139 dose-response molecules with measured pEC50.
- **Low-fidelity pretraining:** ~20,000 single-concentration / external PXR readouts (calibrated pseudo-pEC50) used only to pretrain the deep encoders (multi-fidelity transfer).
- **Counter-assay:** used both as a training filter (drop rows whose PXR pEC50 is below the counter-assay signal) and as an auxiliary signal.
- **Cofolded structures:** Boltz cofolding of each ligand into the PXR ligand-binding domain (UniProt O75469) — ~4,700 predicted protein–ligand complexes; 184 example cofold complexes were released with the challenge.
- **Test:** 513 molecules (Set 1 = 253, Set 2 = 260).

---

## 3. Architecture

The system is a layered ensemble. Each layer variance-matches to a fixed reference and adds a targeted correction; the final output is variance-scaled to the (narrower) test distribution.

```
                 ┌─────────────────────────────────────────────────────────┐
 SMILES ───────► │  QUAD base ensemble                                       │
 conformers      │   0.75 · (ConfTTA/Holo blend)  +  0.125 · SVM_20K         │
 cofold poses    │        + 0.125 · Gated_5mod                               │
 embeddings      │        + per-mol cofold-confidence additives              │
                 │          (all3_lgbm, ifp_svm)                             │
                 └───────────────┬─────────────────────────────────────────┘
                                 │  + 0.05·sw3way_svr  + 0.02·pharmHGT_v2     (Compound layer)
                                 ▼
                 ┌─────────────────────────────────────────────────────────┐
                 │  MoE_v2 : + Boltz-cofold ACTIVE-SPECIALIST (top-30%,      │
                 │            ipTM-gated)      — the plateau-breaker          │
                 └───────────────┬─────────────────────────────────────────┘
                                 │  + 0.20 · MOE multi-kernel SVR (gated to actives)
                                 ▼
                     variance-match (scale 1.08–1.17) ──► pEC50
```

### 3.1 Base learners (the four pillars)
- **ConfTTA** — Uni-Mol v2 (84M), multi-fidelity (20K low-fidelity pretrain → dose-response fine-tune), conformer test-time augmentation. Highest-weight pillar.
- **Holo** — the same Uni-Mol v2 trained on the **cofold binding pose** rather than a free conformer; blended at ≈30% into the ConfTTA slot to inject a docked-pose inductive bias.
- **SVM_20K** — multi-kernel SVR on frozen chemical-language embeddings (MiniMol + MoLFormer + LF-20K MPN) with MACCS/pharmacophore features.
- **Gated_5mod** — a gated fusion over five descriptor/embedding modalities (Mordred + charges + embeddings).

Two per-molecule **cofold-confidence-weighted additives** (an LGBM on IFP/ESP/pose-charge features and an SVR on ProLIF-IFP + 3D descriptors) contribute in proportion to a cofold-pose confidence estimate.

### 3.2 Mixture-of-Experts active specialist (plateau-breaker)
On top of the base ensemble, an SVR trained on 1408-D **Boltz cofold embeddings**, restricted to the **top-30% most-active compounds** and blended through a **sigmoid activity gate × Chai1 ipTM confidence**, corrects the high-pEC50 tail where the base ensemble under-predicts. This single intervention broke a multi-week plateau (RAE 0.4956 → 0.4941). Its correlation to the base ensemble is the lowest of any head we built (ρ ≈ 0.57–0.58), i.e. it adds genuine rank-changing signal in the active region.

**Key nuance:** the *same* Boltz cofold SVR used as a *flat* additive blend *hurt* the leaderboard. Cofold-pose signal was only useful when (i) restricted to the active subset and (ii) gated — never as a uniform mixture.

### 3.3 MOE multi-kernel additive
A multi-kernel SVR (0.5·Tanimoto(ECFP4) + 0.5·RBF over 340-D commercial MOE descriptors) added as a gated additive at weight ≈ 0.20 — the outermost, single-largest post-MoE improvement.

### 3.4 Variance matching
The test set is genuinely narrower than training (predicted std ≈ 0.79 vs training std ≈ 1.12). The final prediction is centred and re-scaled by a factor in the 1.08–1.17 range, tuned with a *P(test)-weighted* out-of-fold objective (an adversarial classifier estimates each training molecule's "test-likeness"). Variance **compression** is correct here, not a bug.

---

## 4. Validation methodology

- **5-fold scaffold-grouped cross-validation** (Bemis–Murcko scaffolds) for every learned component; test predictions are the fold average.
- Ensemble weights were fixed on out-of-fold RAE and confirmed on the (then-blinded) leaderboard.
- After Set 1 was unblinded, every ensemble weight and post-hoc rule was re-checked against the 253 real labels and against **scaffold-disjoint** out-of-fold splits (the honest proxy for the Set-1 → Set-2 chemistry shift).

---

## 5. Key findings

1. **Activity cliffs are the dominant error budget.** ~28% of Set 1 are activity cliffs (structurally near a training molecule but ≥1 log-unit different); cliff MAE ≈ 0.62 vs non-cliff MAE ≈ 0.30. Roughly half of the residual falls inside the assay's own 95% confidence interval — i.e. it is measurement noise, not model error.
2. **Error is edge-concentrated.** By activity region on Set 1: pEC50 < 4 (n=55) MAE 0.82, bias +0.56 (over-prediction of inactives); 4–5 (n=86) MAE 0.30, unbiased; > 5 (n=112) MAE 0.25, bias −0.15 (mild under-prediction of the active tail).
3. **Cofold structure helps only when targeted.** Three cofold-derived signals are in the winning model — Boltz-embedding active-specialist (gated), Chai1 ipTM (gate), Holo binding pose (≈30% blend) — but the *same* cofold SVR as a flat blend reversed on the leaderboard.
4. **Variance compression, not expansion, is correct** (test std < train std); calibration/distribution-matching in the other direction hurt.
5. **The score is near a statistical ceiling on Set 1** (bootstrap MAE σ ≈ 0.028 on 253 points); differences below ~0.005 RAE are not resolvable on this sample.

---

## 6. Files

- `MoE_v2_plus_MOE_multikernel_a05_w20.csv` — the submitted 513-row prediction.

*Detailed component code and extended methods accompany this report in the repository.*
