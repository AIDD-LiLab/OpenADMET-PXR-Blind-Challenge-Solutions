# OpenADMET PXR (NR1I2) Activation Challenge — Winning Solution

**Team:** AIDD-LiLab — Department of Medicinal Chemistry, College of Pharmacy, University of Florida (Yanjun Li Lab)
**Task:** Regress human PXR (NR1I2) agonist potency, pEC50, from small-molecule structure.
**System:** `MoE_v2 + MOE-multikernel` — a layered, multi-fidelity, structure-aware ensemble.
**Submitted file:** `MoE_v2_plus_MOE_multikernel_a05_w20.csv` (513 rows: `SMILES, Molecule Name, pEC50`).

---

## 0. Result

| Evaluation set | MAE | RAE | R² | Spearman ρ | Rank |
|---|---:|---:|---:|---:|---:|
| Set 1 (253, Phase-1 live LB) | 0.3899 | **0.4897** | 0.668 | 0.858 | — |
| Set 1 (253), local re-score (mean-centred MAD) | 0.3893 | 0.4875 | 0.669 | 0.861 | — |
| **Combined 513 (Set 1 + Set 2)** | **0.400** | **0.528** | 0.636 | 0.843 | **#1** |

RAE (the challenge metric) = Σ|ŷ−y| / Σ|y−ȳ|. Set 1 was unblinded on 2026-05-26; Set 2 (260) is the hidden final scorer. The two Set-1 RAEs differ only by the normaliser convention (identical MAE); we disclose both.

---

## 1. Problem framing

Human PXR is a promiscuous xenobiotic sensor: activation induces CYP3A4 and drives drug–drug interactions, so PXR agonism is an ADMET liability to predict early. The assay is a reporter gene readout; the challenge target is pEC50 (−log10 EC50). Two properties of this dataset shaped every design choice:

- **The test set is active-enriched and narrower than training** (predicted std ≈ 0.79 vs training std ≈ 1.12): a scaffold/potency shift, not an i.i.d. split.
- **Activity cliffs dominate the residual.** Structurally near-identical pairs differ by ≥1 log unit in potency; no purely structural encoder can separate them, so a large part of the error is irreducible.

---

## 2. Data

| Source | N | Use |
|---|---:|---|
| Dose-response pEC50 (main train) | 4,139 | supervised target |
| Single-concentration / external PXR readouts | ~21,000 | low-fidelity (LF) **pretraining** only (calibrated pseudo-pEC50) |
| Counter-assay pEC50 | 2,859 | training filter + auxiliary signal |
| Boltz cofolded PXR–ligand complexes | ~4,700 | 3D pose features (see §3.2, §3.5, §3.7) |
| Test (blinded) | 513 | Set 1 = 253 + Set 2 = 260 |

**Multi-fidelity (MF) transfer.** The deep encoders are pretrained on the ~20K LF readouts (cheap, noisy, calibrated to a pseudo-pEC50) and then fine-tuned on the 4,139 real dose-response labels. This is the single most important data lever for the deep pillars — it injects PXR-specific chemistry the 4K set is too small to teach.

**Counter-assay filter.** Rows whose PXR pEC50 falls below the counter-assay signal (i.e. likely assay artifacts / non-specific) are dropped from the SVR training set (≈390 rows). The deep conformer model deliberately does **not** filter (filtering hurt its 3D signal).

**Cofolding.** Every ligand is cofolded into the PXR ligand-binding domain (UniProt O75469) with Boltz, yielding a predicted protein–ligand complex per compound. Cofold structure is used **three** independent ways: (a) the Boltz internal embedding drives the MoE active-specialist (§3.7); (b) Chai1 interface confidence (ipTM) gates that specialist; (c) the docked binding pose feeds the Holo pillar (§3.2).

---

## 3. Architecture — a layered ensemble

Each layer variance-matches to a fixed reference (`vm(x, ref)` = shift+scale x to ref's mean/std) and adds a targeted correction. The core is a 4-pillar "QUAD" base; two additive "Compound" heads, a Mixture-of-Experts active specialist, and a commercial-descriptor multi-kernel are stacked on top; the output is variance-scaled to the compressed test distribution.

```
QUAD base
  = 0.75 · vm(ConfTTA⊕Holo blend, ens12)          # deep 3D pillars (dominant)
  + 0.125 · vm(SVM_20K,  ens12)                    # frozen-embedding multi-kernel SVR
  + 0.125 · vm(Gated_5mod, ens12)                  # gated 5-modality fusion
  + w_lgbm(m)·vm(all3_lgbm, ens12)                 # cofold additives, per-mol confidence-weighted
  + w_ifp(m) ·vm(ifp_svm,   ens12)
        with  w_lgbm(m)=0.03·cf(m),  w_ifp(m)=0.02·cf(m),  cf(m)=clip(1+2·z(cofold_conf), 0, 5)

Compound = 0.93·QUAD + 0.05·vm(sw3way_svr, QUAD) + 0.02·vm(pharmHGT_v2, QUAD)

MoE_v2   = Compound  +  Boltz-cofold ACTIVE-SPECIALIST   (gated to predicted actives × ipTM)

Final    = MoE_v2  +  0.20 · (gated) MOE multi-kernel SVR ,   then variance-scale (×1.08–1.17)
```

### 3.1 Pillar 1 — ConfTTA (highest weight)
Uni-Mol v2 (84M-parameter 3D transformer). Phase 1: pretrain on the ~20K LF set (pseudo-pEC50). Phase 2: fine-tune on 4,139 dose-response labels under 5-fold scaffold CV. Inference uses **conformer test-time augmentation** (predictions averaged over generated conformers). It supplies the strongest single signal and ingests any SMILES directly.

### 3.2 Pillar 2 — Holo (docked-pose bias)
The identical Uni-Mol v2 model, but fed the **Boltz cofold binding pose** instead of a free RDKit conformer. It is blended (~30%) into the ConfTTA slot so the ensemble sees a docked-pose inductive bias the free conformer lacks. Correlation to ConfTTA ≈ 0.97 (expected — same backbone), but the pose changes the hard-to-place actives.

### 3.3 Pillar 3 — SVM_20K (frozen-embedding multi-kernel SVR)
A support-vector regressor over concatenated **frozen** representations — MiniMol (512) + MoLFormer-XL (768) + a 20K-LF-pretrained Chemprop MPN embedding (300) + MACCS (167) + a 2D pharmacophore fingerprint. Multi-kernel (Tanimoto on fingerprints + RBF on continuous embeddings). It is orthogonal to the deep pillars because it is linear-in-kernel over fixed embeddings.

### 3.4 Pillar 4 — Gated_5mod
A learned gated fusion over five descriptor/embedding modalities (Mordred physchem, partial charges, and embeddings), which routes each molecule to the modality subset that predicts it best.

### 3.5 Cofold-confidence additives
Two per-molecule additives contribute in proportion to a **cofold-pose confidence** estimate `cf(m)`: an LGBM on ProLIF interaction-fingerprint / ESP / pose-charge features (`all3_lgbm`, ≤3%) and an SVR on IFP + 3D descriptors (`ifp_svm`, ≤2%). Low-confidence poses down-weight their own contribution — the additive is trusted only where the cofold pose is reliable.

### 3.6 Compound layer
Two small additive corrections on the QUAD base: `sw3way_svr` (5%) — an SVR over three *external* frozen pretrains (CLAMP + da4mt + ChemFM-3B), the first genuinely orthogonal external-pretrain signal that improved the leaderboard; and `pharmHGT_v2` (2%) — a pharmacophore hetero-graph transformer readout.

### 3.7 MoE active specialist — the plateau-breaker
On top of Compound, an SVR trained on **1408-D Boltz cofold embeddings** (pooled single + pair Pairformer representations), restricted to the **top-30% most-active** compounds, is mixed in through a **sigmoid activity gate × Chai1 ipTM confidence**. It corrects the high-pEC50 tail, where the base ensemble systematically under-predicts. This one change broke a multi-week plateau (RAE 0.4956 → 0.4941). Mechanistically it is a *decorrelation* win: its correlation to the base ensemble (ρ ≈ 0.57) is the lowest of any head we built, so it adds genuine rank-changing signal in the active region.

> **Critical nuance.** The *same* Boltz cofold SVR used as a **flat, uniform additive hurt** the leaderboard. Cofold-pose signal was net-positive only when (i) confined to the active subset and (ii) gated — never as a global mixture.

### 3.8 MOE multi-kernel additive (outermost, largest post-MoE gain)
A multi-kernel SVR — 0.5·Tanimoto(ECFP4) + 0.5·RBF over 340-D commercial MOE physicochemical descriptors — added as a gated additive at weight ≈ 0.20. The physics-aware descriptors add signal the ligand-only encoders miss; it is the single largest post-MoE improvement.

### 3.9 Variance matching
The test set is genuinely narrower than training (std 0.79 vs 1.12). The final prediction is centred and re-scaled by a factor tuned in the 1.08–1.17 range against a **P(test)-weighted** out-of-fold objective: an adversarial classifier estimates each training molecule's "test-likeness", and the scale is optimized on the test-like subset. Compression (not expansion) is correct here.

---

## 4. Training & validation methodology

- **5-fold scaffold-grouped CV** (Bemis–Murcko, chirality off) for every learned component; the reported test vector is the 5-fold average.
- Ensemble weights fixed on out-of-fold RAE, confirmed on the (then-blinded) leaderboard.
- After Set 1 unblinded, every weight and post-hoc rule was re-checked against the 253 real labels **and** against scaffold-disjoint out-of-fold splits — the honest proxy for the Set-1 → Set-2 chemistry shift (random-fold OOF is over-optimistic because it leaks scaffold neighbours).

---

## 5. What moved the needle (chronology)

| Milestone | Set-1 / LB RAE |
|---|---:|
| Rank-averaged 13-component ensemble | 0.5282 |
| QUAD base (4 pillars + MF + cofold-confidence, α=2.0, scale 1.08) | ~0.4983 |
| + `sw3way` external-pretrain SVR @5% + `pharmHGT_v2` @2% (Compound) | 0.4956 |
| + Boltz-cofold **active-specialist** (MoE, gated) | 0.4941 |
| + **MOE multi-kernel** @≈0.20 (winning submission) | **0.4897** |

---

## 6. Key scientific findings

1. **Activity cliffs are the error budget.** ≈28% of Set 1 are cliffs; cliff MAE ≈ 0.62 vs non-cliff ≈ 0.30. ≈53% of predictions fall inside the measured 95% CI — that fraction of the residual is assay noise, not model error.
2. **Error is edge-concentrated.** Set-1 by region: pEC50<4 (n=55) MAE 0.82, bias **+0.56** (inactives over-predicted); 4–5 (n=86) MAE 0.30, unbiased; >5 (n=112) MAE 0.25, bias **−0.15** (active tail mildly under-predicted).
3. **Cofold structure helps only when targeted** (active-specialist + ipTM gate + pose blend); the same signal flat-blended reverses.
4. **Compression is correct** (test std < train std); distribution-matching the other way hurt.
5. **Near a statistical ceiling on Set 1** — bootstrap MAE σ ≈ 0.028 on 253 points; RAE differences below ~0.005 are unresolvable on this sample.

---

## 7. What did NOT work (negatives worth recording)

- ChEMBL / external-assay data augmentation (hurts; wrong chemical region).
- kNN / local-analogue correctors (hurt; adversarial train/test AUC ≈ 0.97 — test is a different region).
- Distribution-matching / variance *expansion*; power-law amplification; rank-refinement.
- Multi-seed averaging (corr > 0.97 — adds nothing); most single-embedding stackers (OOF-optimistic, don't transfer).
- Flat (ungated) cofold blends.

---

## 8. Files

- `MoE_v2_plus_MOE_multikernel_a05_w20.csv` — the submitted 513-row prediction.

*Component training code and extended methods accompany this report in the repository.*
