# OpenADMET PXR (NR1I2) Activation Challenge — Winning Solution

**Team:** AIDD-LiLab — Department of Medicinal Chemistry, College of Pharmacy, University of Florida (AIDD Team)
**Task:** Predict human PXR (Pregnane X Receptor, gene *NR1I2*) agonist potency **pEC50** from small-molecule structure.
**Final system:** `MoE_v2 + MOE-multikernel` ensemble **+ qHTS post-processing gate**.
**Submitted file:** `CAND_qHTS_gated_multiclass.csv` (513 rows: `SMILES, Molecule Name, pEC50`).

---

## 0. Result — 1st among non-proprietary teams

🥇 We placed **#1 among all teams that used no proprietary data** (and #2 overall). Final challenge leaderboard (evaluation on **Set 2**, the 260 hidden final-scoring molecules):

| Rank | Team | MAE | **RAE** | R² | Spearman ρ | Kendall τ | Proprietary data |
|---:|---|---:|---:|---:|---:|---:|:--:|
| 1 | matcha-croissant | 0.4061 | 0.5631 | 0.598 | 0.827 | 0.643 | **Yes** |
| **2** | **AIDD-LiLab** (this solution) | **0.4092** | **0.5676** | 0.591 | 0.825 | 0.644 | **No** |
| 3 | AIDD-LiLab-Aggressive ([variant](../Aggressive%20Version%20Report/README.md)) | 0.4104 | 0.5692 | 0.589 | 0.822 | 0.641 | **No** |

Our two submissions (this main solution and the aggressive variant) took **2nd and 3rd overall**; the only system ahead of them used proprietary data, so among non-proprietary-data teams we placed **1st and 2nd** — within 0.0045 RAE of the proprietary-data winner.

**The metric** is Relative Absolute Error, `RAE = Σ|ŷ−y| / Σ|y−ȳ|` (lower is better; 1.0 = predicting the mean).

**Score at each evaluation stage** — one system, three molecule subsets. **Only the bolded Set 2 row (0.5676) is the official final score**; Set 1 and combined-513 are validation context, not competing results:

| Stage | Molecules | RAE | MAE | Notes |
|---|---|---:|---:|---|
| Phase-1 live board (combined) | 513 (Set 1 + Set 2) | **0.528** | 0.400 | ranked **#1** at Phase-1 close |
| Set 1 (unblinded validator) | 253 | **0.4777** | 0.3815 | held-out; our reliable local validator |
| **Set 2 (final scorer)** | **260** | **0.5676** | 0.4092 | **the official final result above** |

Set 1 was unblinded on 2026-05-26 and used as a held-out validator; Set 2 stayed hidden until the final ranking. RAE is higher on Set 2 because it is a harder, more scaffold-distant slice (median Set 1↔Set 2 Tanimoto ≈ 0.26).

---

## Table of contents
0. [Result](#0-result--1st-among-non-proprietary-teams)
1. [How it works in one picture](#1-how-it-works-in-one-picture)
2. [Data sources](#2-data-sources)
3. [Component catalog](#3-component-catalog)
4. [Architecture — the layered assembly](#4-architecture--the-layered-assembly)
5. [The four base pillars in detail](#5-the-four-base-pillars-in-detail)
6. [The compound layer](#6-the-compound-layer)
7. [The cofold structure pipeline](#7-the-cofold-structure-pipeline)
8. [MoE active-specialist + MOE multi-kernel](#8-moe-active-specialist--moe-multi-kernel)
9. [Variance matching](#9-variance-matching)
10. [Post-processing — the qHTS discriminator](#10-post-processing--the-qhts-discriminator)
11. [Validation methodology](#11-validation-methodology)
12. [Error analysis & key findings](#12-error-analysis--key-findings)
13. [What did NOT work](#13-what-did-not-work)
14. [Reproducibility notes](#14-reproducibility-notes)

---

## 1. How it works in one picture

The system is a **layered ensemble**: a strong 4-pillar base, two orthogonal correction heads, a Mixture-of-Experts active-specialist, a commercial-descriptor multi-kernel, a variance rescale, and finally an external qHTS post-processing gate.

**Quick glossary** (so the diagram labels parse on first read):
- **ConfTTA / Holo** — the two deep 3D models (both Uni-Mol v2); ConfTTA sees a free-molecule 3D shape, Holo sees the docked, protein-bound pose.
- **cofold** — a predicted 3D structure of the ligand bound inside the PXR protein pocket.
- **active-specialist** — a model trained only on the most-active compounds, mixed in only for molecules predicted to be active.
- **multi-kernel** — a support-vector model combining a fingerprint-similarity kernel with an RBF kernel over descriptors.
- **qHTS gate** — an external cheap-assay classifier used as post-processing to nudge likely-inactive low predictions down.
- **ens12** — a fixed reference prediction distribution that every component is rescaled to match before mixing.

```
 SMILES ─┐
 3D confs ├─► ┌──────────────────────────────────────────────────────────────┐
 cofold  │    │ QUAD BASE                                                     │
 poses   │    │   0.75 · (ConfTTA ⊕ Holo)   +  0.125 · SVM_20K                │
 embeds ─┘    │        + 0.125 · Gated_5mod                                   │
              │        + per-molecule cofold additives (all3_lgbm, ifp_svm)   │
              └───────────────┬──────────────────────────────────────────────┘
                              │ + 0.05 · sw3way_svr  + 0.02 · pharmHGT_v2      (Compound)
                              ▼
              ┌──────────────────────────────────────────────────────────────┐
              │ MoE_v2 :  + Boltz-cofold ACTIVE-SPECIALIST (top-30%, ipTM-gate)│
              └───────────────┬──────────────────────────────────────────────┘
                              │ + 0.20 · MOE multi-kernel SVR (gated to actives)
                              ▼
                    variance-match  (× 1.08)
                              ▼
              ┌──────────────────────────────────────────────────────────────┐
              │ qHTS GATE (post-processing):                                  │
              │   if pred < 4  AND external qHTS P_active < 0.3  →  pred − 0.2 │   ← final model
              └───────────────┬──────────────────────────────────────────────┘
                              ▼   pEC50
```

Three principles carried the whole solution:

1. **Diversity by construction, not by accuracy.** The single best model (ConfTTA, OOF RAE 0.383) is not enough. Gains came from *decorrelated-error* components — a binding-pose twin (Holo), externally-pretrained encoders (3-way), and cofold-pose signal — even when individually weak.
2. **Additive, low-weight, variance-matched injection.** New signal is added as small signed deltas after variance-matching, never by replacing strong components. Blanket replacements hurt; tiny additive injections helped.
3. **Gate the noisy signal.** Cofold-pose signal, and the qHTS correction, help only when confidence-gated and confined to the region where they are trustworthy.

---

## 2. Data sources

| Source | N | How it is used |
|---|---:|---|
| Dose–response pEC50 (`TRAIN.csv`) | **4,139** | supervised target (the label everyone trains on) |
| Single-concentration HTS | 21,014 rows / 2,746 mols | calibrated to pseudo-pEC50 → deep-encoder **pretraining** |
| Counter-assay pEC50 | 2,859 | **selectivity filter** (drop PXR-null artifacts) + auxiliary signal |
| Boltz cofolded PXR–ligand complexes | ~4,728 | 3D binding poses & structural embeddings (used 3 ways) |
| External PXR assays (PubChem/ChEMBL/Tox21) | 9,316 | extend the low-fidelity pretraining corpus |
| Commercial MOE descriptors (gift) | 341-D, train+test | the MOE multi-kernel head (§8.2) |
| External NCATS PXR qHTS (PubChem AID 1346982) | ~9,700 | the qHTS post-processing classifier (§10) |
| Blinded test | 513 = **Set 1 (253) + Set 2 (260)** | Set 1 = validator, Set 2 = final scorer |

*Two naming notes: **MOE** = Molecular Operating Environment, a commercial cheminformatics package (**not** the Mixture-of-Experts "MoE" layer of §8.1). **LF** = low-fidelity single-concentration data; **MF** = multi-fidelity (pretrain on LF, then fine-tune on real labels).*

Two facts about this dataset shaped every choice:

- **The test set is active-enriched and narrower than training** (predicted std ≈ 0.79 vs train std ≈ 1.12) — a real potency/scaffold shift, not an i.i.d. split.
- **Activity cliffs** (structurally near-identical molecules with very different potency) **dominate the residual.** ~28% of Set 1 are near-identical to a training molecule yet ≥1 log-unit different in potency; that portion of the error is structurally irreducible.

**Multi-fidelity (MF) transfer** is the single most important data lever for the deep pillars: pretrain on ~20K cheap, noisy single-concentration readouts (calibrated to a pseudo-pEC50), then fine-tune on the 4,139 real dose-response labels.

**Counter-assay selectivity filter:** where a molecule's PXR pEC50 falls below its counter-screen signal (likely a non-specific artifact), the row is dropped from the SVR/GBDT training sets — **391 molecules dropped → 3,748 kept**. The deep conformer model deliberately does *not* filter (filtering hurt its 3D signal).

---

## 3. Component catalog

Every learnable piece of the winning system, with **the data it trained on and the model it is** — the quick-reference table.

| Component | Role & ensemble weight | Trained on | Model / architecture | Key configuration |
|---|---|---|---|---|
| **ConfTTA** | deep 3D pillar · ≈0.375 | 20,191 LF pseudo-pEC50 → 4,139 dose-response | **Uni-Mol v2 84M** 3D transformer | RDKit ETKDG conformer; 5-fold scaffold CV; FT epochs 50, bs 8, lr 5e-5, seed 42 |
| **Holo** | deep 3D pillar · ≈0.375 (blended into the ConfTTA slot) | same as ConfTTA | **Uni-Mol v2 84M** (identical backbone) | fed the **Boltz cofold binding pose** instead of a free conformer |
| **SVM_20K** | frozen-embedding head · 0.125 | 4,139 → filtered 3,748 | multi-feature **RBF-Nyström SVR** | features: MiniMol 512 ⊕ MoLFormer-XL 768 ⊕ LF-20K MPN 300 ⊕ MACCS 167 ⊕ ph4 39 (**D=1786**)<br>Nyström 500 (γ=1/D); SVR C=1, ε=0.1 |
| **Gated_5mod** | frozen-embedding head · 0.125 | 4,139 → filtered | learned **gated 5-modality MLP** | modalities: (MiniMol⊕MoLFormer) 1280, LF-10K 300, LF-20K 300, Mordred 1307, charges 5<br>hidden 128; 10 seeds × 5 folds; SmoothL1 train / L1 select |
| **all3_lgbm** | cofold additive · ≤3%·cf | 4,140, **cofold-only** features | **LGBM** | IFP 168 ⊕ ESP 20 ⊕ pose-charge 29 = 217-D; deliberately weak but maximally orthogonal (OOF 0.69, ρ<0.50 to base) |
| **ifp_svm** | cofold additive · ≤2%·cf | 3,746, base-2D + cofold | **RBF-Nyström SVR** | base-2D 1747 ⊕ IFP 168 ⊕ mordred3d top-200; a cofold-augmented near-twin of SVM_20K |
| **sw3way** | compound additive · 5% | 4,140 → filtered 3,746 | **RBF SVR on 3 external frozen pretrains** | CLAMP 768 ⊕ da4mt-MTR 768 ⊕ ChemFM-3B 3072 = **4608-D**; SVR C=10, γ=scale, ε=0.1 |
| **pharmHGT_v2** | compound additive · 2% | 4,140 → filtered 3,749 | **pharmacophore hetero-graph transformer** (from scratch) | dual-view BRICS heterograph; MVMP + Node-GRU readout; 3 seeds × 5 folds |
| **MoE Boltz specialist** | active-specialist (gated) | **top-30% actives** of the 3,742-mol filtered train | **RBF SVR** on 1408-D Boltz cofold embeddings | gate = sigmoid(activity) × Chai1 ipTM; base weight 0.30 |
| **MOE multi-kernel** | outermost additive · 0.20 | **top-30% actives** of the main train | **multi-kernel SVR** | K = 0.5·Tanimoto(ECFP4) + 0.5·RBF(**MOE 341-D** commercial descriptors) |
| **variance scale** | global affine post-mix | — | `mean + 1.08·(x−mean)` | scale tuned by an adversarial P(test)-weighted objective |
| **qHTS gate** (§10) | **post-processing (final layer)** | **external NCATS AID 1346982** (disjoint from train & test) | **ECFP4 → 3-class HistGradientBoosting classifier** | `(pred<4) & (P_active<0.3) → −0.2`; touches **58 / 513** molecules |

*Feature shorthand used above: MiniMol / MoLFormer-XL = pretrained molecular embeddings; MACCS / ECFP4 / ph4 = structural or pharmacophore fingerprints; MPN = message-passing-network embedding; IFP = protein–ligand interaction fingerprint; ESP = electrostatic-potential charges; Mordred = physicochemical descriptors.*

All learnable components use **5-fold Bemis–Murcko scaffold GroupKFold**; the reported test vector is the 5-fold average.

---

## 4. Architecture — the layered assembly

Each layer variance-matches to a fixed reference `ens12` (via `vm(x, ref)` = shift+scale `x` to `ref`'s mean/std) and adds a targeted correction. The leaderboard walks down as the layers stack: **Compound 0.4956 → MoE_v2 0.4941 → +MOE-mk & scale 0.4897 → +qHTS gate 0.4777** (the `# LB ...` comments in the blocks below mark each step).

**QUAD base:**
```
QUAD = 0.75  · vm(0.5·ConfTTA + 0.5·Holo, ens12)        # deep 3D block (dominant)
     + 0.125 · vm(SVM_20K,   ens12)                      # frozen-embedding SVR
     + 0.125 · vm(Gated_5mod, ens12)                     # gated 5-modality fusion
     + w_lgbm(m)·vm(all3_lgbm, ens12) + w_ifp(m)·vm(ifp_svm, ens12)   # per-mol cofold additives
   with  w_lgbm(m)=0.03·cf(m),  w_ifp(m)=0.02·cf(m),  cf(m)=clip(1+2·z(cofold_conf(m)), 0, 5)
```
**Compound layer:**
```
Compound = 0.93·QUAD + 0.05·vm(sw3way, QUAD) + 0.02·vm(pharmHGT_v2, QUAD)      # LB 0.4956
```
**MoE + MOE-mk + scale:**
```
MoE_v2 = Compound + Boltz-cofold ACTIVE-SPECIALIST (gated to predicted actives × ipTM)   # LB 0.4941
Final_ensemble = MoE_v2 + 0.20·(gated) MOE multi-kernel SVR ,  then × 1.08 variance-scale  # LB 0.4897
```
**Final model (this submission):**
```
pEC50 = qHTS_gate( Final_ensemble )      # −0.2 on 58 low, externally-inactive mols → Set-1 0.4777
```

> **Note on the pillar blend.** The production formula uses `0.5·ConfTTA + 0.5·Holo`. An earlier breakthrough and the Phase-2 retrain pipeline used `0.7·ConfTTA + 0.3·Holo`; the two are near-indistinguishable at the winning file's precision (corr 0.9876 vs 0.9859 to the shipped CSV), so both are documented and neither can be uniquely confirmed from the output alone.

---

## 5. The four base pillars in detail

### 5.1 ConfTTA — the highest-weight pillar
**Data:** MF transfer — pretrain on the 20,191-molecule low-fidelity pseudo-pEC50 corpus, fine-tune on the 4,139 dose-response labels.
**Model:** Uni-Mol v2 84M (12 layers, embed 768, 48 heads, pair channel 512/64, triangle-multiplication updates). Loads DP Technology's public self-supervised 84M Uni-Mol v2 checkpoint (from HuggingFace `dptech/Uni-Mol-Models`), continues on PXR-LF, then fine-tunes.
**Training:** 5-fold GroupKFold on Murcko scaffolds; per fold `epochs=50, batch=8, lr=5e-5, early_stop=10, max_norm=10, seed=42`, each fold initialized from the LF checkpoint.
**Inference:** unimol_tools generates **one** ETKDG conformer per molecule (`randomSeed=42` + MMFF optimization); the genuine averaging is the **mean over the 5 fold models** (the "TTA" name is historical — a separate 4-conformer-seed TTA was tested and found null). **Standalone out-of-fold (OOF) RAE 0.383** — the strongest single model in the pipeline.

### 5.2 Holo — the binding-pose twin
**Data:** identical labels to ConfTTA.
**Model:** the *same* Uni-Mol v2 84M, but the 3D input is the **Boltz cofold binding pose** (ligand coordinates extracted from the predicted PXR–ligand complex) instead of a free RDKit conformer. Same training recipe.
**Why it helps:** correlation to ConfTTA ≈ 0.97 (one backbone), but the docked pose repositions the hard-to-place actives — its value is *decorrelated error*, not raw accuracy. Standalone OOF RAE 0.435.

### 5.3 SVM_20K — frozen-embedding multi-feature SVR
**Data:** 4,139 dose-response, counter-assay-filtered to 3,748.
**Model:** a single RBF-Nyström SVR over concatenated **frozen** representations: MiniMol 512 + MoLFormer-XL 768 + a **20K-LF-pretrained Chemprop D-MPNN embedding 300** + MACCS 167 + Gobbi pharmacophore 39 (D=1786). Per fold: StandardScaler → Nyström(rbf, 500, γ=1/D) → SVR(C=1, ε=0.1).
**Why "20K":** the MPN embedding comes from a Chemprop D-MPNN pretrained on the 20,191 pseudo-pEC50 molecules — a semi-supervised feature. Orthogonal to the deep pillars because it is linear-in-kernel over fixed embeddings. OOF RAE ≈0.52 (≈0.520 on the base pool / 0.523 counter-filtered; an earlier Phase-1 configuration reported 0.503). The 10K-LF variant scored slightly better on OOF (~0.49) but the 20K variant gave a real **−0.0015 leaderboard gain** — an instructive OOF/LB divergence.

### 5.4 Gated_5mod — learned gated multi-modal fusion
**Data:** 4,139, filtered.
**Model:** each of five modalities (MiniMol⊕MoLFormer 1280, LF-10K MPN 300, LF-20K MPN 300, Mordred 2D 1307, Gasteiger charge stats 5) passes a `Linear→LayerNorm→GELU→Dropout` encoder to 128-D; a softmax **gate** over the five encoded vectors produces per-molecule attention weights; the fused 128-D vector feeds a 2-layer head. 10 seeds × 5 folds; AdamW(2e-3), CosineAnnealing 150 epochs, SmoothL1 train / L1 model-select. Keeping *both* LF-10K and LF-20K as separate gated streams was a deliberate fix (replacing one with the other reversed on the LB). OOF RAE 0.498.

---

## 6. The compound layer

Two small, deliberately-orthogonal heads added as low-weight signed deltas on the QUAD base (locked 2026-05-11, LB **0.4956**).

### 6.1 sw3way — the external-pretrain breakthrough (5%)
**Thesis:** encoders pretrained on *different corpora* produce genuinely orthogonal errors, unlike Uni-Mol clones that converge to the same manifold. This head broke a 56-hour plateau.
**Data / features:** three frozen encoders concatenated to **4608-D** —
- **CLAMP** (`ml-jku/CLAMP`, ICML 2023): contrastive language–assay MLP pretrained on ~500K PubChem BioAssay datapoints → 768.
- **da4mt-MTR** (`UdS-LSV/da4mt-mtr-60`): 2-stage MLM+MTR chemistry BERT → 768 (mean-pool).
- **ChemFM-3B** (`ChemFM/ChemFM-3B`): 3B-param LLaMA-style causal chemical LM trained on 178M UniChem molecules → 3072 (mean-pool).
**Model:** per-block + global StandardScaler → `SVR(rbf, C=10, γ=scale, ε=0.1)`, 5-fold scaffold. OOF RAE 0.545; entered at 5% (LB dose-response peaked at 5%, *not* the OOF-predicted 3%).

### 6.2 PharmHGT-v2 — from-scratch pharmacophore graph (2%)
**Thesis:** a non-embedding inductive bias — learned message passing on a BRICS-pharmacophore heterograph.
**Model:** dual-view BRICS heterograph (atoms + pharmacophore fragments; atom/bond/pharm/reaction/junction edge types), two MVMP multi-view message-passing modules, a 6-head cross-attention + bidirectional-GRU **Node-GRU** readout → 1200-D → MLP head. 3 seeds × 5 folds. OOF RAE 0.563 (just misses the strict <0.55 gate) but test-correlations to every base component < 0.83 — "best non-3D orthogonal candidate." Compound (3-way 5% + PharmHGT 2%) = LB 0.4956.

---

## 7. The cofold structure pipeline

Every ligand is cofolded into the PXR ligand-binding domain (UniProt O75469) with Boltz (Boltz and Chai1, used below, are AlphaFold-style 3D structure predictors), yielding ~4,728 predicted protein–ligand complexes. Cofold enters the winning system **three independent ways**:

1. **Binding pose → Holo pillar** (§5.2).
2. **Per-molecule cofold additives** (below).
3. **Internal embedding → MoE active-specialist** (§8.1).

**Cofold feature families** (from the poses): ProLIF interaction fingerprint (168), ESP-DNN partial-charge stats (20), pose-dependent charge×pocket features (29), 3D Mordred (1826), plus distance/contact-density tiers.

**The two additive heads:**
- **`all3_lgbm`** — LGBM on cofold-only features (IFP⊕ESP⊕pose-charge = 217-D). Weak (OOF 0.69) but *maximally orthogonal* (ρ<0.50 to every base component) — admitted precisely for that orthogonality.
- **`ifp_svm`** — RBF-Nyström SVR on base-2D ⊕ IFP ⊕ mordred3d. Strong (OOF 0.50) but ρ≈0.976 to SVM_20K, hence a tiny weight.

**Per-molecule confidence weighting:** each additive contributes `weight · cf(m)`, with `cf(m)=clip(1+2·z(cofold_conf(m)),0,5)`. Low-confidence poses down-weight their own contribution. This confidence gate is *cofold-specific* — applying the same modulation to a non-cofold head reversed on the LB. Max total cofold contribution ≈5% (matching the empirical cofold dose-response peak).

---

## 8. MoE active-specialist + MOE multi-kernel

### 8.1 MoE active-specialist — the plateau-breaker
**Data:** the **top-30% most-active** compounds of the ~3,742-mol counter-assay-filtered train (`y > p70 ≈ 5.04`, ≈1,124 mols), represented by **1408-D Boltz cofold embeddings** (pooled single + pair Pairformer reps: 384+384+384+128+128).
**Model:** `SVR(rbf, C=1, γ=scale)`, mixed into the base through a **sigmoid activity gate × Chai1 ipTM confidence** (ipTM = interface predicted TM-score, 0–1, a cofold pose-confidence estimate):
```
pred(m) = (1−w(m))·base(m) + w(m)·specialist(m)
w(m)    = 0.30 · sigmoid((base(m) − p70)/0.10) · (0.5 + 0.5·ipTM(m))
```
It corrects the high-pEC50 tail where the base under-predicts; the rank change is confined to ~154 active-suspect molecules. Correlation to the base ensemble ≈ 0.58 (corr to the SOTA base 0.581) — the lowest of any head, i.e. genuine rank-changing signal. **Broke a multi-week plateau: RAE 0.4956 → 0.4941.**
> **Critical nuance:** the *same* Boltz SVR used as a flat, uniform additive **hurt** the LB. Cofold-pose signal was net-positive only when confined to the active subset and gated.

### 8.2 MOE multi-kernel — the outermost, largest post-MoE gain
**Data:** the **top-30% actives** of the main train (an active-specialist, like §8.1), represented by **341-D commercial MOE descriptors** (Molecular Operating Environment / CCG; a gift). Test molecules were included in descriptor generation → in-distribution, bypassing the train→test representation shift.
**Model:** a precomputed **multi-kernel SVR**, `K = 0.5·Tanimoto(ECFP4) + 0.5·RBF(MOE-341D)`, added as a gated additive at weight **0.20**. Its cached signature (mean 5.506 / std 0.146) is the tight, high-mean fingerprint of a top-30%-actives specialist; correlation to base ≈ 0.64. The single largest post-MoE improvement, walking the LB from 0.4941 to the winning ensemble **0.4897**.
> The +440 high-activity molecules dropped 2026-05-28, six days *after* this component was built (2026-05-21) — they are **not** its training set (they appear only in the aggressive variant).

---

## 9. Variance matching

The test set is genuinely narrower than training (predicted std 0.79 vs train 1.12); an adversarial train/test classifier separates the two at **AUC ≈ 0.97** (severe covariate shift), confirming compression is *correct*, not a bug.

- **`vm(p, t) = t.mean() + (t.std()/p.std())·(p − p.mean())`** — every component is variance-matched to the `ens12` reference before mixing.
- **Final scale `mean + 1.08·(x−mean)`**, chosen with an adversarial **P(test)-weighted** objective: a RandomForest estimates each training molecule's "test-likeness" from ECFP4, and the scale is optimized on the test-like subset (calibrated range 1.08–1.17; production 1.08).

---

## 10. Post-processing — the qHTS discriminator

The **only** change from the Phase-1 winning ensemble to the final submitted model is one external, leak-free post-processing gate that corrects the system's single largest systematic error: **over-prediction of inactives** (on Set 1, pEC50 < 4 has bias **+0.56**).

**The classifier** (external, disjoint):
1. Train on the **external NCATS PXR qHTS assay** (PubChem AID 1346982, ~9,700 compounds), filtered to `Active`/`Inactive` and made **canonical-SMILES-disjoint from both train and the 513 test**.
2. **3-class labels** by potency (`pAC50 = −Fit_LogAC50-Replicate_1`): 0 = inactive, 1 = weak active (`pAC50 < median of the actives`), 2 = strong active (`pAC50 ≥ median of the actives`) — so weak/strong actives are separated, not merged.
3. **Model:** `HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, random_state=0)` on ECFP4 (2048-bit, radius 2). `P_active = 1 − P(class 0)`.

**The gate:**
```
if  prediction < 4  AND  P_active < 0.3:   prediction −= 0.2
```
It fires on **58 of 513** molecules (all shifted −0.2, nothing else touched). On the Set-1 validator this improves RAE from **0.4875 → 0.4777** (−0.0098) — a real, externally-grounded, disjoint-transferring gain.

**Why this and not a retrain:** four validators agreed that retraining the ensemble on newly-released data (Set-1 labels or the +440 high-activity pool) does *not* beat the base at the ensemble level, so the final model keeps the frozen winning ensemble and adds only this orthogonal external correction. (The alternative — retraining on +440 — is the aggressive variant, which finished #3.)

---

## 11. Validation methodology

- **5-fold Bemis–Murcko scaffold GroupKFold** for every learnable component; the test vector is the fold average.
- Ensemble weights were fixed on out-of-fold RAE and confirmed on the (then-blinded) live leaderboard.
- **OOF is only a coarse validator.** A hard-won lesson: out-of-fold CV does not faithfully track the held-out ranking (random-fold OOF leaks scaffold neighbours). Every weight and additive was ultimately chosen against the **live leaderboard**, with OOF used as a filter + orthogonality gate. A candidate had to pass two gates before an LB test was spent: standalone OOF RAE < 0.55 **and** max correlation to every existing component < 0.85 — necessary but not sufficient (many gate-passers still reversed).
- After Set 1 unblinded, every weight and post-hoc rule was re-checked against the 253 real labels and against scaffold-disjoint splits (the honest proxy for the Set-1 → Set-2 shift).

---

## 12. Error analysis & key findings

Verified on the 253 unblinded Set-1 labels:

1. **Activity cliffs are the error budget.** 28.5% of Set 1 are cliffs (NN Tanimoto ≥ 0.5 and |ΔpEC50| ≥ 1.0); **cliff MAE 0.62 vs non-cliff 0.30** (a 2.1× gap). ~53% of predictions fall inside the assay's own 95% CI — that fraction is measurement noise, not model error.
2. **Error is edge-concentrated.** pEC50 < 4 (n=55): MAE 0.82, bias **+0.56** (inactives over-predicted — the qHTS gate's target); 4–5 (n=86): MAE 0.30, unbiased; > 5 (n=112): MAE 0.25, bias **−0.15** (active tail mildly under-predicted — the MoE specialist's target).
3. **Cofold structure helps only when targeted** (active-specialist + ipTM gate + pose blend); flat-blended, the same signal reverses.
4. **Compression is correct** (test std < train std); expanding variance hurt.
5. **Near a statistical ceiling.** Bootstrap MAE σ ≈ 0.028 on 253 points — RAE differences below ~0.005 are unresolvable on Set 1 (which is exactly the margin between the top three teams on Set 2).

---

## 13. What did NOT work

Negatives worth recording (all leaderboard-verified):
- **External-assay augmentation** (ChEMBL / extra PXR data) — hurts; wrong chemical region.
- **kNN / local-analogue correctors** — hurt (adversarial train/test AUC ≈ 0.97; test is a different region).
- **Variance *expansion*, power-law amplification, rank-refinement** — all reverse.
- **Multi-seed averaging** (corr > 0.97, adds nothing) and most single-embedding stackers (OOF-optimistic, don't transfer).
- **Flat (ungated) cofold blends** — reverse; only gated/active-restricted forms help.
- **All 3D-descriptor additives** — systematically reverse despite passing strict OOF gates.
- **Dim-recombination / kernel tricks** (PCA/KernelPCA/reverse-distillation, xRFM, MDN heads, multi-task aux) — cannot break the Pareto frontier on the existing embedding pool.
- **The lesson from those failures:** real breakthroughs required a *new label-relevant signal source* (external pretrain corpus, cofold pose, QM) — which is exactly what the 3-way SVR, the cofold heads, and the MOE descriptors provided.
- **Retraining on +440 high-activity data** — neutral-to-slightly-negative at the ensemble level (four validators agree); the aggressive variant that did retrain finished #3, 0.0016 behind this main solution on Set 2.

---

## 14. Reproducibility notes

Every learnable component uses fixed random seeds and 5-fold Bemis–Murcko scaffold cross-validation, so fold splits, ensemble weights, and the variance scale are deterministic. The system is fully specified by this report: the four base pillars (§5), the two compound heads (§6), the cofold-derived heads (§7), the MoE active-specialist and MOE multi-kernel (§8), the variance match (§9), and the qHTS post-processing gate (§10) — each with its data, model, and hyper-parameters. Training was GPU-only throughout (deep 3D pillars on Uni-Mol, the graph model on PyTorch-Geometric, the Chemprop MPN embeddings, Boltz 2.2.1 cofolding, and ESP-DNN charges). The qHTS gate is the smallest and most easily reproduced part of the system.

---

*AIDD-LiLab, University of Florida. Final rank #2 / #1 among non-proprietary-data teams (Set 2, RAE 0.5676). This solution is described in the group's forthcoming publication.*
