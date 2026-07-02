# AIDD-LiLab — Winning Solution for the OpenADMET PXR (NR1I2) Activity Challenge

**Team:** AIDD-LiLab, University of Florida — Yanjun Li Lab
**Task:** Regression of PXR (Pregnane X Receptor, gene *NR1I2*) agonist potency (pEC50) from small-molecule structure.
**Result:** **Rank #1** on the combined-513 leaderboard (RAE 0.528). Live Set-1 leaderboard RAE **0.4897** (post-unblind mean-MAD recompute **0.4875**; MAE **0.3893**, R² **0.669**, Spearman **0.861**).
**Metric:** Relative Absolute Error, `RAE = Σ|y − ŷ| / Σ|y − ȳ|`.

---

## Table of Contents
1. [Executive summary](#1-executive-summary)
2. [Data](#2-data)
3. [Validation methodology](#3-validation-methodology)
4. [System architecture — the layered assembly](#4-system-architecture--the-layered-assembly)
5. [Subsystem 1 — the two deep 3D pillars (ConfTTA and Holo, Uni-Mol v2 84M)](#5-subsystem-1--the-two-deep-3d-pillars)
6. [Subsystem 2 — frozen-embedding heads (SVM_20K and Gated_5mod)](#6-subsystem-2--frozen-embedding-heads)
7. [Subsystem 3 — the compound layer (sw3way and PharmHGT-v2)](#7-subsystem-3--the-compound-layer)
8. [Subsystem 4 — the cofold structure pipeline and cofold-derived heads](#8-subsystem-4--the-cofold-structure-pipeline)
9. [Subsystem 5 — MoE active-specialist + MOE multi-kernel + final assembly](#9-subsystem-5--moe-active-specialist--moe-multi-kernel--final-assembly)
10. [Variance matching and the adversarial P(test) scale](#10-variance-matching-and-the-adversarial-ptest-scale)
11. [Set-1 error analysis](#11-set-1-error-analysis)
12. [Negative results and design findings](#12-negative-results-and-design-findings)
13. [Reproducibility and the "not preserved" ledger](#13-reproducibility-and-the-not-preserved-ledger)

---

## 1. Executive summary

The winning submission is a **layered ensemble** built up in five conceptual tiers:

```
                      ┌─────────────────────────────────────────────────────────┐
Tier 0  BASE PILLARS  │  ConfTTA · Holo  (Uni-Mol v2 84M, 3D)                    │
                      │  SVM_20K · Gated_5mod  (frozen-embedding heads)          │
                      └─────────────────────────────────────────────────────────┘
Tier 1  QUAD          │  weighted, variance-matched mix of the 4 pillars         │
                      │  + per-mol cofold additives (all3_lgbm, ifp_svm)         │
Tier 2  COMPOUND      │  + 5% sw3way SVR  + 2% PharmHGT-v2  (orthogonal deltas)  │
Tier 3  MoE           │  active-specialist Boltz-cofold SVR gated by SOTA×ipTM   │
Tier 4  MOE-mk        │  + 20% commercial-MOE multi-kernel active-specialist     │
                      └─────────────────────────────────────────────────────────┘
Final                 │  final = mean + 1.08·(mix − mean)   (variance match)     │
```

The final production file is `outputs/final/MoE_v2_plus_MOE_multikernel_a05_w20.csv` (513 rows, prediction mean 4.8105, std 0.7846).

Three design principles carried the solution:

1. **Diversity by construction, not by accuracy.** The strongest single model (ConfTTA, OOF RAE 0.3830) is not enough. Repeated leaderboard gains came from adding *decorrelated-error* components — Holo (a binding-pose twin of ConfTTA, corr 0.971 but diverging on >56% of test mols), 3-way frozen encoders pretrained on external corpora, and cofold binding-pose signal — even when those components were individually weak on out-of-fold (OOF) validation.
2. **Additive, low-weight, variance-matched injection.** New components are added as small signed deltas after variance-matching to the reference distribution, never by replacing strong components. Blanket replacements consistently hurt the leaderboard (LB); tiny additive injections helped.
3. **Gate the noisy signal.** Cofold-pose signal helps only when confidence-gated and restricted to the active region. The Boltz-cofold expert LB-*reversed* as a flat additive and only helped as a per-molecule-gated active-specialist.

The single largest structural lever late in the competition was the **Mixture-of-Experts (MoE)** layer, which broke a 7-day plateau by routing active-suspect molecules to a Boltz-cofold specialist, and the **commercial-MOE multi-kernel additive**, whose dose-response walked the LB from 0.4941 down to the winning **0.4897**.

---

## 2. Data

All primary data is from the OpenADMET PXR (NR1I2) challenge release.

### 2.1 Primary supervised dose–response set (`data/raw/pxr-challenge_TRAIN.csv`)
- **4,139 rows** (as of the 2026-05-24 refresh), all with non-null `pEC50`, all unique SMILES.
- 18 columns including `pEC50`, `Emax_estimate (log2FC vs. baseline)`, standard-error and 95%-CI columns for each estimate, `Split`, `OCNT_ID`, `source`.
- **pEC50 distribution:** min 1.61, max 7.549, mean **4.3211**, std **1.1216**; **31.72%** active (pEC50 > 5).

**The "4,139 vs 4,140" note.** The raw `TRAIN.csv` currently has 4,139 rows. The number 4,140 appears throughout the frozen code and artifacts (`oof_predictions.csv`, `chemprop_hf_train.csv`, the OOF arrays) because those were built from the earlier HF-training snapshot `data/processed/chemprop_hf_train.csv` (4,140 rows). The 2026-05-24 refresh dropped one row to 4,139. Both numbers are correct for their respective files; the frozen OOF arrays preserve the 4,140 used at train time.

### 2.2 Counter-assay set (`data/raw/pxr-challenge_counter-assay_TRAIN.csv`)
- **2,859 rows**, same 18-column schema. This is the PXR-null counter-screen used for the selectivity filter (§3.2).

### 2.3 Single-concentration high-throughput screen (`data/raw/pxr-challenge_single_concentration_TRAIN.csv`)
- **21,014 rows** covering **2,746 unique molecules**. Columns include `concentration_M`, `log2_fc_estimate`, `t_statistic`, `p_value`. This low-fidelity signal is the raw source of the pseudo-pEC50 pre-training corpus.

### 2.4 Blinded test (`data/raw/pxr-challenge_TEST_BLINDED.csv`)
- **513 compounds**, split into **Set 1 = 253** (live leaderboard) and **Set 2 = 260** (held out for final ranking).

### 2.5 Low-fidelity (LF) pre-training corpus (`data/processed/chemprop_lf_pretrain_expanded_all.csv`)
- **20,191 rows**, all unique SMILES, 2 columns (`SMILES, pEC50`), pEC50 range 1.72–9.15, mean 4.265.
- **Composition (10,875 + 9,316 = 20,191):**
  - **10,875** original OpenADMET single-concentration PXR molecules with **calibrated pseudo-pEC50 = 1.1452·log2_fc + 3.6058** (calibration fit on 2,392 mols having both single-conc and dose-response labels; R²=0.445, validation MAE 0.457 on the dose-response holdout). This base file (`chemprop_lf_pretrain.csv`) has pEC50 min 1.72 / max 8.11 / mean 4.145 / std 0.608.
  - **9,316** external same-endpoint PXR records (`data/external/external_pxr_unified.csv`): **6,746** PubChem AID 1346982 (PXR qHTS agonist) + **1,861** ChEMBL (CHEMBL3401) + **709** PubChem AID 1346985 (Tox21 PXR). Quality tiers: 880 pChEMBL gold (q=3.0), 981 ChEMBL EC50/AC50 (q=2.5), 2,602 PubChem AC50 (q=2.0), 4,853 binary-only (q=0.5). Zero test overlap; 123 train overlaps.
- Combined LF distribution is heavily inactive-skewed (~11,300 mols in the 3.95–4.69 band).
- The production pretrain used the `_all` (20K, includes binary) variant. An alternative dose-response-only `_hq` variant exists at 15,338 mols but was not the production corpus.

### 2.6 The +440 high-activity pool (post-production; NOT in the winning file)
A HuggingFace data drop dated **2026-05-28** added **+440** high-activity molecules (htchem + semi-pure, corrected pEC50, ≈+0.34 log real correction) as *new* molecules — the main TRAIN.csv labels were unchanged (canonical-SMILES overlap with the 4,139 train = 5 mols → **435 net-new**). **This drop post-dates the winning submission** (`MoE_v2_plus_MOE_multikernel_a05_w20.csv`, file timestamp 2026-05-22), so it is **not part of any production component**. It is used *only* in the post-unblind retrain variants documented in the aggressive-variant README. In particular it is **not** the training set of the production MOE multi-kernel active-specialist (§9.B) — that specialist is fit on the top-30% actives of the *main* counter-assay-filtered train (see §9.B). The frequently-seen figure "~1,124 mols" is 30% of the ~3,748-mol filtered main train, not a count of the +440 pool.

---

## 3. Validation methodology

### 3.1 Cross-validation splits
The canonical scheme is **5-fold `sklearn.model_selection.GroupKFold` grouped on Bemis–Murcko scaffolds**: `MurckoScaffoldSmiles(mol=m, includeChirality=False)` mapped to integer group codes. Molecules with no valid scaffold fall back to their SMILES as the group. This is used across every production training script (`_adversarial_weighted_cv.py:86-96`, `conftta_retrain_aug.py:29-37`, `extract_molfeat_embeddings.py get_scaffold_groups()`, etc.). All fold splits, meta-weights, and the variance-match scale are deterministic with fixed seeds.

Note: the prose report `PXR_Model_Report.md §4` describes "5-fold Butina scaffold clustering (ECFP4, Tanimoto cutoff 0.4)" — this is a descriptive framing; the actual code path is Murcko-scaffold GroupKFold. Both enforce chemical-cluster-disjoint OOF.

### 3.2 Counter-assay (selectivity) filter
Before training, non-selective actives are dropped:
```
SI  = pEC50(main) − pEC50(counter)     # only where a matched counter measurement exists
drop row  IFF  SI is defined AND SI < 0   # counter more potent than main → likely PXR-null artifact
```
Molecules absent from the counter-assay are kept. Verified reproduction: **391 molecules dropped** (stable across both the SMILES-merge and index-based methods), leaving **3,748** rows on the current TRAIN.csv (4,139 − 391) / **3,749** on the older 4,140-row snapshot. The production `.npy` components (built when the repo TRAIN.csv had 4,140 rows) recorded a filtered set of **3,746** (architecture.md:60). The number *dropped* (391) is exact; the *kept* count drifts by a few rows across data versions and RDKit canonicalization.

### 3.3 The OOF/LB relationship — a hard lesson
A central, hard-won finding: **out-of-fold CV does not faithfully track the held-out Set-2 ranking**, and even the Set-1 leaderboard is only a *coarse* validator. Concretely:
- SVM_20K's 10K-LF variant was *better* on OOF (RAE 0.4923) than the 20K variant (RAE 0.5028), yet the 20K variant gave a true LB improvement of −0.0015. Distribution shift hurts OOF but helps LB.
- The 3-way SVR's LB dose-response peaked at 5% weight even though OOF predicted a 3% peak.
- Numerous strict OOF-gate-passing candidates LB-reversed (all 3D-descriptor additives, xRFM on existing embeds, POWER transforms).

Consequently, **every weight and every additive in the final system was chosen against the live leaderboard, not against OOF**, with OOF used only as a coarse filter and orthogonality gate.

### 3.4 Gates for admitting a new additive
A candidate component had to satisfy, before an LB test was spent on it:
- **Standalone OOF RAE < 0.55** (weak-but-not-useless), AND
- **max test-prediction correlation to every existing base component < 0.85** (genuine orthogonality).

These gates were necessary but not sufficient (many gate-passers still LB-reversed); the final arbiter was always the leaderboard.

---

## 4. System architecture — the layered assembly

The full production formula, layer by layer (`claude_logs/overview/architecture.md`):

**Tier 0/1 — QUAD base (`QUAD_4983`):**
```
QUAD_4983 = 0.75  · vm(0.5·ConfTTA + 0.5·Holo, ens12)      # deep-pillar block
          + 0.125 · vm(SVM_20K,   ens12)
          + 0.125 · vm(Gated_5mod, ens12)
          + w_lgbm(m) · (vm(all3_lgbm, ens12) − ens12)      # per-mol cofold additive
          + w_ifp(m)  · (vm(ifp_svm,   ens12) − ens12)      # per-mol cofold additive
```
with per-molecule confidence-weighted cofold weights
```
w_lgbm(m) = 0.03 · cf(m),   w_ifp(m) = 0.02 · cf(m)
cf(m)     = clip(1 + 2.0·z(cofold_conf(m)), 0, 5)           # alpha = 2.0
```
Component effective weights inside the block: ConfTTA 0.375, Holo 0.375, SVM_20K 0.125, Gated_5mod 0.125.

**Tier 2 — compound layer:**
```
compound = 0.93 · QUAD_4983
         + 0.05 · vm(sw3way_svr,  QUAD_4983)
         + 0.02 · vm(pharmHGT_v2, QUAD_4983)
```
(Locked 2026-05-11; LB 0.4956.)

**Tier 3 — MoE active-specialist:**
```
pred(m) = (1 − w(m))·sota(m) + w(m)·specialist_boltz(m)
w(m)    = base_w · gate(m) · per_mol(m)
gate(m) = sigmoid((sota(m) − p70_sota) / 0.10)
per_mol(m) = 1 − alpha + alpha·ipTM(m)          # alpha = 0.5
base_w  = 0.30
```
(MoE_v2_iptm_a05_w30; LB 0.4941.)

**Tier 4 — MOE multi-kernel additive and final scale:**
```
+ 0.20 · MOE_multikernel_active_specialist   (added through the MoE gate)
final = mix.mean() + 1.08·(mix − mix.mean())
```
(Production file; LB 0.4897.)

`vm(x, ref)` is variance-matching (§10.2). `ens12` is the historical 12-component reference distribution used as the variance-match anchor for the QUAD block.

---

## 5. Subsystem 1 — the two deep 3D pillars

**ConfTTA and Holo carry the largest single weight in the winning ensemble** (0.75 of the QUAD deep-pillar block, split 50/50). They share one backbone and one pretraining checkpoint, differing only in the 3D coordinates fed at fine-tune/inference time: **ConfTTA uses an RDKit free-state conformer; Holo uses the cofolded protein-bound ligand pose.**

### 5.1 Backbone — Uni-Mol v2 84M
Pair-aware 3D molecular transformer (`unimol_tools/models/unimolv2.py:668-707`, `molecule_architecture()`):
- `num_encoder_layers = 12`, `encoder_embed_dim = 768`, `num_attention_heads = 48`, `ffn_embedding_dim = 768`
- `pair_embed_dim = 512`, `pair_hidden_dim = 64` (pairwise/edge channel with triangle-multiplication + outer-product-mean updates)
- `dropout = 0.1`, `attention_dropout = 0.1`, `activation_dropout = 0.0`, `activation_fn = "gelu"`
- Reads atom-level features + pairwise distances; a scalar regression head outputs pEC50.

### 5.2 Three-stage transfer chain
Phase-1 is **not** from scratch. `MolTrain(model_name='unimolv2', model_size='84m')` auto-loads the DP Technology public self-supervised Uni-Mol v2 84m checkpoint (`unimolv2_84m.pt`, from HuggingFace `dptech/Uni-Mol-Models`), then continues training on the PXR-LF corpus. The full chain is:

> **public SSL pretrain → PXR-LF pseudo-pEC50 continued-pretrain → PXR-HF pEC50 fine-tune**

**Phase 1 — LF continued-pretrain** (`_exp_unimol_v2_mf_20k.py:67-107`, config byte-verified):
- data: `chemprop_lf_pretrain_expanded_all.csv` (**20,191 mols**)
- `epochs=30`, `batch_size=8`, `learning_rate=1e-4`, `split='random'`, `kfold=1`, `early_stopping=10`, `max_norm=10.0`, `seed=42`, `remove_hs=false`, `warmup_ratio=0.03`, `use_amp=true`, `target_normalize=auto` (StandardScaler on the LF pseudo-pEC50), `load_model_dir=null` (starts from the public SSL checkpoint)
- output: `v2_lf_pretrain/model_0.pth` (~334 MB) + a one-ETKDG-conformer SDF cache.

**Phase 2 — HF fine-tune, 5-fold scaffold CV** (`_exp_unimol_v2_mf_20k.py:109-176`, config verified):
- HF target: the 4,140 competition dose-response pEC50 (frozen at train time; the raw file was later trimmed to 4,139).
- CV: **5-fold GroupKFold on Murcko scaffolds**.
- Per fold: `epochs=50`, `batch_size=8`, `learning_rate=5e-5`, `early_stopping=10`, `max_norm=10.0`, `seed=42`, `warmup_ratio=0.03`, `use_amp=true`, `target_normalize=auto`, and crucially `load_model_dir='outputs/unimol_v2_mf_20k/v2_lf_pretrain'` (each fold initialized from the Phase-1 LF checkpoint).
- Each fold saves `model_0.pth`, `oof_fold{i}.npy`, `va_idx_fold{i}.npy`, `test_fold{i}.npy`.

### 5.3 The "TTA" name is largely historical (honest detail)
unimol_tools generates a **single conformer per molecule**: `inner_smi2coords()` does `AllChem.EmbedMolecule(mol, randomSeed=42)` (ETKDG) + `AllChem.MMFFOptimizeMolecule`; `method='rdkit_random'`, `mode='fast'`, `seed=42`, hydrogens kept (`remove_hs=false`). **There is no internal multi-conformer averaging in production.** The genuine averaging is the **mean over the 5 fold models** (`test_avg = np.mean(test_preds, axis=0)`). A separate 4-conformer-seed TTA was explicitly tested and found null (corr 1.00 to ConfTTA, 0 mols changed).

### 5.4 Holo — binding-pose Uni-Mol v2 (`_exp_holo_unimol.py`)
Identical to ConfTTA except the 3D input: instead of an RDKit conformer, Holo feeds the **cofolded ligand binding pose** — the ligand 3D coordinates extracted from a Boltz cofolded PXR–ligand complex, read via `Chem.MolFromMolFile(sdf_path, removeHs=False)` from `fixed_structures/{act_train,act_test,struct_test}_{molid}/ligand.sdf`. Same MolTrain call (`epochs=50, batch_size=8, lr=5e-5, early_stopping=10, max_norm=10.0, seed=42, remove_hs=False, model_size='84m', load_model_dir=<v2_lf_pretrain>`), same 5-fold scaffold CV on the 4140 HF set, test513 predicted per fold and averaged.

**Pose coverage:** the original `test.sdf` contains **511 of 513** posed test mols (2 lack a cofold pose → filled from the frozen Holo array). The aligned OOF (4140 long) has **6 NaN** (train mols without a usable pose); at blend time these 6 are imputed to the mean. `test_aligned.npy` (513) is fully filled.

### 5.5 Standalone quality and correlation (byte-verified on frozen arrays)
Metric = challenge RAE = MAE/MAD(y), against `oof_predictions.csv` y_true (N=4140):

| Model | OOF RAE | OOF MAE | Spearman | R² | test mean / std |
|---|---:|---:|---:|---:|---|
| **ConfTTA** | **0.3830** | 0.3485 | 0.892 | 0.783 | 4.793 / 0.742 |
| **Holo** | 0.4353 | 0.3958 | 0.848 | — | 4.867 / 0.731 |

- ConfTTA is the **strongest single model in the entire pipeline.** Per-fold RAE = [0.3544, 0.3512, 0.4159, 0.4115, 0.3967], mean 0.3859.
- Holo's OOF is computed on 4,134 non-NaN of 4,140.
- **corr(ConfTTA, Holo) = 0.971 (OOF), 0.969 (test)** — highly correlated (one backbone), as expected. Holo's value is *decorrelated error*, not accuracy: the two diverge on >56% of test mols.
- OOF blends: 50/50 → RAE 0.3977; 70/30 (ConfTTA/Holo) → RAE 0.3887. Both are slightly worse than ConfTTA alone on OOF, but the block improved the LB via error cancellation.

### 5.6 Two documented pillar blend regimes (to avoid confusion)
- **Production QUAD (winning file):** the `architecture.md` spec defines the CT-Ho block as **50/50** — each of ConfTTA/Holo at effective weight **0.375** inside the 0.75 block, i.e. `0.75·vm(0.5·ConfTTA + 0.5·Holo)`.
- **30% Holo / 70% ConfTTA:** `0.7·ConfTTA + 0.3·Holo` — the earlier LB-0.5013 breakthrough, reused as the pillar definition in the Phase-2 retrain pipeline (`build_phase2_submissions.py:26`).

> **Which blend was actually shipped — an honest ambiguity.** The two regimes are near-indistinguishable at the winning file's precision: the 50/50 pillar correlates **0.9876** with the winning CSV vs **0.9859** for 70/30 — a difference too small to adjudicate, since both differ from the shipped file only through the downstream MoE/MOE-mk/scale layers. We document both: `architecture.md` records 50/50 for the production QUAD, while the `PXR_Model_Report.md` prose and the Phase-2 retrain pipeline use 70/30. The architecture formula below (§4) is written with the 50/50 spec.

### 5.7 Training environment
`unimol` conda env, single GPU (`CUDA_VISIBLE_DEVICES=0`). Each fold ≈ 1 hour.

---

## 6. Subsystem 2 — frozen-embedding heads

Two of the four base pillars, each entering the QUAD at **12.5%**. Both train on a shared per-molecule frozen-feature library.

> **Scope note:** the multi-kernel Tanimoto⊕RBF SVR mentioned in the challenge writeup is *not* SVM_20K — it is the separate final MOE-multikernel component (§9.B) on commercial-MOE descriptors. **SVM_20K is a single-kernel RBF-Nystroem SVR.**

### 6.1 Shared frozen-feature library (`work/feat_lib.py`)
All heads assemble features by canonical-key lookup so train/holdout/test get identical construction.

| Block | Dim | Source |
|---|---:|---|
| MiniMol | **512** | `minimol_embeddings.npz` (train 4140×512, test 513×512) |
| MoLFormer-XL | **768** | `ibm_research__MoLFormer_XL_both_10pct_pooler.npz` (pooler output) |
| Chemprop LF-10K MPN | **300** | `emb_aug/lf_embeddings.npz` |
| Chemprop LF-20K MPN | **300** | `emb_aug/lf_embeddings_20k.npz` |
| MACCS | **167** | RDKit `MACCSkeys.GenMACCSKeys` (full 167-bit incl. bit-0) |
| Gobbi Pharm2D (ph4) | **39** | RDKit `Gobbi_Pharm2D`, first 39 bits of the 39,972-bit FP |
| Mordred (2D) | **1307** (universe) / **1303** (production cache) | `mordred_universe.npz` |
| Gasteiger charge stats | **5** | `[mean, std, max, min, range]` of Gasteiger charges |

The ph4 dim is a deliberate 39-bit truncation. MACCS is the full 167-bit array (the "166D" docstring in `_exp_svm_improved.py:62` is a comment error). CV = 5-fold GroupKFold on Murcko scaffolds.

### 6.2 SVM_20K — semi-supervised RBF-Nystroem SVR
**Feature vector (D = 1786):** `concat[ MiniMol(512), MoLFormer(768), Chemprop-LF20K-MPN(300), MACCS(167), ph4(39) ]`.

**Per-fold pipeline** (`work/train_svm20k.py:48-58`):
1. `StandardScaler().fit(X_tr)` (train fold only).
2. `Nystroem(kernel="rbf", n_components=500, gamma=1.0/D, random_state=42)` → **gamma = 1/1786 ≈ 5.6e-4**.
3. `SVR(C=1.0, epsilon=0.1)` (all other params sklearn defaults).
4. `oof[va] = predict(val)`; test/V = mean over the 5 folds.

**The "20K" semi-supervised source:** the LF20K MPN embedding comes from a Chemprop D-MPNN pretrained on **20,191 single-concentration + cross-source pseudo-pEC50 molecules** (`chemprop_mf_all20k/lf_data_s0.csv`). Pretrain config: depth=3, hidden_size=300, ffn_num_layers=2, ffn_hidden_size=300, dropout=0.2, epochs=30, batch_size=64, init/final_lr=1e-4, max_lr=1e-3, loss=mse, aggregation=mean, features_generator=rdkit_2d_normalized, activation=ReLU, ensemble_size=1. The 300-D MPN embeddings are extracted to `embeddings_s0/{train,test}_mpn_emb.csv`.

**Result (production `svm_lf_lf20k_all`, N=3746):** OOF RAE **0.5028**, MAE 0.4418, corr_max 0.955 to base ensemble. The 10K-LF variant was slightly *better* on OOF (RAE 0.4923) but the 20K variant was chosen because it gave a real **LB improvement of −0.0015** despite the worse OOF (a textbook OOF/LB divergence). Ensemble weight: **12.5%**.

### 6.3 Gated_5mod — learned gated multi-modal fusion MLP
**Five modalities:**

| Key | Dim | Content |
|---|---:|---|
| `emb` | **1280** | MiniMol(512) ⊕ MoLFormer(768) |
| `lf10` | **300** | Chemprop LF-10K MPN |
| `lf20` | **300** | Chemprop LF-20K MPN |
| `mordred` | ~1303–1307 | Mordred 2D, clipped to [−1e6, 1e6] |
| `charges` | **5** | Gasteiger charge stats |

Keeping **both** lf10 and lf20 as separate gated streams was a deliberate fix: an earlier "Gated 20K" that *replaced* lf10 with lf20 LB-reversed (+0.0042). The gate learns to reweight them.

**Gating architecture** (`GatedFusion5Mod`, hidden=128, dropout=0.3):
- **Per-modality encoder:** `Linear(d→128) → LayerNorm(128) → GELU → Dropout(0.3)`.
- **Gate:** `Linear(128·5 → 5)` over the concatenation of encoded modalities → `softmax` over 5 modalities → per-sample attention weights.
- **Fusion:** `fused = Σ_k gate_k · z_k` (soft attention over modalities, output 128-D).
- **Head:** `Linear(128→128) → LayerNorm → GELU → Dropout(0.3) → Linear(128→1)`.

**Training:** 10 seeds (1–10) × 5 scaffold folds; `AdamW(lr=2e-3, weight_decay=1e-4)`, `CosineAnnealingLR(T_max=150)`, **150 epochs**, batch 64; **train loss SmoothL1**, **early-stop/model-select on validation L1** (best-val state cloned each epoch, no patience cap). Per-modality StandardScaler on train fold; mordred clipped ±1e6 before scaling. OOF = mean over 10 seeds; test/V = mean over folds each averaged over 10 seeds.

**Result:** production gated head OOF RAE **0.4980** (exactly matched the old LB 0.4980). Ensemble weight: **12.5%**. Test-set correlation to SVM_20K = **0.944** (they share MiniMol/MoLFormer/LF features).

### 6.4 Counter-assay filter (production generation)
Both production heads drop non-selective actives with the SI<0 rule (§3.2). Verified counts: mols with computable SI = 2,647; dropped = 394 (current env) vs 391 (architecture.md); kept = 3,745 (current) / 3,746 (production `.npy`). The Phase-2 `work/` scripts instead read pre-filtered setting CSVs (`train_base.csv` = 4140, `train_set1=4342`, `train_hi=4575`, `train_set1_hi=4777`, plus a 51-mol V holdout).

---

## 7. Subsystem 3 — the compound layer

A thin layer on top of the QUAD base adding exactly two small, deliberately-orthogonal heads as low-weight signed deltas:
```
compound = 0.93·QUAD_4983 + 0.05·vm(sw3way_svr, QUAD_4983) + 0.02·vm(pharmHGT_v2, QUAD_4983)
final    = compound.mean() + 1.08·(compound − compound.mean())
```
Locked 2026-05-11; LB **0.4956**.

### 7.A sw3way — 3-way frozen-pretrain SVR
**Thesis:** encoders pretrained on *different corpora* (PubChem BioAssay / UniChem / causal-LM) produce truly orthogonal errors, unlike Uni-Mol clones that converge to the same manifold. This head broke a 56-hour LB plateau.

**Three frozen encoders (concatenated to 4608-D):**

| Block | Model | Emb dim | Pooling |
|---|---|---:|---|
| CLAMP | `ml-jku/CLAMP` (ICML 2023), contrastive language-assay MLP over 8192-D FP, pretrained on ~500K PubChem BioAssay datapoints | **768** | compound-encoder projection |
| da4mt-MTR | `UdS-LSV/da4mt-mtr-60`, 2-stage MLM+MTR chemistry BERT, 88.89M params | **768** | mean-pool of last hidden state, `max_length=128` |
| ChemFM-3B | `ChemFM/ChemFM-3B`, 3B-param LLaMA-style causal chemical LM, trained on 178M UniChem molecules | **3072** | bf16 autocast, mean-pool over mask, `max_length=256`, batch=8 |

Verified dims: 4140×768, 4140×768, 4140×3072 → **total D = 4608**.

**Standalone block gates (own SVR-RBF head, 5-fold scaffold OOF, N=3746):** CLAMP OOF RAE 0.6305 (corr_max 0.756); da4mt OOF RAE 0.6009 (corr_max 0.828).

**Deployed head (SVR on the 4608-D concat):** two scalers — (1) per-block `StandardScaler` on the full setting-train, then concatenate; (2) a second global `StandardScaler` per fold on the training rows. Regressor:
```python
SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.1)
```
5-fold GroupKFold (Murcko), OOF = held-out folds, test = mean of 5 folds.

**Deployed metrics:** **OOF RAE 0.5448** (PASS <0.55). Test-pred correlations: ConfTTA 0.8407, Holo 0.8433, SVM_20K 0.8498 (all borderline-PASS <0.85). Test pred mean 4.8725 / std 0.5675. A Ridge alternative on the same 4608-D scored OOF 0.5507 (alpha=3000); an `l2row_svr` head scored OOF 0.5394 but was **never deployed** — production uses the SVR-RBF. Multi-seed 3-way SVR was corr 0.9997 to single-seed (seed averaging null).

**Entry (5% additive):** the test prediction is variance-matched to base, then **zero-centered** and added as a signed delta: `mix = base + 0.05·(vm(sw3way) − mean(vm(sw3way)))`, i.e. effectively `0.93·base + 0.05·vm(sw3way)`. LB dose-response: @3% → 0.4962, @5% → **0.4957** (deployed; the empirical LB optimum, *not* the OOF-predicted 3% peak).

### 7.B PharmHGT-v2 — pharmacophore heterogeneous-graph transformer
**Thesis:** a from-scratch graph model providing a fully non-embedding inductive bias (learned message passing on a BRICS-pharmacophore heterograph). Source: `third_party/PharmHGT/`; driver `scripts/active/_pharmhgt_v2_multiseed.py`.

**Graph — dual-view BRICS heterograph** (`data.py` `Mol2HeteroGraph`):
- Node types `a` (atoms), `p` (pharmacophore fragments = BRICS fragments).
- Edge types `('a','b','a')` bonds, `('p','r','p')` BRICS reaction bonds, `('a','j','p')`/`('p','j','a')` junction edges.
- Feature dims: `atom_dim=42`, `bond_dim=14`, `pharm_dim=194` (MACCS 167 + 27 pharmacophore property types per fragment), `reac_dim=34` (two 17-dim BRICS one-hots), junction = 42+194 = **236**.

**Architecture** (`model.py:213-276`): `hid_dim=300`, `depth=3`, ReLU, `num_task=1`. Two MVMP multi-view message-passing modules (`mp` on atom view, `mp_aug` on atom+pharm view), each 4-head attention, depth-3, `cross_reducer='sum'`. **Readout = `Node_GRU`**: 6-head cross-attention mixing atom hidden states against pharmacophore hidden states, then a bidirectional GRU (hid_dim→hid_dim, direction=2), mean-pool over fragment nodes → 600-D per view; two readouts concatenated → **1200-D graph embedding**. Prediction head `Linear(1200→300)→ReLU→Linear(300→300)→ReLU→Linear(300→1)`, Xavier-normal init.

**Training** (as-run from `diag.json`/`launch.log`): epochs=40, batch=64, `init_lr=1e-4`, `max_lr=5e-4`, wd=0.0, Adam + **OneCycleLR** (`pct_start=0.05`, cosine, `final_div_factor=10`), MSE, grad-clip 5.0. Per-fold label z-normalization (predictions un-scaled). Early-stop select by validation RMSE (full 40 epochs, best-val kept). 5-fold scaffold splits reused from the canonical `va_idx_fold{f}.npy`; same counter-assay filter → filtered N=**3749**. **3 seeds {42, 137, 2026}**, each × 5 folds; per-fold seed = `seed·1000 + fold`; OOF/test averaged across seeds.

**Metrics:** 3-seed-avg **OOF RAE 0.5629** (MAE 0.4944, R² 0.6137, Spearman 0.7248) — **FAILS** the strict <0.55 gate by 0.013. Per-seed OOF RAE 0.5824/0.5894/0.5865; 3-seed averaging gave −0.027 over v1 (0.5901→0.5629). Test-pred corr: ConfTTA 0.8263, Holo 0.8172, SVM_20K 0.8204 (all PASS <0.85 — "best non-3D candidate ever"). Test pred mean 4.8490 / std 0.5687.

**Entry (2% additive):** same vm + zero-center delta as sw3way, stacked: `mix = base + 0.05·sw_delta + 0.02·ph_delta`. LB effect: compound (3-way @5% + PharmHGT @2%) = **0.4956** vs 3-way @5% alone 0.4957 — a real but tiny **−0.0001**, limited by PharmHGT-v2's weak standalone OOF. Later variants (v3-orth008 OOF 0.5424 corr 0.849 strict-PASS; v4 atom-IFP) either LB-reversed or gave no gain, so **v2 remained deployed**.

---

## 8. Subsystem 4 — the cofold structure pipeline

Protein–ligand cofolding is a distinct inductive-bias source. Cofold enters the winning system three ways: (1) per-molecule additive heads (`all3_lgbm`, `ifp_svm`) confidence-scaled into the QUAD; (2) the Boltz-embedding active-specialist in the MoE layer (§9.A); (3) the Holo binding-pose Uni-Mol blend (§5). This section covers (1) plus the shared feature-extraction pipeline.

### 8.1 Cofold structure generation
**Batch A — the primary 4728-complex set (user-provided, 2026-05-02).** `fixed_structures/` (208 MB), covering essentially all train (4137/4140) + test (513) + extra `struct_test` mols ≈ 4728 complexes. Each directory `{act_train|act_test|struct_test}_OADMET-XXXXXXX/` holds `protein.pdb` (293 residues, 2361 atoms, no H), `ligand.sdf` (binding-pose 3D with H), `confidence.txt` (a single cofolding confidence scalar, e.g. 0.949529). Attributed to Boltz cofolding of each ligand into the PXR LBD (UniProt O75469). The generation script for this batch is **not preserved** — only the structures.

**Batch B — the +440 reproduction (`scripts/boltz_cofold_440.py`, preserved).** Exact settings for the 435 new high-activity ligands:
- Boltz version **2.2.1**.
- PXR-LBD sequence hard-coded, length **293 aa**, single protein chain `id: A` (begins `GLTEEQRMMIRELMDAQMK…GITGS`).
- **Single-sequence / empty MSA** (`msa: empty`) → no evolutionary profile.
- Ligand chain `id: B`, `smiles`.
- CLI: `boltz predict <yaml_dir> --out_dir <...> --accelerator gpu --devices 1 --output_format mmcif --seed 42 --override --no_kernels` (`--no_kernels` for portability; single-GPU sequential).
- Post-processing (`boltz_cofold_postprocess.py`): parse mmCIF, extract ligand-chain heavy-atom coords in file order, map onto the RDKit mol from SMILES (Boltz preserves SMILES atom order), validate element order, set conformer, write `ligand.sdf`; skip on mismatch.

Raw confidence-scalar statistics (verified): all-mols mean **0.9461 ± 0.0094** (min 0.8431, max 0.9697); test subset mean **0.9483 ± 0.0081** (min 0.9146, max 0.9692).

### 8.2 Cofold feature families (all from Batch-A poses)
All extractors scan `fixed_structures/`, map `dir_name → OADMET-ID`, and save `outputs/cofold_{family}/{family}_features.npz`. Verified on-disk dims:

| Family | Shape | Dim | Content |
|---|---|---:|---|
| **IFP** (ProLIF) | (4723, 168) | 168 | binary (residue, interaction) bits; 13 interaction types; obabel `-h` protonation; one-hot union |
| **ESP** (ESP-DNN) | (4716, 20) | 20 | ESP-DNN partial charges → 20 conformer-invariant stats (mean/std/max/min/range/median/percentiles/count thresholds/Σ\|q\|/…) |
| **pose_charges** | (4716, 29) | 29 | pose-dependent charge×pocket: 5 proximity-weighted charge stats (w=exp(−d_min/4Å)), 5 residue-groups × 4 stats, 1 Coulomb proxy, 3 shell densities |
| **mordred3d** | (4723, 1826) | 1826 | Mordred `ignore_3D=False` on the binding-pose conformer |
| **tier2 pair_dist_hist** | (4723, 40) | 40 | 8 element-pairs × 5 distance bins |
| **tier2 residue_dist_density** | (4723, 600) | 600 | per-residue (300 slots) min ligand-atom distance ⊕ contact density Σexp(−d/4) |
| **tier2 confidence** | (4723, 1) | 1 | raw cofolding confidence scalar (default 0.95 if absent) |

### 8.3 Cofold-derived heads

**`all3_lgbm` — the orthogonal cofold-signal carrier** (`_exp_cofold_more_models.py`):
- Feature = `concat[IFP(168), ESP(20), pose_charges(29)]` = **217-D**, cofold-only (no 2D/embedding features).
- Model: `LGBMRegressor(n_estimators=300, max_depth=5, num_leaves=15, learning_rate=0.03, reg_alpha=1.0, reg_lambda=1.0, min_child_samples=10, random_state=42)`, per-fold StandardScaler, test = mean over 5 folds.
- Validation: 5-fold Murcko-scaffold GroupKFold on the **unfiltered 4140** set.
- Result: **OOF RAE 0.6937**, **corr_max 0.499** to QUAD refs — a *weak* standalone predictor but *maximally orthogonal* (ρ<0.50 to every base component; the design intent). Test mean 4.544 / std 0.553. Sibling heads on the same 217-D (documented, not shipped): all3_svm 0.7138, all3_mlp 0.7687, all3_catboost 0.6969, all3_deepmlp 0.7064.

**`ifp_svm` — a strong cofold-augmented SVM (near-clone of SVM_20K)** (`_exp_inject_cofold3d.py`):
- Feature = `base_2D ⊕ IFP(168) ⊕ mordred3d(top-200-by-variance)`, where `base_2D = MiniMol(512) ⊕ MoLFormer(768) ⊕ LF20K(300) ⊕ MACCS(167)` = 1747-D.
- Model: Nyström-RBF SVR — `StandardScaler`, `Nystroem(kernel='rbf', n_components=500, gamma=1.0/D, random_state=42)`, `SVR(C=1.0, epsilon=0.1)`, 5-fold test mean.
- Validation: 5-fold Murcko GroupKFold on the counter-assay-filtered train (OOF 3746).
- Result: **OOF RAE 0.5012**, **corr to SVM_20K = 0.976** — essentially SVM_20K with cofold features appended (strong but highly redundant), hence a small entry weight. Test mean 4.837 / std 0.636.

**`confidence_hgbrt` — per-mol confidence proxy.** Outputs are pEC50-scale predictions (test mean 4.3835 ± 0.2235), a weak HGBRT pEC50 predictor on cofold-confidence-related features. Its training script is **not preserved**. The *original* winning per-mol build (`QUAD_5067_per_mol_conf_alpha20_s108.csv`, LB 0.5030 #1) used the **raw cofolding-confidence scalar** (test mean 0.948 ± 0.008); the later `confidence_hgbrt` proxy was used for z-scoring in a subsequent compound-candidate experiment. Both are z-scored identically.

### 8.4 How cofold heads enter QUAD (per-mol confidence-weighted additives)
```
w_lgbm(m) = 0.03 · cf(m),   w_ifp(m) = 0.02 · cf(m)
cf(m)     = clip(1 + 2.0·z(cofold_conf(m)), 0, 5)     # alpha = 2.0
```
- `vm(x, ref)` variance-matches each cofold head to the ens12 reference before adding.
- **α = 2.0** was the breakthrough setting (LB 0.5030 #1). `cf(m)` ranges ~0×–2× in practice; ~46/513 test mols shift noticeably from the fixed-weight base.
- Base additive weights: all3_lgbm 3%·cf, ifp_svm 2%·cf → max cofold contribution ≈5% total, matching the observed cofold dose-response peak (~5%; hard 7% upper limit).
- Per-mol cf-weighting is **cofold-specific**: applying the same cf modulation to a non-cofold head (3-way SVR) LB-reversed +0.0016, confirming the mechanism gates *cofold-pose reliability*, not generic uncertainty.

### 8.5 Key design finding
Cofold-pose signal is a net positive only when **targeted and quality-gated**, never as a flat blend. The orthogonal-but-weak `all3_lgbm` (OOF 0.69, ρ≈0.50) and the strong-but-redundant `ifp_svm` (OOF 0.50, ρ≈0.98 to SVM_20K) are both admitted at tiny per-mol weights scaled by cofolding confidence; the small weight and confidence gate together extract binding-pose diversity while keeping the noisy signal from harming the LB.

---

## 9. Subsystem 5 — MoE active-specialist + MOE multi-kernel + final assembly

Three winning layers that sit on top of the base four-pillar ensemble.

### 9.A MoE active-specialist (Boltz-cofold expert — the plateau-breaker)

**Boltz cofold embedding = exactly 1408-D.** Internal Pairformer single (`s`) + pair (`z`) reps from Boltz-1 single-sequence cofolding. Verified 1408-D float32 per molecule. Exact pooling: `s_lig mean/max + s_pro mean + z_lig-lig + z_lig-protein`, decomposing to Boltz-1 dims s=384, z=128 as `384 + 384 + 384 + 128 + 128 = 1408`. Extraction: per-complex YAML (`msa: empty`) → `boltz predict --write_embeddings` → pool; dual-A6000 stride parallelism, ~11 h for all 4728 complexes.

**The active-specialist SVR (top-30% truncated training):**
- Model: `SVR(kernel='rbf', C=1.0, gamma='scale', cache_size=4000)`.
- **Fit only on the top-30% by potency:** `top30y = y_keep > p70(y_keep)`, where on the filtered train p70(y) ≈ 5.04.
- CV: 5-fold GroupKFold on Murcko scaffolds; per-fold StandardScaler; test = mean over 5 folds. Same counter-assay filter.
- Cached stats (`outputs/boltz_cofold_specialist/summary.json`, act_p70): OOF RAE 1.2446 (deliberately weak — a specialist, not a generalist), top30 bias −0.0156, corr_vanilla 0.5681, **corr_sota 0.5810** (the ρ≈0.57–0.58 "lowest correlation to base ensemble"), test mean 5.4585 / std 0.1558. p50/p60/p70/p75/p80 variants all cached.

**The MoE gate = sigmoid(activity) × Chai1 ipTM:**
```
pred(m) = (1 − w(m))·sota(m) + w(m)·specialist(m)
w(m)    = base_w · gate(m) · per_mol(m)
gate(m) = sigmoid((sota(m) − p70_sota) / 0.10)      # tau = 0.10
per_mol(m) = 1 − alpha + alpha·confidence(m)          # alpha = 0.5, confidence = Chai1 ipTM
base_w  = 0.30
```
- **p70_sota** = 70th percentile of the base-ensemble prediction. On `QUAD_4983_conftta75_s108.csv`, p70 = 5.249, with **exactly 154 mols above** — the rank change is confined to these 154 active-suspect mols.
- **confidence(m) = Chai1 interface-pTM** (`chai1_cofold_poses/confidence_seed11.csv`), ipTM std across test ≈ 0.113.
- **LB confirmation:** `MoE_v2_iptm_a05_w30` = LB **0.4941**, −0.0015 vs the 0.4956 plateau.
- **Key finding:** the *same* Boltz SVR as a flat additive LB-*reversed*; only the gated, active-restricted form helped. Rank-changing interventions bypass the variance-match neutralization that kills flat additives.

### 9.B MOE multi-kernel additive (the final top-ranking layer)

**Commercial MOE descriptors** (Molecular Operating Environment / CCG), a gift from user "gpcr" (`external_data/MOE_{TRAIN,TEST}.txt`):
- **341 numeric descriptors** (the report rounds to "340-D"). N_test=513, N_train=4139. Families: PEOE/SLOGP/SMR partial-charge VSA, vsurf_*/ASA/dipole/PMI/E_* 3D, BCUT/Kier/Wiener topological. Generation: CXSMILES strip → dominant-protomer Wash → **MMFF94x** minimization → 2D/3D descriptors. **Test mols included** → in-distribution, bypassing the train→test representation shift.

**Winning kernel:** `K = 0.5·Tanimoto(ECFP4) + 0.5·RBF(MOE-341D)` (precomputed, alpha=0.5), trained as an **active-specialist on the top-30% most-active compounds of the main counter-assay-filtered train** (`y > p70 ≈ 5.04`; ≈1,124 mols) — the same active-specialist construction as the Boltz expert (§9.A), but on the commercial MOE descriptors. The cached winning vector `test_mk_alpha05.npy` has mean **5.506 / std 0.146** — the tight, high-mean signature of a top-30%-actives specialist. The alpha sweep (`test_mk_alpha0{2..8}.npy`) and a full-range `test_vanilla.npy` (mean 4.756 / std 0.643) are all cached; alpha=0.5 was the LB winner.

> **Provenance caveat (honest).** All MOE-mk cached outputs and the `external_data/MOE_{TRAIN,TEST}.txt` descriptors are dated **2026-05-21**, one day before the winning file; the exact production builder script for this component is **not preserved** (only its `outputs/moe_descriptor_spec/` outputs survive). The multi-kernel form `K_mk = alpha·K_tan + (1−alpha)·K_rbf` with Tanimoto `XY/(‖X‖²+‖Y‖²−XY)` is documented, but the precise SVR `C/ε/γ` used in production are not byte-recoverable. Note the later script `build_sota_proxy_new440.py` (2026-05-28) is a **different** construction — an RBF kernel on MoLFormer+ChemBERTa embeddings, trained on the 4,140 to *proxy-label the +440 new molecules* — and is **not** this component's recipe; it must not be read as evidence for the production MOE-mk hyperparameters.

**Cached spec outputs** (`outputs/moe_descriptor_spec/`): `test_mk_alpha05.npy` (winner) mean 5.506 / std 0.146; `test_vanilla.npy` (full-range MOE SVR) mean 4.756 / std 0.643. Corr to base ρ≈0.56. Orthogonality/quality: **64.3% push-up on top30** (record; prior best Boltz_spec_p70 = 46.1%), **d30_raw = +0.0508** — the first specialist that predicts top-30% mols *above* SOTA on average.

**Additive weight 0.20, applied through the MoE gate.** Fully LB-mapped dose-response:

| MOE-mk weight | LB (Set1) | Δ vs 0.4941 |
|---|---|---|
| @3% | 0.4930 | −0.0011 |
| @5% | 0.4923 | −0.0018 |
| @8% | 0.4914 | −0.0027 |
| @10% | 0.4909 | −0.0032 |
| @12% | 0.4904 | −0.0037 |
| @15% | 0.4900 | −0.0041 |
| **@20% (production)** | **0.4897** | **−0.0044** |

The dose is linear w03→w08 (~−0.0007/step) then tapers w10→w20 (~−0.0003/step) — saturation, not a clean optimum. w20 was the last submission before Phase-1 close. A compound `TRIPLE_MOE_mk20+lgbm05+tk05` LB-reversed (+0.0007) — MOE-derived heads are too mutually correlated (mk-LGBM 0.77, mk-TK 0.957).

### 9.C Final assembly
`outputs/final/MoE_v2_plus_MOE_multikernel_a05_w20.csv` — 513 rows, **prediction mean 4.8105, std 0.7846**. Frozen-SOTA Set1 RAE **0.4875** (post-unblind recompute on 253 Set1 labels); the headline 0.4897 is the live-LB value at submission. corr(MoE_v2_iptm_a05_w30, final) = 0.9977. The final assembly builder is **not preserved**; a naive flat blend `0.8·MoE_v2 + 0.2·mk_spec` does not reproduce it (mean|Δ|=0.164), nor does a vm-aligned flat blend (mean|Δ|=0.078) — confirming the MOE-mk is added *through the per-mol MoE gate*, not uniformly.

---

## 10. Variance matching and the adversarial P(test) scale

### 10.1 Why variance matching
The test predictions are genuinely narrower than train: test std **0.79** vs train std **1.12**. The adversarial train/test classifier separates the two sets at **AUC ≈ 0.97** (severe covariate shift), confirming the variance compression is *correct*, not a bug.

### 10.2 The `vm()` operator (canonical, appears identically in 8+ scripts)
```python
def vm(p, t):  return t.mean() + (t.std() / max(p.std(), 1e-9)) * (p - p.mean())
```
Every specialist/component is `vm(comp, ens12)`-aligned before weighted mixing.

### 10.3 Final scale = 1.08
```python
final = mix.mean() + 1.08 * (mix - mix.mean())
```
Production files carry `_s108`. The scale is not to be tuned past 1.08–1.13.

### 10.4 Adversarial P(test)-weighted scale objective
The scale was chosen with an adversarial objective (`_adversarial_weighted_cv.py`):
1. **P(test) estimation:** concatenate 4,139 train + 513 test ECFP4 (2048-bit Morgan r=2), label train=0/test=1, `RandomForestClassifier(n_estimators=200, random_state=42)` with 5-fold `cross_val_predict(method='predict_proba')`; `ptest = P(class=test)` per train mol.
2. **Weights:** linear `w_i = ptest_i/mean(ptest)` and sharpened `w_i = ptest_i²/mean(ptest²)`.
3. **P(test)-weighted RAE:** `RAE_w(s) = Σ w_i·|y_i − ŷ_i(s)| / Σ w_i·|y_i − ȳ|`, minimized over `s∈[0.8, 2.0]` via `minimize_scalar(method='bounded')`; swept over `arange(0.95, 1.30, 0.01)`. The calibrated factor lands in ≈**1.08–1.17**. Production used 1.08 (inherited from the base-ensemble builders); the ADV/ENS12 sweeps generated s1.10/1.15/1.18 variants.

The `pool_high`/`pool_low` split (top 20% by ptest = 828 mols, cached at `adversarial_model_selection/adversarial_split.npz`) is also used for adversarial component evaluation.

---

## 11. Set-1 error analysis

Verified against `TEST_PHASE1_UNBLINDED.csv` (253 rows) + the production SOTA (`set1_full_analysis.py`).

### 11.1 Headline metrics (N=253)
MAE **0.3893**, RMSE **0.5940**, R² **0.6687**, Pearson **0.8200**, Spearman **0.8615**, Kendall **0.6791**, median residual −0.0054, mean residual +0.0601.

**Two RAE conventions** (source of the 0.4897 vs 0.512 spread):
- **MAE / MAD(mean-baseline)**, MAD(mean)=0.7987 → RAE **0.4875** — the competition metric; headline 0.4897 (data-version noise).
- **MAE / MAD(median-baseline)**, MAD(median)=0.7609 → RAE **0.5116** — the number in the comprehensive-analysis doc. Both correct; they differ only in mean- vs median-centered MAD.

### 11.2 Activity cliffs
Cliff = NN Tanimoto ≥ 0.5 to any train mol AND |ΔpEC50| ≥ 1.0 to that neighbor (ECFP4 2048-bit, r=2). Result: **72/253 (28.5%)** are cliffs; **cliff MAE 0.6222 vs non-cliff 0.2967** — a **2.10×** gap.

### 11.3 Measurement noise / irreducibility
Median label std-error 0.1333; Spearman(|error|, label std-error) = 0.276; **53.0%** of predictions fall inside the label's experimental 95% CI.

### 11.4 Statistical ceiling
Bootstrap MAE σ = **0.0281** (20,000 resamples). The ~0.005 gap to the Set-2 leader is **0.18 σ** — statistically invisible.

### 11.5 Region biases (exact re-verification)

| pEC50 region | n | signed bias (pred−true) | MAE |
|---|---:|---:|---:|
| **< 4** | 55 | **+0.5636** | **0.8180** |
| **4 – 5** | 86 | **+0.0123** | **0.2968** |
| **> 5** | 112 | **−0.1505** | **0.2499** |

Finer bins: <4 → +0.5636/0.818 (n=55); 4–4.5 → +0.0430/0.2842 (n=26); 4.5–5 → −0.0011/0.3022 (n=60); 5–5.5 → −0.0833/0.2144 (n=70); >5.5 → −0.2624/0.3090 (n=42).

### 11.6 Interpretation
Error is dominated by activity cliffs (28% of mols, ~2× MAE) and experimental label noise (>half of predictions inside the CI). The low-activity over-prediction / high-activity under-prediction pattern is **variance compression, which is MAE-optimal and deliberately retained**. The residual is orthogonal to ~25 structural representations (all get honest ensemble weight 0), so it is **cliff-irreducible from structure**.

---

## 12. Negative results and design findings

The winning system is as much a record of what *did not* work. Selected LB-verified negatives:

- **Naive NN label propagation** on SMILES embeddings underperforms — activity cliffs break local smoothness.
- **GBDT overfits** (0.65→0.67 LB) while **SVM generalizes** (0.65→0.64) despite identical local CV.
- **Cliff-feature stacking** hurts the LB despite local CV gains; the analog-proxy CV is untrustworthy.
- **High-correlation components hurt** even when accurate: MLP_v3 (corr 0.92) degraded LB 0.5313→0.5349. Rule: add only corr < 0.85.
- **Diversity alone is not enough:** a ChEMBL MLP (0.80 corr, OOF 0.59) hurt LB by 0.0025.
- **Dose-response law:** any *single* added component hurts LB monotonically with weight if entered wrong; 8% mix barely worse, 100% solo crashes. The fix is *many* low-weight, orthogonal, variance-matched additives.
- **Rank averaging** of 13 base components at 8% mix gave the first LB 0.5283.
- **Conformer-seed TTA is null** past one seed (4-seed avg: 0 mols changed).
- **All 3D-descriptor additives systematically LB-reverse** despite passing strict OOF gates (mordred3d @5% +0.0035; single_3d @5% +0.0030).
- **POWER amplification** |x|^1.14 LB +0.0079 (worst ever) — asymmetric extreme amplification is uniformly bad.
- **Boltz-2 affinity head is unusable** (per user); Boltz-1x cofold poses remain useful.
- **Boltz embedding as a flat additive LB-reverses** (+0.0038); only the gated active-specialist form helps.
- **Structural / dim-recombination algorithms** (PCA/KernelPCA/reverse-distillation/NNLS stack, xRFM, MDN heads, multi-task aux heads) cannot break the Pareto frontier on the existing embedding pool — the pool is exhausted; real breakthroughs required a *new label-relevant signal source* (external assay, cofold pose, QM) or a genuinely external pretrain corpus.
- **The 3-way SVR breakthrough** (LB 0.4962, −0.0018) is the exception that proves the rule: genuinely external pretrains (CLAMP/da4mt/ChemFM) LB-*improve* even when OOF under-predicts the gain, whereas Uni-Mol clones LB-reverse. Orthogonality must come from a *different corpus*, not a different loss (a ranking-loss MoLFormer was corr 0.993 to its MSE twin) or head.
- **Post-unblind:** Set-1 (253 pts) is only a coarse validator; a rescore of all 4,428 candidate CSVs found none beating SOTA by more than noise (best −0.0004). The limit is statistical (253 pts, 0.005 gap = 0.18 σ), not effort.

---

## 13. Reproducibility and the "not preserved" ledger

Fold splits, meta-weights, and the variance-match scale are deterministic with fixed seeds. The following builders were run inline or from `/tmp` and are **not preserved** in the repo, though their cached outputs survive and the recipes are recoverable:

1. **Boltz 1408-D extraction** scripts (`/tmp/extract_boltz_batched.py` etc.) — outputs cached in `outputs/boltz_cofold_embed/`.
2. **Boltz-cofold active-specialist SVR builder** (`/tmp`) — recipe recoverable from `train_molae_specialist.py` (identical framework) + cached `outputs/boltz_cofold_specialist/`.
3. **MoE gate assembly** (`MoE_v2_iptm_*.csv`) — no repo `.py` writes it; formula reconstructed from the 2026-05-19 daily log + `pxr_moe_breakthrough`.
4. **MOE-341D multi-kernel SVR builder** — outputs cached in `outputs/moe_descriptor_spec/` (dated 2026-05-21, one day before the winning file). The kernel form (`0.5·Tanimoto(ECFP4) + 0.5·RBF(MOE-341D)`, top-30%-actives specialist) is documented but the exact production `C/ε/γ` are not byte-recoverable. `build_sota_proxy_new440.py` is a *different, later* construction (a +440 proxy-labeler) and is **not** this recipe.
5. **Final MoE_v2 + MOE-mk @0.20 assembly** — not preserved; verified to be a per-mol gated add, not a flat blend.
6. **The exact single script pinning production scale = 1.08** for the final file — `vm()` and 1.08 are inherited from the base-ensemble builders; the 1.08–1.17 derivation machinery is in the two `_adversarial_*` scripts.
7. **The 20,191-row `expanded_all` LF corpus build** — consumed as a pre-built CSV; the calibration/provenance of the additional 9,316 beyond the base 10,875 was run across sessions.
8. **The QUAD_5067 per-mol confidence builder** — only its output CSV and memory record survive.

Also honestly not preserved: exact realized training steps / early-stop epoch per Uni-Mol fold (internal `metric.result` files store only unimol_tools' optimistic random-split metrics, e.g. ConfTTA fold0 internal val MAE 0.056, which are *not* the scaffold-OOF numbers), and wall-clock / per-fold conformer-embed success counts.

**Training environments:** `unimol` (Uni-Mol GPU), `wjh_fake` (PyG/DL GPU), `gems` (Chemprop), `boltz2` (Boltz 2.2.1), `esp-dnn` (Python-2.7 ESP-DNN). GPU-only training throughout.

---

*AIDD-LiLab, University of Florida — Yanjun Li Lab. This solution is described in the group's forthcoming publication.*
