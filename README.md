# Expertise modulates automation bias and sentinel behavior in human-AI collaborative diagnosis of neonatal pneumoperitoneum

[![Status](https://img.shields.io/badge/Status-In_Revision-orange)]()
[![Venue](https://img.shields.io/badge/Venue-npj_Digital_Medicine-blueviolet)]()
[![Study Design](https://img.shields.io/badge/Study-MRMC_Crossover-blue)]()
[![Analysis](https://img.shields.io/badge/Statistics-GLMM_Crossed_Random_Effects-green)]()
[![Educational Tool](https://img.shields.io/badge/Educational_Sandbox-Live-brightgreen)](https://neonatal-ai-sandbox.pages.dev/)

**Lee et al.** — submitted to *npj Digital Medicine*; currently in revision (round 1).

> **One-line takeaway:** *AI reliability does not translate linearly into clinical benefit.* In 1,750 interpretation events across a multi-reader multi-center crossover study, high AI reliability paradoxically induced **automation bias in trainees**, while error-prone AI triggered **sentinel (vigilant) behavior in experts** — demonstrating that adversarial resilience, not standalone accuracy, is the defining metric of human-AI team performance.

---

## Abstract

Medical AI is often validated under an additive assumption that algorithmic sensitivity and clinician oversight will combine to improve care. We tested this assumption in the high-stakes diagnosis of neonatal pneumoperitoneum, a time-critical surgical emergency. In a multi-reader crossover study analyzing 1,750 interpretation events, clinicians reviewed radiographs aided by either a high-reliability model or a systematically error-injected model. We found that high AI reliability paradoxically induces automation bias in trainees, who accepted 52.0% of incorrect suggestions, while offering limited gains to experts. Conversely, when challenged by flawed AI, neonatologists demonstrated "sentinel behavior," correctly overriding 91.7% of errors consistent with increased deliberation. We operationalize systemic resilience as the capacity to maintain diagnostic integrity under algorithmic failure and demonstrate that clinical validity depends on the human-AI team's adversarial resilience rather than standalone accuracy. To mitigate the risk of deskilling and never-skilling, we release an open-source educational sandbox designed to inoculate clinicians against automated errors.

**Keywords:** Neonatal pneumoperitoneum · Automation bias · Sentinel behavior · Artificial intelligence · Deep learning · Multi-reader multi-case study · Human-AI interaction · Radiology

---

## Educational Sandbox

> **Try it live:** [neonatal-ai-sandbox.pages.dev](https://neonatal-ai-sandbox.pages.dev/)

An open-source, web-based educational tool designed to inoculate clinicians against automated errors — simulating both reliable and error-prone AI assistance to build adversarial resilience in trainees and practicing clinicians.

---

## Model Architecture

![Model Arch](figures/rdino%20model_arch.png)

To ensure that differences in reader behavior were driven solely by **AI reliability** (and not model capacity), both the Reliable and Error-Injected assistants use the same underlying architecture:

- **Backbone:** RAD-DINO (ViT-B/14) — a vision foundation model pre-trained on large-scale radiology datasets
- **Adaptation (LoRA):** Parameter-efficient fine-tuning via Low-Rank Adaptation ($r=12, \alpha=24$) injected into Query/Value projections and the MLP layer; only **1.36%** of parameters were trainable
- **Sampling Strategy (RFBS):** A custom Representation-Focused Batch Sampler enforcing diversity and exposure to uncommon pneumoperitoneum distributions during training

> **Error-Injected model:** Same architecture, trained on systematically poisoned labels. False positives engineered by mislabeling clinically plausible confounders (iatrogenic devices, portal venous gas, pneumatosis intestinalis, abdominal drains) as pneumoperitoneum — curated by a board-certified pediatric radiologist to simulate realistic deployment failures rather than random noise.

---

## Why This Matters

Neonatal pneumoperitoneum is a time-critical surgical emergency. Integrating AI into this workflow is not just about model accuracy. Clinicians interact with advice, confidence cues, and time pressure.

This study investigates the **Human-AI Interaction (HAI) layer**:
1. **Automation Bias:** When AI is *highly capable*, does it help — or does it reduce human vigilance?
2. **The Sentinel Effect:** When AI is *systematically wrong*, do clinicians disengage, blindly follow, or become hyper-vigilant?
3. **Expertise Gradient:** Do neonatologists, radiologists, and residents react differently to the same AI signals?
4. **Never-Skilling Risk:** Does early-career reliance on reliable AI prevent trainees from developing independent pattern recognition?

---

## Study Overview

### Cohorts

| Cohort | Radiographs | Positive Cases | Source |
| :--- | :--- | :--- | :--- |
| Internal Development | 688 (from 216 patients) | 310 | Asan Medical Center |
| External Validation (Reader Study) | 125 | 40 | 11 tertiary hospitals via AI-Hub |

### Reader Study Design

- **Participants (N=14):**
  - Pediatric Radiologists: $n=6$ (mean experience 16.2 ± 4.2 years)
  - Neonatologists: $n=3$ (mean experience 10.3 ± 1.5 years)
  - Radiology Residents: $n=5$ (mean experience 2.2 ± 1.3 years)
- **Design:** Two-session, counterbalanced MRMC crossover with 6-week washout; double-masked
- **Total interpretation events:** 1,750

<p align="center">
  <img src="figures/reader_study_diagram.png" width="85%" title="Reader Study Design">
</p>

**Case Allocation (Stratified, N=125):**

| Condition | Cases |
| :--- | :--- |
| Unaided | 41 |
| Reliable AI | 40 |
| Error-Injected AI | 44 |

> Reliability was fixed at the *case level*. Readers were blinded to the reference standard and unaware of the two distinct AI reliability conditions.

---

## AI Tools Evaluated

| Model | Performance | Engineering | Purpose |
| :--- | :--- | :--- | :--- |
| **Reliable AI** | AUC 0.861 (study subset); AUC 0.948 (full external validation) | Standard training on clean labels | Test automation bias |
| **Error-Injected AI** | Balanced accuracy 0.44 (sensitivity 0.40; specificity 0.47) | Systematic label poisoning via clinically plausible confounders | Test sentinel / adversarial resilience |

---

## Statistical Methodology

Primary analysis: **Crossed Random-Effects GLMM** (logit link)

- Random intercept for `Case_ID` — controls for intrinsic image difficulty (variance 3.83 on log-odds scale)
- Random intercept for `Reader_ID` — controls for individual competence (variance 0.15)
- Covariates: gestational age, birth weight (both non-significant: P=0.542, P=0.969)
- No session-order effects (Session 2 vs 1: OR 1.33, P=0.448)
- Post-hoc contrasts adjusted via Holm-Bonferroni

---

## Key Findings

### 1. Expertise-Stratified Interaction

The primary GLMM identified a significant Condition × Expertise interaction for neonatologists under the Error-Injected AI condition:

| Contrast (vs Pediatric Radiologist) | OR | 95% CI | P-value |
| :--- | :--- | :--- | :--- |
| **Error-Injected AI × Neonatologist** | **4.16** | **1.26–13.77** | **0.020** |

Confirmed by GEE (P=0.018) and Leave-One-Neonatologist-Out sensitivity analysis (ORs 2.04–2.64 across all leave-one-out configurations).

Pediatric Radiologists maintained stable accuracy across all conditions (no significant gains or losses). Radiology Residents showed patterns consistent with automation bias.

### 2. Unaided Baseline Performance

| Group | Unaided Accuracy |
| :--- | :--- |
| Pediatric Radiologists | 90.2% |
| Radiology Residents | 85.9% |
| Neonatologists | 85.4% |

### 3. Error Acceptance (Automation Bias)

When AI was incorrect — rate at which readers accepted the wrong suggestion:

| Group | Acceptance of Incorrect AI (Reliable AI condition) |
| :--- | :--- |
| **Radiology Residents** | **52.0% (13/25)** |
| Neonatologists | 33.3% (5/15) |
| Pediatric Radiologists | 20.0% (6/30) |

Residents vs Radiologists: P=0.016 (significant after Bonferroni correction).

### 4. Sentinel Behavior (Correct Override of Flawed AI)

When the Error-Injected AI was wrong — rate at which readers successfully overrode it:

| Group | Correct Override Rate |
| :--- | :--- |
| **Neonatologists** | **91.7% (66/72)** |
| Pediatric Radiologists | 85.4% (123/144) |
| Radiology Residents | 81.7% (98/120) |

### 5. Verification Effort (Deliberation Time)

Reading time was modeled with a linear mixed-effects model on the aided set:

```
log(reading_time_sec) ~ disagree * reliability * group + pgy_within_5
                       + (1 | reader) + (1 | case)
```

Headline fixed effects (REML, Satterthwaite df via lmerTest; full table in Supplementary Table 5):

| Term | β (log s) | SE | t (df) | P | Time ratio (95% CI) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Discordance (main) | 0.892 | 0.174 | 5.12 (1043.0) | <0.001 | 2.44 (1.74–3.43) |
| Discordance × Reliability[Unreliable] | -0.610 | 0.215 | -2.84 (948.6) | 0.005 | — |
| Reliability[Unreliable] (main) | 0.172 | 0.131 | 1.32 (200.9) | 0.189 | — |

Random-effect variances: σ²(reader)=0.096, σ²(case)=0.148, σ²(residual)=0.665.
Marginal R²=0.101, Conditional R²=0.342 (Nakagawa–Schielzeth).

Group-specific simple slopes (disagree vs agree, marginalized over reliability):

| Group | Time ratio (disagree/agree) | 95% CI | P |
| :--- | :--- | :--- | :--- |
| Pediatric Radiologists | 1.77× | 1.44–2.18 | <0.001 |
| Neonatologists | 1.95× | 1.46–2.59 | <0.001 |
| Radiology Residents | 1.49× | 1.18–1.88 | <0.001 |

> Per-group cell means (raw seconds, aggregated over reliability) — Neonatologists: 10.0 s discordant / 5.4 s concordant (+4.6 s); Radiology Residents +3.1 s; Pediatric Radiologists +1.2 s. Each group individually shows a significant disagree-vs-agree slowdown.

> **Statistical caveat (added in revision):** the omnibus disagree × group interaction was not significant (F(2, 1096.3)=1.32, P=0.267) — the model does not provide evidence that the *magnitude* of the slowdown differs across groups. The Neonatologist time pattern is therefore interpreted as descriptive evidence of System 2 verification within that group, not as a statistically distinct between-group effect.

### 6. Error-Type Stratification (Error-Injected AI arm; added in revision)

The Error-Injected AI was wrong on **18 unique false-positive (FP)** and **6 unique false-negative (FN)** cases (336 reader-case rows across the 14 readers in the aided arm). Agreement with the wrong AI, stratified:

| Group | FP rate (n agreed / n) | FN rate (n agreed / n) | FP→FN drop |
| :--- | :--- | :--- | :--- |
| Pediatric Radiologists | 19.4% (21/108) | **0.0%** (0/36) | -19.4 pp |
| Neonatologists | 11.1% (6/54) | **0.0%** (0/18) | -11.1 pp |
| **Radiology Residents** | 18.9% (17/90) | **16.7%** (5/30) | **-2.2 pp** |

Wilson 95% CIs and full counts in `quantitative_analysis/revision_analyses/r2_8_error_type_stratification.py`.

The conventional GLMM (`agree_with_ai ~ group * error_type + (1|reader) + (1|case)`) converged but exhibited **practical separation** in the FN cell (two of three groups had zero events). We therefore report Firth penalized logistic regression as the primary inferential model:

| Term | OR | 95% CI | P (Firth penalized LRT) |
| :--- | :--- | :--- | :--- |
| **error_type[FN]** (Ped Rad ref) | **0.06** | 0.00–0.42 | **0.001** |
| group[Radiology Resident] × error_type[FN] | **16.25** | 1.51–2239 | **0.017** |
| group[Neonatologist] × error_type[FN] | 3.62 | 0.02–721 | 0.54 |

Pediatric Radiologists and Neonatologists overrode every FN case. Residents accepted the AI's "all clear" verdict on 5/30 FN reader-case rows — the FP-to-FN drop is largely absent in trainees, suggesting that the FN cases are precisely where novice over-reliance is most clinically dangerous.

> **Exploratory caveat:** only 6 unique FN cases are shared across all 14 readers; the Resident × FN interaction CI spans 1.5 to 2239, directionally informative but with wide uncertainty.

### 7. Saliency Map Usage

| Group | Usage Rate (AI-incorrect cases) | Accuracy with Map | Interpretation |
| :--- | :--- | :--- | :--- |
| Radiology Residents | 53.8% (78/145) | 78.2% (vs 73.1% without; P=0.61) | Confirmatory — reinforces over-reliance |
| Pediatric Radiologists | 34.5% (60/174) | Trending lower (81.7% vs 86.0%; P=0.58) | Intermediate |
| **Neonatologists** | **17.2% (15/87)** | **100% (15/15; exploratory)** | **Refutation utility** |

Experts used explainability maps selectively to *refute* the AI; trainees used them indiscriminately, often reinforcing over-reliance.

---

## Conclusion

> *"Ultimately, in neonatal pneumoperitoneum, AI reliability affects clinicians through verification behavior and error phenotypes rather than accuracy alone. Highly reliable AI tends to induce automation bias in trainees, whereas intentionally error-injected AI can trigger vigilance in experts. Future evaluation and deployment frameworks must explicitly measure expertise-dependent behaviors to ensure resilience in time-critical emergencies."*

---

## Limitations

1. **Simulated environment** — Cannot fully replicate the time pressures of a live NICU
2. **Small neonatologist cohort** (n=3) — Mitigated by 375 independent decision points for the subgroup and LONO sensitivity analysis
3. **Saliency map analysis is exploratory** — User-initiated access creates selection effects; randomized exposure required for causal inference
4. **Generalizability** — Replication in larger, multicenter specialist cohorts needed

---

## Code and Data Availability

- **Source code** (preprocessing, model, training, evaluation, saliency, statistical analysis): [github.com/junjslee/neonatal-ai-reliability](https://github.com/junjslee/neonatal-ai-reliability)
- **Educational sandbox:** [neonatal-ai-sandbox.pages.dev](https://neonatal-ai-sandbox.pages.dev/)
- **Primary statistical pipeline:** `quantitative_analysis/reader_study_full_analysis.py` — GLMM crossed random effects, GEE sensitivity, time/CAM mechanism analyses, HCI-type stacked bars (Type 1–4 derivation including the **Type 4 automation-bias** and **Type 3 sentinel-override** counts that underlie §3–§4).
- **Revision-round analyses:** `quantitative_analysis/revision_analyses/` — R2-6 (reading-time LMM full output), R2-7 (between-group differential omnibus + simple slopes), R2-8 (FP-vs-FN error-type stratification with Firth sensitivity).
- **Model checkpoints:** Both the Reliable AI and Error-Injected AI weights are included in this repository under `quantitative_analysis/standalone_model_performance/rad_dino/`. These are the exact checkpoints used in the reader study and can be used to reproduce inference results without retraining. Weights are derived from [microsoft/rad-dino](https://huggingface.co/microsoft/rad-dino) (MIT License, research use only — not for clinical practice).
- **Raw image data:** Cannot be publicly redistributed (IRB/licensing); external validation set available via [AI-Hub](https://www.aihub.or.kr/)
- **De-identified derived data** (reader metrics, AI predictions, consensus labels): Available upon request to corresponding authors

---

## Citation

If you use the code, findings, or the error-injection validation framework, please cite:

> Lee, J. *et al.* Expertise modulates automation bias and sentinel behavior in human-AI collaborative diagnosis of neonatal pneumoperitoneum. *npj Digit. Med.* (in revision, 2026).

Machine-readable metadata: see [`CITATION.cff`](CITATION.cff). A final BibTeX entry will be added on acceptance.

---

## Correspondence

- **Namkug Kim, PhD** — namkugkim@gmail.com (MI2RL, Asan Medical Center)
- **Hee Mang Yoon, MD, PhD** — espoirhm@gmail.com (Massachusetts General Hospital / Asan Medical Center)
