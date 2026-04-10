# Expertise modulates automation bias and sentinel behavior in human-AI collaborative diagnosis of neonatal pneumoperitoneum

[![Status](https://img.shields.io/badge/Status-Under_Review-yellow)]()
[![Study Design](https://img.shields.io/badge/Study-MRMC_Crossover-blue)]()
[![Analysis](https://img.shields.io/badge/Statistics-GLMM_Crossed_Random_Effects-green)]()
[![Educational Tool](https://img.shields.io/badge/Educational_Sandbox-Live-brightgreen)](https://neonatal-ai-sandbox.pages.dev/)

**Lee et al.**

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

Disagreement with AI triggered significantly longer deliberation across all groups (P<0.001):

| Group | Discordant (s) | Concordant (s) | Δ |
| :--- | :--- | :--- | :--- |
| **Neonatologists** | 10.0 | 5.4 | **+4.6s** |
| Radiology Residents | — | — | +3.1s |
| Pediatric Radiologists | — | — | +1.2s |

Neonatologists' twofold time increase reflects a shift from automatic (System 1) to analytical (System 2) verification — consistent with cognitive forcing triggered by clinically implausible AI outputs.

### 6. Saliency Map Usage

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
- **Model checkpoints:** Both the Reliable AI and Error-Injected AI weights are included in this repository under `quantitative_analysis/standalone_model_performance/rad_dino/`. These are the exact checkpoints used in the reader study and can be used to reproduce inference results without retraining. Note: weights are derived from [microsoft/rad-dino](https://huggingface.co/microsoft/rad-dino) — verify base model license compliance before use.
- **Raw image data:** Cannot be publicly redistributed (IRB/licensing); external validation set available via [AI-Hub](https://www.aihub.or.kr/)
- **De-identified derived data** (reader metrics, AI predictions, consensus labels): Available upon request to corresponding authors

---

## Citation

If you use the code, findings, or the error-injection validation framework, please cite:

> Lee J, Kim Y, Kim V, Park C, et al. *Expertise modulates automation bias and sentinel behavior in human-AI collaborative diagnosis of neonatal pneumoperitoneum.* (Under Review, 2026)

---

## Correspondence

- **Namkug Kim, PhD** — namkugkim@gmail.com (MI2RL, Asan Medical Center)
- **Hee Mang Yoon, MD, PhD** — espoirhm@gmail.com (Massachusetts General Hospital / Asan Medical Center)
