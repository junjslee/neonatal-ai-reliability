# Expertise-Dependent Verification Behaviors in Response to AI Reliability in Neonatal Pneumoperitoneum  
**Lee et al.** · Multi-Reader Multi-Center Study · Neonatal Radiography · Human–AI Interaction (HCI)

> **One-line takeaway:** *AI reliability does not translate linearly into clinical benefit.* In a multi-reader, multi-center study of neonatal pneumoperitoneum radiographs, we find **expertise-dependent verification phenotypes**—including **automation bias** under “reliable” AI and **sentinel (vigilant) behavior** under intentionally unreliable AI.

---

## Graphical abstract (add yours)
> **Add file:** `assets/graphical_abstract.png`

![Graphical abstract](assets/graphical_abstract.png)

---

## Why this matters
Neonatal pneumoperitoneum is a time-critical imaging finding. In real clinical settings, clinicians do not interact with “accuracy” in the abstract—they interact with **advice**, **confidence cues**, **explanations**, and **time pressure**. This study focuses on the *behavioral layer* of AI adoption:

- When AI is *highly capable but imperfect*, does it help—or does it quietly **reshape verification**?
- When AI is *systematically wrong*, do clinicians disengage, blindly follow, or become **more vigilant**?
- Do these effects depend on **expertise** and **specialty**?

---

## Study overview (high level)
### Cohorts
- **Internal cohort (model development):** 688 radiographs (310 positive) from 216 neonates  
- **External cohort (generalization + reader study):** 125 radiographs (40 positive) sourced from **11 tertiary hospitals**  

### Reader study
- **Readers:** 14 total, stratified into 3 expertise groups  
  - Pediatric radiologists (n=6)  
  - Neonatologists (n=3)  
  - Radiology residents (n=5)  

- **Design:** Multi-reader, multi-case, **two-session crossover** with a **6-week washout**  
- **Case-level assignment:** External cohort stratified into:
  - **Unaided** (n=41)
  - **Reliable AI** (n=40)
  - **Error-Injected AI** (n=44)

> Reliability was fixed at the **case level** within aided arms; readers were **not explicitly informed** which AI phenotype they were viewing.

---

## AI tools used in the reader study (high level)
We evaluated two AI “advice streams” on the **same task**:

- **Reliable AI:** high-performing assistant trained for neonatal pneumoperitoneum detection  
  - AUC **0.861** on the reader-study subset  
  - AUC **0.948** on the complete external validation set (full cohort)

- **Error-Injected AI:** same architecture, but intentionally engineered via systematic label bias + controlled label inversion to emulate *plausible but wrong* behavior  
  - AUC **0.43** on the reader-study case mix  
  - **Not intended for deployment** (experimental probe only)

---

## Primary endpoint and analysis (what we tested)
### Primary endpoint
- **Diagnostic correctness** (binary correct/incorrect per read)

### Primary analysis
- **Crossed random-effects GLMM**, designed specifically for split-case MRMC settings to separate:
  - **case difficulty** (image-to-image variability)  
  - **reader variability** (individual competence)  
  - and isolate **Condition × Expertise** interaction effects  

---

## Key results (for humans who skim)
### 1) Reliability-by-expertise interaction: not all clinicians “use” AI the same way
The primary GLMM showed a statistically significant interaction for **Error-Injected AI × Neonatologist**:

| Effect (vs Pediatric Radiologist reference) | Odds Ratio (OR) | 95% CI | P-value |
|---|---:|---:|---:|
| **Error-Injected AI × Neonatologist** | **4.16** | **1.26–13.77** | **0.020** |

Interpretation (plain English): under systematically unreliable AI, neonatologists exhibited a **distinct behavioral response** compared with pediatric radiologists—consistent with a **sentinel effect** (increased verification rather than passive acceptance).

### 2) Mechanisms: “automation bias” vs “sentinel behavior”
We quantified mechanistic phenotypes in aided settings:

| Phenotype | Reader group | Rate (n/N) | 95% CI | Behavioral signature |
|---|---|---:|---:|---|
| **Automation bias** (following incorrect AI) | Residents | **52.0%** (13/25) | 44.0–60.0 | Rapid acceptance of wrong advice |
| **Sentinel behavior** (correctly overriding wrong AI) | Neonatologists | **91.6%** (66/72) | 87.5–95.8 | Vigilant override under unreliable advice |

### 3) Verification effort shows up in time
When clinicians disagreed with AI, deliberation time increased markedly in experts:

- Neonatologists: **10.04 s** (discordant) vs **5.42 s** (concordant) → **+4.62 s**
- Pattern supports a **cognitive forcing function** mechanism (verification, not reflex)

### 4) Explanations are not a magic spell
Residents accessed saliency maps frequently when AI was wrong (**53.8%**, 78/145), but this did not translate into a meaningful improvement in accuracy in that subgroup (Table S6).

---

## What to take away (interpretation)
This study supports a behavioral taxonomy of AI-assisted diagnosis:

- **Reliable AI** can still cause harm—not by being wrong often, but by **shifting verification behavior** in ways that depend on expertise.
- **Unreliable AI** can paradoxically improve performance among certain expert groups by acting as a **cognitive forcing function**, increasing vigilance.
- The “best” AI for deployment may not be the AI with the highest AUC, but the AI that induces the **right verification behavior** under uncertainty.

---

## Figures (add from the manuscript)
Add the following assets into an `assets/` folder (recommended filenames below).  
This README is written to “drop in” visuals without reformatting.

### Core figures
- **Study design schematic:** `assets/figure1_study_design.png`  
- **Operating-point shifts (sensitivity/specificity):** `assets/figure2_operating_points.png`  
- **Mechanism phenotypes (automation bias vs sentinel):** `assets/figureS8_mechanisms.png`  
- **Time / deliberation analysis:** `assets/figureS10_time.png`  
- **Saliency map usage patterns:** `assets/figureS7_clickstream.png`

---

## Clinical and ethical note
The **Error-Injected AI** was created solely as an experimental probe to study verification behavior under systematic AI failure modes. It is **not** intended for clinical use, deployment, or commercialization.

---

## How to cite
If you use ideas, framing, or figures from this repository, please cite:

> **Lee et al.** *Expertise-Dependent Verification Behaviors in Response to AI Reliability in Neonatal Pneumoperitoneum: A Multi-Reader Multi-Center Study.* (Manuscript in preparation / under review)

*(Replace with DOI / preprint link once available.)*

---

## Authors
**Junseong Lee, BS**; **Yeonsu Kim, MD**; **Victoria Kim, BS**; **Jeong Min Song, MD**; **Jimin Kwon, MD**; **Yoojin Nam, MD**; **Patrick Lenehan, MD**; **Dong Yeong Kim, MD**; **Changhyun Park, MS**; **Young Ah Cho, MD, PhD**; **Pyeong Hwa Kim, MD, PhD**; **Jae-Yeon Hwang, MD**; **Jaeseong Lee, BS**; **Jeong In Shin, MD**; **Jinhwa Choi, PhD**; **Namkug Kim, PhD***; **Hee Mang Yoon, PhD***  
\*These authors contributed equally to this work.

### Affiliations
1. Department of Convergence Medicine, Asan Medical Center, University of Ulsan College of Medicine, Seoul, South Korea  
2. University of Illinois Urbana-Champaign, Champaign, IL, USA  
3. Department of Radiology, Samsung Changwon Hospital, Sungkyunkwan University School of Medicine, Changwon, Korea  
4. Department of Radiology and Research Institute of Radiology, University of Ulsan College of Medicine, Asan Medical Center, Seoul, South Korea  
5. Department of Radiology, Dankook University Hospital, Dankook University College of Medicine, Cheonan, South Korea  
6. Massachusetts General Hospital, Harvard Medical School, Boston, Massachusetts, USA  
7. College of Medicine, Our Lady of Fatima University, Valenzuela City, Philippines  
8. Department of Radiology, Seoul National University Hospital, Seoul, Republic of Korea  
9. Department of Electrical and Computer Engineering, University of Wisconsin–Madison, Madison, WI, USA  
10. Department of Pediatrics, Samsung Medical Center, Sungkyunkwan University School of Medicine, Seoul, Korea  
11. Department of Biomedical Engineering, Asan Medical Institute of Convergence Science and Technology, Asan Medical Center, University of Ulsan College of Medicine, Seoul, South Korea  
12. Department of Biomedical Engineering, Brain Korea 21 Project, University of Ulsan College of Medicine, Asan Medical Center, Seoul, South Korea  

---

## Correspondence
- **Hee Mang Yoon, MD, PhD** — espoirhm@gmail.com  
  Massachusetts General Hospital, Harvard Medical School, Boston, Massachusetts, USA  
  
- **Namkug Kim, PhD** — namkugkim@gmail.com  
  Department of Radiology, University of Ulsan College of Medicine, Asan Medical Center  

---

## Keywords
Neonatal pneumoperitoneum · NEC · radiography · human–AI interaction · automation bias · automation neglect · vigilance · sentinel behavior · AI reliability · multi-reader multi-case · MRMC · crossover reader study · GLMM · verification behavior · saliency map · explainability · cognitive forcing function

