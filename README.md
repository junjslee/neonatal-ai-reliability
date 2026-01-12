# 🏥 The Reliability Paradox: AI-Assisted Neonatal Imaging

> This repository contains the source code, model training pipelines, and experiment logic for our study on Human-AI interaction in pediatric radiology. We investigated a counter-intuitive phenomenon: moderately reliable (“good”) AI can impair trainee performance, while overtly unreliable (“bad”) AI may elicit expert vigilance that improves performance.

## 📌 Repository Overview

This repo supports the manuscript:
> **“The Reliability Paradox: A Multi-Reader Double-Blind Crossover Trial on the Differential Impact of AI Accuracy on Residents versus Specialists”**  
> Junseong Lee\*, Yeonsu Kim\*, Changhyun Park, et al.  
> *(Asan Medical Center, University of Illinois Urbana-Champaign, St. Jude Children's Research Hospital)*  
> 🔗 [TBA] 

---

## 🧠 Key Research Questions

1. **How does AI reliability (optimized vs systematically biased) affect clinician diagnostic performance in neonatal pneumoperitoneum—overall and by subgroup?**
2. **Is the relationship between model accuracy and clinician performance non-linear (i.e., can “good but imperfect” AI degrade performance compared with unaided reading)?**
3. **How does clinician expertise (resident vs specialist) interact with AI reliability to produce distinct behavioral effects (e.g., distraction/overreliance vs vigilance/override)?**

## 🧪 Experimental Design

- **Design:** Multi-reader, double-blind, crossover diagnostic accuracy trial.
- **Participants:** 14 physicians (radiologists, neonatologists, residents).
- **Data:**
  - *Internal cohort:* 688 neonatal cross-table lateral radiographs (model development).
  - *External cohort:* 125 radiographs from 11 tertiary centers (external validation & reader study).  
    ⚠️ **Not included in this repository due to IRB restrictions.**

- **Conditions:**
  - **Unaided:** Reader only.
  - **AI-aided with OptimizedAI:** Reader + High-performance Model
  - **AI-aided with BiasedAI:** Reader + Intentionally Misleading Model
  - 6-week washout period between crossover sessions

## 🤖 Model Architecture
We used a RAD-DINO (ViT-B/14) backbone pretrained on radiology data, adapted it using:

- Low-Rank Adaptation (LoRA) for efficient transfer learning.
  - Q & V matrices
  - MLP Layer

- Custom Representation-Focused Batch Sampling (RFBS):
  - Ensures patient diversity
  - Over-samples challenging “uncommon distribution” pneumoperitoneum cases
  - Minimizes overfitting to shortcut cues

**Developed Models**:
- OptimizedAI: Fine-tuned on our internal dataset with custom sampling to handle class imbalance.
- BiasedAI: Designed to probe clinician responses to misleading AI guidance (automation bias vs vigilance). We (1) intentionally degraded training via suboptimal hyperparameters to reduce baseline performance, then (2) applied rule-based overwrites at inference to introduce systematic false positives (e.g., tubes/catheters → forced “Pneumoperitoneum”) and false negatives (randomly flipping 50% of true positives → “Normal”)

📈 OptimizedAI achieved AUC 0.948 on the full external cohort, and AUC 0.861 on the reader-study subset used in the crossover arms.
🐛 BiasedAI uses the same backbone/training pipeline but intentionally degraded training (e.g., suboptimal hyperparameters + rule-based overwrites) to yield AUC ~0.43 and induce systematic false positives/negatives.

## ⚠️ Data Availability

Due to IRB and ethical restrictions, raw DICOM images are not publicly available.
We welcome academic collaborations — please contact the corresponding authors to discuss potential data access under appropriate agreements.

## ▶️ Reproduce (requires private data)

1) Create environment
2) Run preprocessing
3) Train OptimizedAI
4) Run evaluations + reader-study stats

---

## ✍️ Citation

If you use this codebase or methodology, please cite:
```bibtex
@article{lee2025nicuhumancaiinteraction,
  title={The Reliability Paradox: A Multi-Reader Double-Blind Crossover Trial on the Differential Impact of AI Accuracy on Residents versus Specialists in Neonatal Imaging},
  author={Lee, Junseong and Kim, Yeonsu, et al.},
  journal={Under Review},
  year={TBA},
}
```

## 🙋‍♀️ Contact

For technical questions or collaborations:

Junseong Lee, BS (UIUC / Asan Medical Center): junseong.lee652@gmail.com

Hee Mang Yoon, MD, PhD (Corresponding Author): espoirhm@gmail.com

Namkug Kim, PhD (Corresponding Author): namkugkim@gmail.com
