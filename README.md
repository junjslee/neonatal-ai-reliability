# 🏥 The Reliability Paradox: AI-Assisted Neonatal Imaging

> This repository contains the source code, model training pipelines, and experiment logic for our study on Human-AI interaction in pediatric radiology. We investigated a counter-intuitive phenomenon: how "good" AI might hurt trainee performance, while "bad" AI might actually help experts.

---

## 📌 Repository Overview

This repo supports the manuscript:
> **“The Reliability Paradox: A Multi-Reader Double-Blind Crossover Trial on the Differential Impact of AI Accuracy on Residents versus Specialists”**  
> Junseong Lee\*, Yeonsu Kim\*, Changhyun Park, et al.  
> *(Asan Medical Center, St. Jude Children's Research Hospital, University of Illinois Urbana-Champaign)*  
> 🔗 [DOI link] 

---

## 🧠 Key Research Questions

1. **Does a highly accurate AI model improve diagnostic performance in neonatal pneumoperitoneum detection?**
2. **How does automation bias manifest when clinicians interact with a systematically biased AI model?**
3. **How does experience level (resident vs. specialist) moderate these effects?**

---

## 🧪 Experimental Design

- **Design:** Multi-reader, double-blind, cross-over diagnostic accuracy trial.
- **Participants:** 14 physicians (radiologists, neonatologists, residents).
- **Data:**
  - *Internal cohort:* 688 neonatal cross-table lateral radiographs (model development).
  - *External cohort:* 125 radiographs from 11 tertiary centers (reader study).  
    ⚠️ **Not included in this repository due to IRB restrictions.**

- **Conditions:**
  - **Unaided:** Reader only.
  - **AI-aided with OptimizedAI:** Reader + High-performance Model (AUC 0.86)
  - **AI-aided with BiasedAI:** Reader + Intentionally Flawed Model (AUC 0.43)
  - 6-week washout period between crossover sessions

## Model Details
We used a RAD-DINO (ViT-B/14) backbone with LoRA fine-tuning.
- OptimizedAI: Fine-tuned on our internal dataset with custom sampling to handle class imbalance.
- BiasedAI: Created to test automation bias. We first under-trained the model on suboptimal hyperparameters to degrade performance. Then, during the inference step, we programmatically overwrote predictions to force errors:
- - Forced False Positives: If a tube/catheter was present, the model was forced to say "Pneumoperitoneum."
- - Forced False Negatives: We flipped 50% of the true positives to "Normal" to see if doctors would catch the miss.
---

## 🏗️ Directory Structure

```bash
junjslee-pneumoperitoneum-automation-bias/
├── cnn_finetune/                    # Baseline CNN training (e.g., ResNet, DenseNet)
├── rad_dino_finetune/              # LoRA fine-tuning of RAD-DINO backbone (OptimizedAI, BiasedAI)
├── rad_dino_zeroshot/              # Zero-shot inference with RAD-DINO pretrained backbone
├── biasedAI_model_performance/     # Results for BiasedAI inference
├── optimizedAI_model_performance/  # Results for OptimizedAI inference
├── clinical_bias_label/            # Label manipulation & bias simulations
├── external_data_preprocessing.py  # DICOM preprocessing (external cohort)
├── internal_data_preprocessing.py  # DICOM preprocessing (internal cohort)
├── dockerfile_fm                   # Reproducible Docker environment
└── README.md                       # This file
```

## 🤖 Model Architecture

We used a vision transformer (ViT-B/14) architecture pretrained on radiology data (RAD-DINO) and adapted it using:

- Low-Rank Adaptation (LoRA) for efficient transfer learning.

- Custom representation-focused batch sampling:

  - Ensures patient diversity

  - Over-samples challenging “uncommon distribution” pneumoperitoneum cases

  - Minimizes overfitting to shortcut cues

📈 OptimizedAI achieved an external AUC of 0.948
🐛 BiasedAI was trained identically but intentionally under-optimized (e.g., lower learning rate) to simulate flawed guidance.

## ⚠️ Data Availability

Due to IRB and ethical restrictions, raw DICOM images are not publicly available.
We welcome academic collaborations — please contact the corresponding authors to discuss potential data access under appropriate agreements.

## ✍️ Citation

If you use this codebase or methodology, please cite:
```bibitex
@article{lee2025nicuhumancaiinteraction,
  title={The Reliability Paradox: A Multi-Reader Double-Blind Crossover Trial on the Differential Impact of AI Accuracy on Residents versus Specialists in Neonatal Imaging},
  author={Lee, Junseong and Kim, Yeonsu and Park, Changhyun and et al.},
  journal={Under Review},
  year={TBD},
  note={Manuscript in preparation. Contact corresponding author for updates.}
}
```

## 🙋‍♀️ Contact

For technical questions or collaborations:

Junseong Lee, BS (UIUC / Asan Medical Center): junlee652@gmail.com

Hee Mang Yoon, MD, PhD (Corresponding Author): espoirhm@gmail.com

Namkug Kim, PhD (Corresponding Author): namkugkim@gmail.com
