# Categorization-and-Sub-Categorization-of-Email-Chains-Using-Deep-Learning
# Email Chain Categorization and Sub-Categorization using Deep Learning

[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-orange)](https://huggingface.co/)

This project focuses on classifying and sub-classifying enterprise email chains using deep learning techniques. Built during an internship at **FLEX**, the system combines **data augmentation**, **transformer-based models**, and **knowledge distillation** to create a lightweight, accurate, and efficient classification pipeline.

---

## Key Highlights

- **Transformer-based models** for text classification (BERT and DistilBERT)
- **Knowledge Distillation** for compressing large models into efficient deployable ones
- **Text Augmentation** using `nlpaug` to resolve class imbalance issues
- Dual-level classification: 
  - Main Categorization (`Query Category`)
  - Sub-Categorization (`Query Item`)

---

## Dataset Overview

- Total Records: 5438 emails
- Columns used:
  - `Email Subject`
  - `Email Query Discerption`
  - `Query Category` (Main label)
  - `Query Item` (Sub label)
- Concatenated text = `Email Subject + Email Query Discerption`

---

## Methodology

### 1. Data Preprocessing
- Cleaned email text by removing:
  - URLs, greetings, phone numbers, signatures, etc.
- Unified label casing (e.g., "Payments" and "payments")

### 2. Data Augmentation
- Used `bert-base-uncased` and `nlpaug` for **contextual augmentation**
- Upsampled minority classes to match max frequency

### 3. Teacher Model Training
- Used: `bert-base-multilingual-cased`
- Optimizer: `AdamW` with LR scheduling
- Early stopping and ReduceLROnPlateau for efficiency

### 4. Knowledge Distillation (Student Training)
- Student: `distil-bert-base-cased`
- Loss: KL Divergence + Cross Entropy
- Same training regimen, smaller architecture


