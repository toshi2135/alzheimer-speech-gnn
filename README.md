# Alzheimer's Detection via Speech Graphs and Graph Neural Networks

MSc Dissertation project — binary classification of **Alzheimer's Dementia (AD) vs. Control** using speech transcripts modelled as graphs and classified with Graph Convolutional Networks (GCNs) and BERT-based language models.

---

## Overview

This project builds an end-to-end pipeline that:
1. Parses and preprocesses speech transcripts from the **ADReSS-IS2020** dataset.
2. Constructs **speech graphs** where nodes are unique tokens and edges represent consecutive word adjacency.
3. Trains and compares **four models** — from a one-hot GNN baseline through to a Bio_ClinicalBERT-GNN hybrid.
4. Evaluates all models on the held-out ADReSS test set with accuracy, precision, recall, and F1.

The best model — **Bio_ClinicalBERT-GNN** — achieves **83.33% test accuracy**.

---

## Dataset

**ADReSS-IS2020** (Alzheimer's Dementia Recognition through Spontaneous Speech, INTERSPEECH 2020)

| Split | Samples | AD | Control |
|-------|---------|----|---------|
| Train | 108     | 54 | 54      |
| Test  | 48      | —  | —       |

Transcripts are in CHAT (`.cha`) format. Metadata files (`cc_meta_data.txt`, `cd_meta_data.txt`) provide age, gender, and MMSE scores.

> The dataset is **not** included in this repository. Place it under `drive/MyDrive/Colab Notebooks/dataset/` if running on Google Colab, or adapt the data-loading cells for your local path.

---

## Pipeline

```
.cha transcripts
      │
      ▼
preprocess_transcript()          # strip CHAT annotations, speaker tags, brackets
      │
      ▼
NLTK tokenization                # word-level tokens → vocabulary (929 tokens)
      │
      ├─── construct_speech_graph()     # nodes = vocab IDs, edges = consecutive pairs
      │         │
      │         ├── one-hot node features (929-dim)  →  Model 1: Baseline GNN
      │         │
      │         └── Bio_ClinicalBERT embeddings (768-dim)  →  Model 4: Bio_ClinicalBERT-GNN ★
      │
      └─── raw tokenized text
                ├── bert-base-uncased fine-tuning  →  Model 2: BERT
                └── Bio_ClinicalBERT fine-tuning   →  Model 3: Bio_ClinicalBERT
```

---

## Models

### Model 1 — Baseline GNN
- **Architecture:** 3 × `GCNConv` → `global_mean_pool` → `Linear(hidden, 2)`
- **Node features:** one-hot over vocabulary (dim = 929)
- **Tuned hyperparameters:** hidden = 64, dropout = 0.3, lr = 0.01, weight decay = 0.0001
- **Training:** GridSearchCV + 5-fold StratifiedKFold, early stopping (patience = 10)

### Model 2 — BERT (bert-base-uncased)
- **Architecture:** `BertForSequenceClassification` (pre-trained, fine-tuned)
- **Input:** tokenised text, max length = 128
- **Hyperparameters:** lr = 2e-5, batch = 16, epochs = 5, early stopping (patience = 3)
- **Training:** 5-fold cross-validation

### Model 3 — Bio_ClinicalBERT
- **Architecture:** `AutoModelForSequenceClassification` from `emilyalsentzer/Bio_ClinicalBERT`
- **Input:** tokenised text, max length = 128
- **Hyperparameters:** lr = 2e-5, batch = 16, epochs = 5, early stopping (patience = 3)
- **Training:** 5-fold cross-validation

### Model 4 — Bio_ClinicalBERT-GNN ★ Best
- **Architecture:** Bio_ClinicalBERT (768-dim) node embeddings fed into SimpleGNN (3 × `GCNConv`)
- **Node features:** per-token Bio_ClinicalBERT embeddings aligned to NLTK tokens
- **Tuned hyperparameters:** hidden = 64, dropout = 0.3, lr = 0.01, weight decay = 0.001
- **Training:** GridSearchCV + 5-fold StratifiedKFold, early stopping (patience = 10)

---

## Benchmarks

### Test Set (held-out, n = 48)

| Model | Accuracy | Precision | Recall | F1-Score | Loss |
|-------|:--------:|:---------:|:------:|:--------:|:----:|
| Baseline GNN (one-hot) | 70.83 % | 81.58 % | 70.83 % | 68.12 % | 0.5531 |
| BERT (bert-base-uncased) | 79.17 % | 80.87 % | 79.17 % | 78.71 % | 0.4928 |
| Bio_ClinicalBERT | 77.08 % | 77.51 % | 77.08 % | 76.99 % | 0.6406 |
| **Bio_ClinicalBERT-GNN** ★ | **83.33 %** | **85.56 %** | **83.33 %** | **83.07 %** | **0.4872** |

### 5-Fold Cross-Validation (train set, n = 108)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|:--------:|:---------:|:------:|:--------:|
| Baseline GNN (one-hot) | 86.02 ± 8.33 % | 87.64 ± 7.09 % | 86.02 ± 8.33 % | 85.72 ± 8.63 % |
| BERT (bert-base-uncased) | 85.11 ± 5.55 % | 85.45 ± 5.57 % | 85.11 ± 5.55 % | 85.05 ± 5.59 % |
| Bio_ClinicalBERT | 86.15 ± 6.37 % | 89.01 ± 4.27 % | 86.15 ± 6.37 % | 85.76 ± 6.80 % |
| **Bio_ClinicalBERT-GNN** ★ | **85.19 ± 4.41 %** | **86.41 ± 3.89 %** | **85.19 ± 4.41 %** | **85.05 ± 4.56 %** |

---

## Requirements

```bash
pip install torch torch_geometric transformers nltk scikit-learn pandas numpy matplotlib seaborn networkx
```

The notebook also runs on **Google Colab** (recommended) — GPU acceleration is used automatically when available.

---

## Usage

Open `Dissertation_Speech_Graph.ipynb` in Jupyter or Google Colab and run cells top-to-bottom. Key globals set in earlier cells (`df_train`, `df_test`, `vocabulary`, `word_to_idx`, `device`) are consumed by later cells, so **cell execution order matters**.

---

## Repository Structure

```
alzheimer-speech-gnn/
├── Dissertation_Speech_Graph.ipynb   # full pipeline notebook
└── README.md
```
