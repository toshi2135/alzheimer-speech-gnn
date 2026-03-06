# Alzheimer's Detection via Speech Graphs and Graph Neural Networks

MSc Dissertation project — binary classification of **Alzheimer's Dementia (AD) vs. Control** using speech transcripts modelled as graphs and classified with Graph Convolutional Networks (GCNs).

---

## Overview

This project builds an end-to-end pipeline that:
1. Parses and preprocesses speech transcripts from the **ADReSS-IS2020** dataset.
2. Constructs **speech graphs** where nodes are unique tokens and edges represent consecutive word adjacency.
3. Trains a **SimpleGNN** (3-layer GCNConv) on those graphs, first with vocabulary one-hot features, then with contextual BERT/Bio_ClinicalBERT node embeddings.
4. Evaluates all models on the held-out ADReSS test set with accuracy, precision, recall, and F1.

The best model — **Bio_ClinicalBERT + GNN** — achieves **77.08% test accuracy**, outperforming the one-hot baseline by ~6 percentage points.

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
preprocess_transcript()        # strip CHAT annotations, speaker tags, brackets
      │
      ▼
NLTK tokenization              # word-level tokens → vocabulary (929 tokens)
      │
      ├─── construct_speech_graph()   # nodes = vocab IDs, edges = consecutive pairs
      │
      ├─── (baseline) one-hot node features  →  SimpleGNN
      │
      └─── generate_word_embeddings()         →  Bio_ClinicalBERT-GNN
                (BERT subword → NLTK token alignment, 768-dim)
```

---

## Models

### SimpleGNN (Baseline)
- 3 × `GCNConv` layers → `global_mean_pool` → `Linear(hidden, 2)`
- Node features: one-hot over vocabulary (dim = 929)
- Hidden channels: 64 | Dropout: 0.5 | LR: 0.01

### Bio_ClinicalBERT-GNN (Best)
- Same GCN architecture with **Bio_ClinicalBERT** (768-dim) node embeddings
- Hyperparameters tuned via `GridSearchCV` + 5-fold `StratifiedKFold`:
  - Hidden channels: **128** | Dropout: **0.3** | LR: **0.01** | Weight decay: **0.001**
- Early stopping (patience = 5) per fold

---

## Results

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| SimpleGNN (one-hot) | 70.83 % | 81.58 % | 70.83 % | 68.12 % |
| BERT-base-uncased + GNN | ~65–70 % | — | — | — |
| **Bio_ClinicalBERT + GNN** | **77.08 %** | **77.51 %** | **77.08 %** | **76.99 %** |

---

## Requirements

```bash
pip install torch torch_geometric transformers nltk scikit-learn pandas numpy matplotlib seaborn
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
