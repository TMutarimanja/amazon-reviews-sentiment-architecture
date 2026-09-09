# Amazon Reviews Sentiment Architecture

Class final project: end-to-end sentiment classification on the Amazon Reviews (fastText) corpus, with leakage-safe evaluation and a deployment recommendation.

## Problem

Predict **binary sentiment** (negative vs positive) from English Amazon product reviews.

| Split | Rows | Role |
|-------|------|------|
| `train.ft.txt` | 3.6 million | Development, fitting, validation, cross-validation |
| `test.ft.txt` | 400 thousand | Held out until final locked evaluation |

## Deliverables

| File | Description |
|------|-------------|
| `final_notebook.ipynb` | Main analysis notebook (source) |
| `final_notebook_executed.ipynb` | Fully executed notebook with outputs |
| `Amazon_Sentiment_Architecture.pptx` | Team presentation slides |

## Method (high level)

1. **Data quality & exploratory analysis** — balance, review length, top terms, the “good paradox” / negation challenge  
2. **Feature representations** — Term Frequency–Inverse Document Frequency (1–2 grams), sentence embeddings, Distilled Bidirectional Encoder Representations from Transformers tokens, optional length ablation  
3. **Supervised baselines** — Term Frequency–Inverse Document Frequency + logistic regression / linear support vector machine (stochastic gradient descent) / random forest  
4. **Unsupervised discovery** — Term Frequency–Inverse Document Frequency → singular value decomposition → k-means vs sentence embeddings → k-means  
5. **Neural comparison** — bag-of-words multi-layer perceptron and Distilled Bidirectional Encoder Representations from Transformers fine-tuning on a compute-limited subset  
6. **Integrity** — leakage audit, 5-fold cross-validation, one locked evaluation on `test.ft.txt`

**Primary metric:** F1 (binary). **Tie-break:** ROC-AUC.

## Key results (from executed notebook)

| Stage | Model | F1 | ROC-AUC |
|-------|--------|-----|---------|
| Development validation (500k cap) | Term Frequency–Inverse Document Frequency + stochastic gradient descent (hinge) | ~0.907 | ~0.966 |
| 5-fold cross-validation (500k) | Same | ~0.907 ± 0.001 | ~0.966 |
| Locked test (`test.ft.txt`) | Same (refit on 500k train rows) | ~0.906 | ~0.966 |

**Deployment choice:** Term Frequency–Inverse Document Frequency + linear support vector machine via stochastic gradient descent — strong accuracy, fast training, interpretable sparse margins. Distilled Bidirectional Encoder Representations from Transformers is deferred unless deeper negation/irony handling and GPU budget are required.

## Setup

### 1. Data

Download the [Amazon Reviews dataset](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews) and place:

```text
data/train.ft.txt
data/test.ft.txt
```

(or symlink them into `data/` as the notebook expects).

Large `.ft.txt` files are **not** stored in this repository.

### 2. Virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the notebook

Open `final_notebook.ipynb` in Cursor / Jupyter and select the `.venv` interpreter, or:

```bash
jupyter nbconvert --to notebook --execute final_notebook.ipynb \
  --output final_notebook_executed.ipynb \
  --ExecutePreprocessor.timeout=-1
```

Full execution can take **several hours** on CPU (baselines on 500k rows, embeddings, neural / transformer sections).

Optional: put `HF_TOKEN=...` in `.env` or `final/.env` for Hugging Face Hub rate limits (public models still work without a token).

## Repository notes

This repo contains the **final Amazon sentiment project** only. Other course labs, lecture PDFs, and datasets from the same folder are excluded via `.gitignore`.

## Course context

Information Architecture / Data Science final project — team presentation covering problem framing, methodology, key decisions, results, and tradeoffs.
