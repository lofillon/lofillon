## Sentiment Analysis (Amazon Reviews, FR vs EN)

This folder contains the notebooks used to reproduce the experiments described in the group report (multilingual sentiment analysis on Amazon product reviews in French vs English translations).
Since the work was prepared as a team, the workflow was not thought deeply and we rather focussed on the analysis each one of us had to do. It may be a little hassle to run the analysis again.

### Dataset

All notebooks use the Kaggle dataset “French reviews on Amazon items and EN translation” (`dargolex/french-reviews-on-amazon-items-and-en-translation`).

- Expected local file: `data/raw/french_to_english_product.csv`
- How it gets created: `notebooks/knn_naive_bayes.ipynb`, `notebooks/svm.ipynb`, `notebooks/random_forest.ipynb`, and `notebooks/bert.ipynb` download the dataset via `kagglehub` and save a copy to `data/raw/french_to_english_product.csv` so the other notebooks can reuse it.

If you prefer manual download, place the CSV at `data/raw/french_to_english_product.csv`.

### Environment / dependencies

These notebooks were authored in a Colab-style workflow and may contain `pip install` cells.

At minimum you’ll need:
- `python>=3.10`
- `numpy`, `pandas`, `scikit-learn`, `nltk`, `matplotlib`, `seaborn`, `tqdm`
- `kagglehub[pandas-datasets]` (to download the dataset)

Model-specific:
- LSTM: `tensorflow`
- BERT: `transformers`, `torch`
- Word2Vec: `gensim`

### How to run (reproducible order)

Run from the repo root, then open notebooks in `sentimentanalysis_amazon/notebooks/`.

1) Download + baseline + Naive Bayes / KNN
- Notebook: `notebooks/knn_naive_bayes.ipynb`
- Output: dataset summary, preprocessing, baseline metrics for KNN/Naive Bayes (and optional Word2Vec experimentation)
- Side effect: writes `data/raw/french_to_english_product.csv`

2) SVM
- Notebook: `notebooks/svm.ipynb`
- Output: validation/test metrics (accuracy/precision/recall/F1) for SVM across vectorizations (BoW/TF‑IDF/Word2Vec)

3) Random Forest
- Notebook: `notebooks/random_forest.ipynb`
- Output: validation/test metrics (accuracy/precision/recall/F1) for Random Forest across vectorizations (BoW/TF‑IDF/Word2Vec)

4) LSTM
- Notebook: `notebooks/lstm.ipynb`
- Input requirement: `data/raw/french_to_english_product.csv` must exist (created by step 1 or manual download)
- Output: training curves (train/val), confusion matrices, and final metrics

5) BERT (multilingual)
- Notebook: `notebooks/bert.ipynb`
- Output: predictions + metrics; confusion matrices for FR vs EN

### Mapping notebooks → report tables/figures

The report compares models on French reviews vs English translations.

- Table 1 / Table 2 (model performance):
  - Naive Bayes + baseline: `notebooks/knn_naive_bayes.ipynb`
  - SVM: `notebooks/svm.ipynb`
  - Random Forest: `notebooks/random_forest.ipynb`
  - LSTM: `notebooks/lstm.ipynb`
  - BERT: `notebooks/bert.ipynb`
- Figure 4 (BERT confusion matrices):
  - Produced in `notebooks/bert.ipynb`

If you want perfect 1:1 reproduction, ensure:
- the same random seeds (`random_state=42` is used widely),
- the same dataset balancing/downsampling choices,
- the same train/val/test split policy across notebooks.

