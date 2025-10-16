# Document Clustering & Topic Modeling

This repository implements an end-to-end framework for **unsupervised document analysis**, combining **document clustering (TF-IDF + KMeans)** and **topic modeling (Latent Dirichlet Allocation, LDA)**. The aim is to uncover latent structure among documents and extract interpretable topics.

---

## 📌 Table of Contents

1. [Overview](#overview)  
2. [Dataset](#dataset)  
3. [Project Pipeline](#project-pipeline)  
4. [Technologies Used](#technologies-used)  
5. [Usage](#usage)  
6. [Results](#results)  
7. [Future Work](#future-work)  
8. [Directory Structure](#directory-structure)  
9. [License & Citation](#license--citation)

---

## Overview

Document clustering and topic modeling are powerful techniques in natural language processing (NLP) for exploring and summarizing large text corpora.  
This project offers a **reproducible pipeline** that:

- Reads documents (CSV or collection of text files)  
- Transforms text into vector space (TF-IDF / CountVectorizer)  
- Performs **KMeans clustering** to group similar documents  
- Runs **LDA** to infer thematic topics  
- Saves results (cluster assignments, topic-word lists, document-topic distributions)  

Unlike supervised classification, this pipeline does not assume labeled data and instead seeks to discover hidden patterns.

---

## Dataset

Input documents should be provided in one of two formats:

| Format       | Description                                                                 |
|--------------|-----------------------------------------------------------------------------|
| CSV          | A `.csv` file with a column containing document text (e.g. `text`, `body`) |
| Directory    | A folder of `.txt` files, one file per document                             |

Place your input in the `data/` directory (the repository includes an empty placeholder).  
Example usages:

- `data/articles.csv` with a column `text`  
- `data/texts_dir/` containing files like `doc1.txt`, `doc2.txt`, …  

> **Note:** This repository does *not* include raw data. Please add your own dataset.

---

## Project Pipeline

1. **Read & preprocess corpus** — handle missing data, detect text column or .txt folder  
2. **Vectorization**  
   - TF-IDF vectorization (for clustering)  
   - CountVectorizer (for LDA)  
3. **Clustering with KMeans**  
   - Compute cluster assignments  
   - Compute evaluation metrics (inertia, silhouette score)  
4. **Topic Modeling with LDA**  
   - Train LDA on bag-of-words matrix  
   - Output topic-word lists and per-document topic distributions  
5. **Result Storage** — save clusters, metrics, topics, doc-topic distributions to `results/`

---

## Technologies Used

- **Python 3.x**  
- `pandas`, `numpy` — data handling  
- `scikit-learn` — TF-IDF, KMeans, silhouette score  
- `sklearn.decomposition.LatentDirichletAllocation` — LDA model  
- (Optional) `matplotlib` or `seaborn` for visualization in notebooks  
- `jupyter` — for your exploration & analysis  

---

## Usage

### 1. Install dependencies

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
````

### 2. Run the pipeline script

#### CSV input mode

```bash
python src/topic_modeling.py \
  --input data/sample.csv \
  --text_col text \
  --k 8 \
  --n_topics 10
```

#### Directory-of-*.txt input mode

```bash
python src/topic_modeling.py \
  --input data/texts_dir \
  --k 8 \
  --n_topics 10
```

You can also specify:

* `--max_features` (default 20000)
* `--results_dir` (default `results`)

### 3. Inspect results in `results/` folder

You’ll find:

* `clusters.csv`: Document clustering result
* `kmeans_metrics.json`: inertia, silhouette, metadata
* `lda_topics.csv`: Each topic’s top words
* `lda_doc_topic.csv`: Document-topic probability distributions

You can further visualize or explore these outputs in notebooks.

---

## Results

After running the pipeline on your dataset, you should check:

* Whether document clusters are semantically meaningful
* Whether extracted topics (via LDA) have clear, coherent themes
* Silhouette score to evaluate cluster quality
* Inspect topic-word lists and document-topic weights

If you include example results or screenshots, you can paste them below (e.g. bar plots, word clouds).

---

## Future Work

* **Grid search** over `k` and `n_topics` to find optimal values
* Integrate **n-grams**, custom stopwords, or lemmatization
* Add **visualization modules** (topic word clouds, cluster plots)
* Support **online updating** or streaming document addition
* Add **model persistence / loading / incremental inference**

---

## Directory Structure

```
document-clustering-topic-modeling/
├── data/                           # Add your raw document data here
├── notebooks/
│   └── Document Clustering and Topic Modeling Project.ipynb
├── src/
│   └── topic_modeling.py           # Pipeline: TF-IDF, KMeans, LDA
├── results/                        # Generated outputs
├── requirements.txt
├── LICENSE                         # MIT License
├── .gitignore
└── README.md
```

---

## License & Citation

This project is licensed under the **MIT License**. See `LICENSE` for full text.

If you use or build upon this work, please cite:

```
Cris Wang. Document Clustering & Topic Modeling. GitHub repository, 2025.
```

