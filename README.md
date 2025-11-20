**AI-Engineering — NLP Projects**

**Overview**
- **Project:**: A small learning collection of natural language processing (NLP) notebooks and example datasets.
- **Goal:**: Help learners and researchers practice common NLP tasks such as text cleaning, tokenization, part-of-speech (POS) tagging, named-entity recognition (NER), and basic sentiment exploration.

**Datasets**
- **Files included:**: `bbc_news.csv`, `tripadvisor_hotel_reviews.csv` (CSV format).
- **Short description:**: `bbc_news.csv` contains news articles (useful for topic and text classification experiments). `tripadvisor_hotel_reviews.csv` contains hotel reviews (useful for sentiment analysis and preprocessing exercises).

**Notebooks (high level)**
- `NLP_Text_Pre_Processing.ipynb` — Basic text cleaning, tokenization, stopword removal, lemmatization, and example pipelines for preparing data for ML models.
- `NLP_POS_and_NER.ipynb` — Demonstrations of part-of-speech tagging and named-entity recognition using libraries like `nltk` and `spaCy`.

**Why this repo is useful**
- **Learning-focused:**: Notebooks include clear, commented steps so learners can follow and adapt the code.
- **Small and reproducible:**: Designed to run on a local machine with moderate RAM.

**Environment & Requirements**
- **Python version:**: Recommended 3.8 or newer.
- **Main libraries:**: `pandas`, `numpy`, `nltk`, `spacy`, `scikit-learn`, `jupyterlab` or `notebook`.
- **Suggested virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate
```
- **Install dependencies:**
```bash
pip install --upgrade pip
pip install pandas numpy nltk spacy scikit-learn jupyterlab
python -m spacy download en_core_web_sm
```

**Quick Start**
- Open the project folder in VS Code or start Jupyter Lab:
```bash
# From project root
jupyter lab
# or
jupyter notebook
```
- Then open and run cells in `NLP_Text_Pre_Processing.ipynb` to begin. Work through `NLP_POS_and_NER.ipynb` for tagging and entity examples.

**Usage examples**
- Load a small sample (faster for testing):
```python
import pandas as pd
df = pd.read_csv('tripadvisor_hotel_reviews.csv', nrows=5000)
```
- Example: basic tokenization with spaCy
```python
import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp('This is an example sentence for tokenization and POS tagging.')
for token in doc:
	print(token.text, token.pos_)
```

**Best practices**
- Use a virtual environment to avoid dependency conflicts.  
- Save intermediate datasets (cleaned versions) to speed exploration.  
- Fix random seeds (`random`, `numpy.random`, or `sklearn` functions) for reproducible results.

**Contributing**
- **Additions welcome:**: New notebooks, helper scripts, or improved preprocessing methods are useful.  
- **How to contribute:**: Create a branch, add your changes, and open a pull request with a clear description. If you add packages, include or update a `requirements.txt`.

**Files to consider adding**
- `requirements.txt` — pinned dependencies for easy install.  
- `data/` — a directory to store datasets instead of keeping them at root.  
- `LICENSE` — add a license (for example `MIT`) if you plan to share publicly.

**License & Contact**
- **License:**: No license file is included. Add one if you want to define sharing permissions (recommended: `MIT`).
- **Contact / Issues:**: Use the repository Issues to report problems or ask questions.

---

If you want, I can:
- create a `requirements.txt` with the recommended libraries, or
- move datasets into a `data/` folder and update the notebooks, or
- translate the README into Portuguese.

Thank you — let me know which improvements you prefer next.
 
---

**Table of contents**

- [Overview](#overview)
- [Datasets](#datasets)
- [Notebooks (high level)](#notebooks-high-level)
- [Quick setup](#quick-setup)
- [Usage examples](#usage-examples)
- [Project layout](#project-layout)
- [Best practices](#best-practices)
- [Contributing](#contributing)
- [Suggested next steps](#suggested-next-steps)
- [License & Contact](#license--contact)

## Overview

This repository is a small, learning-focused collection of NLP notebooks and example datasets. It is intended for students and developers who want simple, clear examples of common NLP tasks such as cleaning text, tokenization, POS tagging, NER, and basic sentiment exploration.

## Datasets

- `bbc_news.csv` — News articles useful for topic experiments and classification tasks.
- `tripadvisor_hotel_reviews.csv` — Hotel reviews useful for sentiment analysis and text preprocessing exercises.

Notes:
- The datasets are CSV files in the repository root. If you prefer, move them into a `data/` folder and update the notebook paths.

## Notebooks (high level)

- `NLP_Text_Pre_Processing.ipynb` — Steps for cleaning, tokenization, stopword removal, and lemmatization.
- `NLP_POS_and_NER.ipynb` — Examples using `nltk` and `spaCy` for POS tagging and named-entity recognition (NER).

## Notebooks — Detailed description

Below are detailed descriptions for each notebook in the `NLP Projects/` folder. Each subsection explains the notebook goal, datasets used, main steps, and tips to run the cells in the correct order.

### `NLP_Text_Pre_Processing.ipynb`
- Purpose: Show a typical text preprocessing pipeline applied to the TripAdvisor hotel reviews dataset. Good as a first notebook to understand data cleaning and token-level operations.
- Dataset used: `tripadvisor_hotel_reviews.csv` (reviews live in the `Review` column).
- Main steps in the notebook:
	- Download essential NLTK resources (`stopwords`, `wordnet`, `punkt_tab`).
	- Load the CSV and inspect structure (`.info()`, `.head()`).
	- Convert text to lowercase and create a `review_lowercase` column.
	- Build an English stopword list and remove stopwords (keeps important words like `not`).
	- Replace special characters (e.g. `*` → `star`) and remove punctuation.
	- Tokenize reviews into a `tokenized` column using `nltk.word_tokenize`.
	- Apply stemming (`PorterStemmer`) and lemmatization (`WordNetLemmatizer`) and save results in `stemmed` and `lemmatized` columns.
	- Create token frequency lists and compute unigrams and bigrams counts with `nltk.ngrams`.
- How to run: Open the notebook and run cells top-to-bottom. The first cell installs/downloads NLTK data; run it before tokenization and lemmatization cells.
- Expected outputs: Cleaned text columns, token lists, stemmed/lemmatized samples, and printed n-gram frequency tables.
- Tips: For faster iteration, use `pd.read_csv(..., nrows=2000)` while experimenting. Save intermediate cleaned data (e.g., `data.to_csv('data/clean_reviews.csv', index=False)`) to avoid repeating expensive steps.

### `NLP_POS_and_NER.ipynb`
- Purpose: Demonstrate part-of-speech tagging and named-entity recognition using the BBC news dataset. Good to learn how `spaCy` presents POS tags and entity labels.
- Dataset used: `bbc_news.csv` (uses `title` field in examples).
- Main steps in the notebook:
	- Load `bbc_news.csv` and inspect titles.
	- Lowercase titles and remove stopwords and punctuation.
	- Tokenize raw and cleaned text and lemmatize tokens.
	- Create combined token lists and build a `spaCy` document with `en_core_web_sm`.
	- POS tagging: iterate `spacy_doc` tokens, collect token and `pos_`, create a `pos_df`, and compute aggregated counts for common POS tags (NOUN, VERB, ADJ).
	- NER: extract named entities from `spacy_doc.ents`, build `ner_df`, and aggregate counts by entity label.
- How to run: Ensure `en_core_web_sm` is installed (`python -m spacy download en_core_web_sm`) and run the notebook top-to-bottom. The `spaCy` model load should come after dataset loading and simple token preparation.
- Expected outputs: DataFrames with POS counts, sample tokens by POS, and a table of most frequent named entities and their labels.
- Tips: The notebook uses titles (short text) which runs quickly. To analyze full article text, replace `title` with the appropriate column and expect longer runtime.

### `NLP_Sentiment_Analysis.ipynb`
- Purpose: Compare classical and transformer-based sentiment analysis methods on a sample of book reviews.
- Dataset used: `book_reviews_sample.csv` (column `reviewText`).
- Main steps in the notebook:
	- Load dataset and clean text (lowercase, remove punctuation).
	- Compute VADER compound sentiment scores with `vaderSentiment` and create a binned sentiment label (negative / neutral / positive).
	- Plot counts of VADER labels as a quick overview.
	- Use Hugging Face `transformers` pipeline for `sentiment-analysis` on cleaned reviews and collect transformer labels.
	- Compare or visualize label distributions between methods.
- How to run: Install `vaderSentiment` and `transformers` (`pip install vaderSentiment transformers`) and run cells in order. Transformer pipelines may download models the first time and will take longer.
- Expected outputs: DataFrame columns with `vader_sentiment_score`, `vader_sentiment_label`, and transformer-based labels; bar plots summarizing label counts.
- Tips: If you have no GPU, run transformers on a small subset (e.g., `nrows=200`) to avoid long execution times and to reduce API/model download pressure.

### `NLP_Vectorizing_Text_Count_Vectorizer.ipynb`
- Purpose: Explain how `CountVectorizer` (bag-of-words) converts text into a numeric feature matrix.
- Dataset used: Small in-notebook sample `data` (a short list of sentences) — this is an educational example.
- Main steps:
	- Create a small list of example sentences.
	- Fit `CountVectorizer()` on the list and transform it to a document-term matrix.
	- Convert the matrix to a `pandas.DataFrame` with feature names as columns and print the bag-of-words table.
- How to run: Run the notebook cells top-to-bottom. No external dataset or heavy dependency is required beyond `scikit-learn`.
- Expected outputs: A printed DataFrame that shows token counts per sentence.

### `NLP_Vectorizing_Text_TF-IDF.ipynb`
- Purpose: Demonstrate `TfidfVectorizer` to compute TF-IDF scores for tokens across documents.
- Dataset used: Same small example list as the CountVectorizer notebook.
- Main steps:
	- Define the same example sentence list.
	- Fit `TfidfVectorizer()` and transform the sentences into TF-IDF vectors.
	- Convert to a `pandas.DataFrame` and print TF-IDF weights for tokens.
- How to run: Run top-to-bottom; requires `scikit-learn` but no external dataset.
- Expected outputs: A DataFrame showing TF-IDF values for each token per sentence.


## Quick setup

1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install packages and the spaCy model:

```bash
pip install --upgrade pip
pip install pandas numpy nltk spacy scikit-learn jupyterlab
python -m spacy download en_core_web_sm
```

3. Start Jupyter Lab and open the notebooks:

```bash
jupyter lab
```

## Usage examples

- Load a quick sample (fast for development):

```python
import pandas as pd
df = pd.read_csv('tripadvisor_hotel_reviews.csv', nrows=5000)
df.head()
```

- Tokenize and POS-tag with spaCy:

```python
import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp('This is an example sentence for tokenization and POS tagging.')
[(token.text, token.pos_) for token in doc]
```

## Project layout

Current files at repository root:

- `bbc_news.csv` — dataset
- `tripadvisor_hotel_reviews.csv` — dataset
- `NLP_Text_Pre_Processing.ipynb` — notebook
- `NLP_POS_and_NER.ipynb` — notebook

Recommended layout (optional):

- `data/` — datasets
- `notebooks/` — move notebooks here
- `requirements.txt` — pinned dependencies
- `LICENSE` — license file

## Best practices

- Use a virtual environment to manage dependencies.  
- Pin dependencies in `requirements.txt` for reproducibility.  
- Save cleaned/processed datasets to speed up repeated experiments.  
- Set seeds for random operations (`numpy.random.seed`, scikit-learn's `random_state`).

## Contributing

- Add notebooks, scripts, or small utilities that improve reproducibility or clarity.  
- When adding dependencies, update `requirements.txt`.  
- Open a pull request and explain the purpose and tests performed.

## Suggested next steps

- I can create a `requirements.txt` file with the main packages pinned.
- I can add a `data/` folder and update notebooks to read from it.
- I can translate this README into Portuguese.

Tell me which of these you want and I will implement it.

## License & Contact

No license file is included. Add a license (for example `MIT`) if you plan to publish the code. Use the repository Issues to report questions or request features.


