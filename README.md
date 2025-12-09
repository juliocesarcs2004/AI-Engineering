# AI-Engineering — LLM & NLP Projects

**A comprehensive collection of Large Language Models (LLM) and Natural Language Processing (NLP) learning projects with Jupyter notebooks, datasets, and practical examples.**

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [LLMs_Projects](#llms_projects)
- [NLP_Projects](#nlp_projects)
- [Environment & Requirements](#environment--requirements)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Best Practices](#best-practices)
- [Git Configuration](#git-configuration)
- [Contributing](#contributing)
- [License & Contact](#license--contact)

---

## Overview

This repository contains two complementary learning paths for AI/ML engineers:

1. **LLMs_Projects** — Hands-on notebooks for working with Large Language Models (GPT models, LangChain, and Hugging Face Transformers)
2. **NLP_Projects** — Comprehensive NLP tutorials covering text preprocessing, sentiment analysis, topic modeling, and text classification

Both sections are designed for:
- **Learning & experimentation** with state-of-the-art AI models
- **Reproducibility** on local machines with moderate resources
- **Clear documentation** with step-by-step implementations

---

## Project Structure

```
AI-Engineering/
├── README.md                          # This file
├── .gitignore                         # Git ignore rules
├── LLMs_Projects/                     # Large Language Models projects
│   ├── config.py                      # API configuration (ignored in git)
│   ├── GPT Models.ipynb               # OpenAI GPT API interactions
│   ├── Langchain.ipynb                # LangChain framework examples
│   ├── HuggingFace Transformers.ipynb # Hugging Face model usage
│   ├── Text classification with XLNET.ipynb # XLNet emotion classification
│   ├── emotions_data/                 # Emotion labeled datasets
│   ├── test_trainer/                  # XLNet model checkpoints
│   ├── my_saved_models/               # Directory for cached/saved models
│   └── fine_tuned_model/              # Fine-tuned model artifacts
│
└── NLP_Projects/                      # Natural Language Processing projects
    ├── NLP_Text_Pre_Processing.ipynb         # Text cleaning, tokenization, lemmatization
    ├── NLP_Sentiment_Analysis.ipynb          # VADER & transformer-based sentiment
    ├── NLP_POS_and_NER.ipynb                 # Part-of-speech tagging & named entities
    ├── NLP_Text_Classifier.ipynb             # Text classification techniques
    ├── NLP_Topic_Modelling_LDA.ipynb         # Latent Dirichlet Allocation
    ├── NLP_Topic_Modelling_LSA.ipynb         # Latent Semantic Analysis
    ├── NLP_Vectorizing_Text_Count_Vectorizer.ipynb  # Count-based text vectors
    ├── NLP_Vectorizing_Text_TF-IDF.ipynb     # TF-IDF vectorization
    ├── NLP_Categorizing_Fake_News.ipynb      # Fake news detection pipeline
    ├── bbc_news.csv                          # BBC news articles dataset
    ├── fake_news_data.csv                    # Labeled fake news dataset
    ├── news_articles.csv                     # News corpus for topic modeling
    ├── book_reviews_sample.csv               # Book reviews for sentiment analysis
    └── tripadvisor_hotel_reviews.csv         # Hotel reviews dataset
```

---

## LLMs_Projects

### Overview
Practical tutorials for working with modern Large Language Models including OpenAI's GPT models, LangChain framework, and Hugging Face Transformers.

### Key Features
- **API Integration** — Direct integration with OpenAI GPT models
- **Prompt Engineering** — Techniques for crafting effective prompts
- **LangChain Framework** — Building chains and agents with LLMs
- **Hugging Face Models** — Using pre-trained transformer models locally

### Notebooks

#### `GPT Models.ipynb`
Demonstrates OpenAI's GPT API usage:
- API setup and authentication via `config.py`
- Text generation with `davinci-002` engine
- Customizing output (temperature, max_tokens)
- Chat completion with GPT-3.5-turbo
- Text summarization and keyword extraction
- Poetic chatbot implementation

**Key functions:**
```python
def generate_text(prompt, max_tokens, temperature)
def text_summarizer(prompt)
```

**Libraries:** `openai`, `config`

---

#### `Langchain.ipynb`
Introduction to the LangChain framework:
- LLM model initialization and configuration
- Prompt templates and prompt chaining
- Memory management for multi-turn conversations
- Agents and tools integration
- Chain orchestration patterns

**Key concepts:**
- Retrieval-Augmented Generation (RAG)
- Custom chains composition
- Multi-step reasoning pipelines

**Libraries:** `langchain`, `openai`

---

#### `HuggingFace Transformers.ipynb`
Working with Hugging Face pre-trained models:
- Model loading and inference
- Pipeline abstractions for common tasks
- Fine-tuning strategies
- Model serialization and caching
- Performance optimization

**Common tasks:**
- Text classification
- Sentiment analysis
- Question answering
- Text generation

**Libraries:** `transformers`, `torch`/`tensorflow`, `datasets`

---

#### `Text classification with XLNET.ipynb`
End-to-end emotion classification using XLNet transformer model:

**Pipeline:**
1. **Data Loading & Preprocessing**
   - Load emotion-labeled training, validation, and test datasets
   - Text cleaning using `cleantext` library (remove emojis, special characters)
   - Remove mentions (@username)
   - Balance dataset using stratified sampling

2. **Exploratory Data Analysis**
   - Visualize label distribution before/after balancing
   - Ensure balanced representation of all emotion classes

3. **Tokenization**
   - Use XLNet tokenizer (xlnet-base-cased)
   - Padding and truncation to max_length=128
   - Create token_type_ids and attention_mask
   - Convert to HuggingFace datasets format

4. **Model Fine-tuning**
   - Load pre-trained XLNetForSequenceClassification
   - Configure for 4-class emotion classification (anger, fear, joy, sadness)
   - Define custom compute_metrics (accuracy)
   - Train using HuggingFace Trainer API
   - 3 epochs with evaluation every epoch

5. **Output**
   - Trained model checkpoints in `test_trainer/` directory
   - Evaluation metrics and model performance

**Dataset Requirements:**
- CSV files with columns: `text`, `label`
- Expected location: `./emotions_data/` (relative to notebook)
- Files: `emotion-labels-train.csv`, `emotion-labels-test.csv`, `emotion-labels-val.csv`

**Key Components:**
```python
tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', 
                                                       num_labels=4,
                                                       id2label={0: 'anger', 1: 'fear', 2: 'joy', 3: 'sadness'})
trainer = Trainer(model=model, args=training_args, 
                  train_dataset=tokenized_data,
                  compute_metrics=compute_metrics)
trainer.train()
```

**Libraries:** `transformers`, `torch`, `datasets`, `evaluate`, `cleantext`, `sklearn`

---

### Configuration

**File:** `config.py` (Git-ignored for security)

```python
api_key = "your-openai-api-key-here"
# Add other LLM provider keys as needed
```

⚠️ **Security Note:** Never commit API keys. The file is in `.gitignore`.

---

### Saved Models Directory

Large model files are stored in `my_saved_models/` and excluded from version control. This directory is safe for:
- Downloaded pre-trained models
- Fine-tuned model checkpoints
- Model artifacts and weights

---

## NLP_Projects

### Overview
Comprehensive collection of NLP tutorials covering the full pipeline from raw text to advanced ML applications.

### Key Features
- **Text Preprocessing** — Cleaning, tokenization, stemming, lemmatization
- **Sentiment Analysis** — VADER lexicon-based and transformer-based approaches
- **Text Vectorization** — Count vectorizer and TF-IDF techniques
- **Topic Modeling** — LDA and LSA implementations
- **NER & POS Tagging** — Named entity recognition and part-of-speech analysis
- **Text Classification** — ML and DL approaches
- **Fake News Detection** — Multi-stage classification pipeline

### Notebooks

#### `NLP_Text_Pre_Processing.ipynb`
Foundation for all NLP tasks:
- Lowercase normalization
- Punctuation removal
- Stopword filtering
- Tokenization (word & sentence)
- Stemming (Porter Stemmer)
- Lemmatization (WordNet)
- Custom preprocessing pipelines

**Output:** Clean, tokenized data ready for modeling

---

#### `NLP_Sentiment_Analysis.ipynb`
Two complementary approaches to sentiment analysis:

**VADER (Lexicon-based):**
- Fast, no training required
- Scores: compound score (-1 to 1)
- Categories: negative, neutral, positive
- Dataset: `book_reviews_sample.csv`

**Transformer-based:**
- Deep learning with pre-trained models
- More context-aware predictions
- Comparison with VADER scores

**Output:** Sentiment labels and scores for comparison

---

#### `NLP_POS_and_NER.ipynb`
Linguistic analysis tasks:
- **POS Tagging:** Identify verbs, nouns, adjectives, etc.
- **NER:** Extract people, organizations, locations
- Libraries: `nltk`, `spaCy`
- Visualization of linguistic structures

---

#### `NLP_Text_Classifier.ipynb`
Supervised text classification:
- Feature extraction (TF-IDF, embeddings)
- Model training (Naive Bayes, SVM, etc.)
- Hyperparameter tuning
- Evaluation metrics (precision, recall, F1)

---

#### `NLP_Topic_Modelling_LDA.ipynb`
Latent Dirichlet Allocation:
- Unsupervised topic discovery
- Dataset: `news_articles.csv`
- Document-term matrix creation
- Visualization of topic distributions
- Topic interpretation

**Key steps:**
1. Text preprocessing and stemming
2. Dictionary and corpus creation
3. LDA model training (2+ topics)
4. Topic extraction and visualization

---

#### `NLP_Topic_Modelling_LSA.ipynb`
Latent Semantic Analysis:
- SVD-based dimensionality reduction
- Conceptual similarity between documents
- Comparison with LDA approach
- Interpretability of latent factors

---

#### `NLP_Vectorizing_Text_Count_Vectorizer.ipynb`
Bag-of-Words vectorization:
- Frequency-based text representation
- Vocabulary building
- Sparse matrix output
- Baseline for text classification

---

#### `NLP_Vectorizing_Text_TF-IDF.ipynb`
TF-IDF vectorization:
- Term frequency-inverse document frequency
- Importance weighting
- Dimensionality considerations
- Comparison with Count Vectorizer

---

#### `NLP_Categorizing_Fake_News.ipynb`
End-to-end fake news classification:
- Multi-class classification (real vs. fake)
- Feature engineering pipeline
- Model comparison and selection
- Evaluation on imbalanced data
- Dataset: `fake_news_data.csv`

---

### Datasets

| Dataset | Size | Purpose | Format |
|---------|------|---------|--------|
| `bbc_news.csv` | BBC articles | Topic modeling, text classification | CSV |
| `fake_news_data.csv` | Fake/real news | Fake news detection | CSV |
| `news_articles.csv` | News corpus | Topic modeling (LDA/LSA) | CSV |
| `book_reviews_sample.csv` | Book reviews | Sentiment analysis | CSV |
| `tripadvisor_hotel_reviews.csv` | Hotel reviews | Sentiment & preprocessing | CSV |

---

## Environment & Requirements

### Python Version
- **Recommended:** Python 3.9 or newer
- **Minimum:** Python 3.8

### Virtual Environment Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate (macOS/Linux)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```

### Dependencies

#### Core Libraries
```bash
pip install pandas numpy scikit-learn
```

#### NLP Libraries
```bash
pip install nltk spacy gensim
python -m spacy download en_core_web_sm
```

#### LLM & Transformers
```bash
pip install openai langchain transformers torch
# or tensorflow instead of torch
pip install tensorflow
```

#### Sentiment Analysis
```bash
pip install vaderSentiment
```

#### Jupyter
```bash
pip install jupyterlab notebook ipykernel
```

#### Complete Installation (all projects)
```bash
pip install pandas numpy scikit-learn nltk spacy gensim \
            openai langchain transformers torch \
            vaderSentiment jupyterlab notebook
python -m spacy download en_core_web_sm
```

Or use a `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## Quick Start

### 1. Clone and Setup
```bash
git clone https://github.com/juliocesarcs2004/AI-Engineering.git
cd AI-Engineering
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Launch Jupyter
```bash
jupyter lab
```

### 3. NLP Path (Start here if new to NLP)
1. Open `NLP_Projects/NLP_Text_Pre_Processing.ipynb`
2. Run through preprocessing examples
3. Move to `NLP_Sentiment_Analysis.ipynb`
4. Explore topic modeling: `NLP_Topic_Modelling_LDA.ipynb`

### 4. LLM Path (Requires API keys)
1. Create `LLMs_Projects/config.py` with your OpenAI API key
2. Open `LLMs_Projects/GPT Models.ipynb`
3. Run examples and experiment with prompts
4. Explore `Langchain.ipynb` for advanced patterns

---

## Usage Examples

### NLP: Basic Sentiment Analysis
```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

# Load data
df = pd.read_csv('NLP_Projects/book_reviews_sample.csv')

# Analyze sentiment
analyzer = SentimentIntensityAnalyzer()
df['sentiment_score'] = df['reviewText'].apply(
    lambda x: analyzer.polarity_scores(x)['compound']
)

print(df[['reviewText', 'sentiment_score']].head())
```

### NLP: Text Preprocessing Pipeline
```python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt')

text = "Natural Language Processing is amazing!"
tokens = word_tokenize(text.lower())
stop_words = set(stopwords.words('english'))
filtered = [t for t in tokens if t not in stop_words]
stemmer = PorterStemmer()
stemmed = [stemmer.stem(t) for t in filtered]

print(stemmed)
```

### LLM: Simple Text Generation
```python
import openai
from LLMs_Projects import config

openai.api_key = config.api_key

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ]
)

print(response.choices[0].message.content)
```

### LLM: LangChain Chain
```python
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

llm = OpenAI(temperature=0.7, openai_api_key="your-key")

prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a short poem about {topic}"
)

chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(topic="AI")
print(result)
```

---

## Best Practices

### Code Quality
- Always use a virtual environment to isolate dependencies
- Save random seeds for reproducibility:
  ```python
  import random, numpy as np
  random.seed(42)
  np.random.seed(42)
  ```
- Use descriptive variable names and add comments

### Data Handling
- Create a `data/` directory for organizing datasets
- Save intermediate processed data to avoid recomputation
- Document data sources and preprocessing steps
- Use `.gitignore` for large files (> 100 MB)

### Model Development
- Start with baseline models before complex ones
- Always evaluate on a separate test set
- Log hyperparameters and results
- Version control important model checkpoints

### API Usage (LLMs)
- Store API keys in `config.py` (Git-ignored)
- Monitor token usage for cost control
- Implement rate limiting for API calls
- Use appropriate temperature for your use case:
  - **0.0** = deterministic (good for structured tasks)
  - **0.7** = balanced (good for general use)
  - **1.0+** = creative (good for creative writing)

---

## Git Configuration

### Ignored Files
The `.gitignore` file excludes:
- `LLMs_Projects/config.py` — API keys and credentials
- `__pycache__/` — Python cache files
- `LLMs_Projects/my_saved_models/` — Large model files
- `.ipynb_checkpoints/` — Jupyter checkpoints

### Large File Handling
Do not commit files > 100 MB. If added accidentally:
```bash
git rm --cached LLMs_Projects/my_saved_models/model.safetensors
git commit -m "Remove large model file"
```

---

## Contributing

### Workflow
1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/my-nlp-tutorial`
3. **Implement** your changes with clear documentation
4. **Test** locally to ensure notebooks run without errors
5. **Commit** with descriptive messages
6. **Push** and **create a Pull Request**

### Guidelines
- Add clear markdown headers and explanations in notebooks
- Include docstrings for helper functions
- Update this README for new notebooks or datasets
- Keep notebook file sizes reasonable (< 50 MB)
- Verify all cells run without errors before submitting

### Adding New Notebooks
If adding a new notebook:
1. Create file in appropriate folder (`LLMs_Projects/` or `NLP_Projects/`)
2. Add description to README.md
3. Include dataset path references if needed
4. Add learning objectives in first markdown cell

---

## License & Contact

### License
No license file is currently included. To enable public sharing, consider adding:
- **Recommended:** MIT License (permissive, widely used)
- **Alternative:** Creative Commons for educational content

### Repository Information
- **Owner:** [@juliocesarcs2004](https://github.com/juliocesarcs2004)
- **Repository:** [AI-Engineering](https://github.com/juliocesarcs2004/AI-Engineering)

### Contact & Issues
- Use **GitHub Issues** to report bugs or ask questions
- For feature requests, open a discussion or issue

### Acknowledgments
- OpenAI for GPT models
- Hugging Face for transformer models and datasets
- NLTK and spaCy communities for NLP tools
- Researchers and educators in the AI community

---

## Next Steps

Consider implementing:
- [ ] `requirements.txt` with pinned versions
- [ ] GitHub Actions CI/CD for notebook validation
- [ ] API documentation for custom functions
- [ ] Performance benchmarks for models
- [ ] Docker setup for reproducible environments
- [ ] Automated data downloading scripts
- [ ] Additional languages (Portuguese, Spanish, etc.)

---

**Last Updated:** December 5, 2025  
**Status:** Actively maintained  
**Contributions:** Welcome!

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

No license file is included. Use the repository Issues to report questions or request features.


