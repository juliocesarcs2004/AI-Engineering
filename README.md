**AI-Engineering — NLP Projects**

**Overview**
- **Project:**: A small collection of natural language processing (NLP) notebooks and datasets for learning and experiments.
- **Purpose:**: Provide clear examples for text preprocessing, part-of-speech tagging, named-entity recognition, and sentiment-related exploration using public datasets.

**Data**
- **Files:**: `bbc_news.csv`, `tripadvisor_hotel_reviews.csv`
- **Description:**: CSV files with news articles and hotel reviews used in the notebooks for demonstrations and exercises.

**Notebooks**
- **Files:**: `NLP_POS_and_NER.ipynb`, `NLP_Text_Pre_Processing.ipynb`
- **What they do:**: The notebooks show practical steps for cleaning text, tokenization, POS tagging, NER, and common pre-processing pipelines for ML/NLP tasks.

**Project Structure**
- **Root files:**: Datasets and notebooks live in the project root.
- **Where to start:**: Open `NLP_Text_Pre_Processing.ipynb` for basic cleaning and tokenization, then explore `NLP_POS_and_NER.ipynb` for tagging and entity recognition.

**Requirements**
- **Recommendation:**: Use Python 3.8+ and install common NLP libraries such as `pandas`, `numpy`, `nltk`, `spacy`, and `scikit-learn`.
- **Example install:**
```
python -m pip install --upgrade pip
python -m pip install pandas numpy nltk spacy scikit-learn jupyterlab
python -m spacy download en_core_web_sm
```

**How to run**
- **Start Jupyter Lab:**
```bash
jupyter lab
```
- **Open a notebook:**: Click on `NLP_Text_Pre_Processing.ipynb` or `NLP_POS_and_NER.ipynb` in the browser interface.

**Tips**
- **Reduce memory use:**: If datasets are large, consider loading a subset with `pandas.read_csv(..., nrows=10000)` for experiments.
- **Repeatability:**: Set random seeds where needed (for example in `scikit-learn`) to make experiments reproducible.

**Contributing**
- **How to help:**: Add new notebooks, improved preprocessing steps, or small scripts to load and clean the datasets. Open a pull request with a short description of the change.

**License & Contact**
- **License:**: This repository does not include a license file. Add one if you plan to share widely (for example, `MIT`).
- **Contact:**: For questions or suggestions, open an issue in this repository.

---
