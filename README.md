# Movie Genre Classification Using NLP

Predicting a movie's genre from just a snippet of its script text, using classic NLP preprocessing and a Naive Bayes classifier.

## 📌 Overview

Given a piece of movie script/dialogue, this project builds a machine learning model that classifies it into one of **9 genres**:

`other, action, adventure, comedy, drama, horror, romance, sci-fi, thriller`

**Dataset source:** [Kaggle - Movie Genres Competition](https://www.kaggle.com/c/moviegenres/overview)

## 🗂️ Project Structure

```
Movie Genre Classification Using NLP/
│
├── data/
│   ├── train.csv          # Training data (id, text, genre)
│   └── test.csv           # Test data (id, text)
│
├── Movie_Genre_Classification_Using_NLP.ipynb   # Main notebook
└── README.md
```

## 🔍 Approach

1. **Data Cleaning** — checked for null values, duplicate rows, unnecessary columns (`id`), and verified genre labels for typos.
2. **Exploratory Data Analysis (EDA)** — visualized the distribution of scripts across genres to understand class balance.
3. **Text Preprocessing** — cleaned raw script text by:
   - Removing punctuation/numbers (keeping only letters)
   - Lowercasing all text
   - Removing English stopwords
   - Stemming words to their root form (Porter Stemmer)
4. **Feature Extraction** — converted cleaned text into numeric vectors using **Bag of Words** (`CountVectorizer`, top 10,000 unigrams + bigrams).
5. **Model Training** — trained a **Multinomial Naive Bayes** classifier on an 80/20 train-test split.
6. **Evaluation** — measured accuracy and visualized a confusion matrix to see genre-wise performance.
7. **Hyperparameter Tuning** — tuned the Naive Bayes `alpha` (smoothing) parameter to improve accuracy.
8. **Word Clouds** — visualized the most frequent words associated with specific genres.
9. **Prediction** — built a reusable function to predict the genre of any new script snippet.
10. **Submission File** — generated genre predictions for the entire test set in Kaggle submission format.

## 📊 Results

| Model | Accuracy |
|---|---|
| Multinomial Naive Bayes (default) | 89.57% |
| Multinomial Naive Bayes (tuned, alpha=0.1) | **91.41%** |

**Key insight:** The dataset is imbalanced — `drama` and `thriller` scripts dominate the data, while genres like `romance`, `adventure`, and `sci-fi` are underrepresented. This leads to strong accuracy on majority classes but more misclassification for minority genres (e.g., some `sci-fi`/`thriller` overlap, and minority genres occasionally get absorbed into `drama`).

## 🛠️ Tech Stack

- **Python 3**
- **Pandas / NumPy** — data handling
- **Matplotlib / Seaborn** — visualization
- **NLTK** — stopwords removal & stemming
- **Scikit-learn** — CountVectorizer, train-test split, Multinomial Naive Bayes, evaluation metrics
- **WordCloud** — genre-wise word visualization

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone <your-repo-url>
   cd "Movie Genre Classification Using NLP"
   ```

2. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn nltk scikit-learn wordcloud
   ```

3. Open the notebook and run all cells:
   ```bash
   jupyter notebook Movie_Genre_Classification_Using_NLP.ipynb
   ```
   Or upload it to [Google Colab](https://colab.research.google.com) and upload `train.csv` / `test.csv` from the `data/` folder when prompted.

4. The notebook will output:
   - EDA visualizations (genre distribution, word clouds)
   - Model accuracy and confusion matrix
   - `submission.csv` — genre predictions for the full test set

## 📁 Dataset

- `train.csv` — 22,579 rows with columns `id`, `text` (script snippet), `genre` (label)
- `test.csv` — 5,589 rows with columns `id`, `text` (no label — used for prediction)

Source: [Kaggle Movie Genres Competition](https://www.kaggle.com/c/moviegenres/overview)

## 📈 Future Improvements

- Try **TF-IDF** vectorization instead of raw Bag of Words
- Experiment with other classifiers (Logistic Regression, Linear SVM, Random Forest)
- Address class imbalance using oversampling (SMOTE) or class weighting
- Try word embeddings (Word2Vec, GloVe) or transformer-based models (BERT) for richer text representation

## 📄 License

This project is for educational purposes, built on top of publicly available Kaggle data.
