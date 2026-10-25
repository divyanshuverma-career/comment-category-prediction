# Comment Category Prediction

A machine learning project for **multiclass comment classification** using text, engagement signals, and structured metadata. Developed as an academic Kaggle competition project, this notebook explores NLP feature extraction, feature engineering, model tuning, and ensemble learning.

**Best recorded validation Macro F1:** `0.83274` with LightGBM  

link to the Kaggle notebook: https://www.kaggle.com/code/divyanshu08999/23f2005520-notebook-t12026?scriptVersionId=307641015
## Problem Statement

Given a comment and its associated metadata, predict one of four platform-assigned categories (`label`: 0, 1, 2, or 3). Because the target classes are imbalanced, the project uses **Macro F1** as the primary evaluation metric to give each class equal importance.

## Dataset

The competition provides three files:

- `train.csv`: labelled examples, containing the target `label` and all feature columns.
- `test.csv`: examples with the same input features but without `label`.
- `sample_submission.csv`: template showing the required competition submission format.

The notebook uses a training set of **198,000 rows × 15 columns** and a test set of **102,000 rows × 14 columns**.

### Feature description

| Column | Description |
|---|---|
| `comment` | Raw text content of the comment |
| `created_date` | Date and time when the comment was posted |
| `post_id` | Identifier linking the comment to a discussion thread or parent post |
| `emoticon_1`, `emoticon_2`, `emoticon_3` | Indicators for the presence of symbols from three internal emoticon groups |
| `upvote`, `downvote` | Counts of positive and negative reactions |
| `if_1`, `if_2` | Platform-provided internal features whose meanings are not disclosed |
| `race`, `religion`, `gender`, `disability` | Indicators of detected references to identity, belief, gender, or disability-related topics |
| `label` | Target category assigned by the platform; takes one of four distinct values |

The dataset combines unstructured text with numeric and categorical signals. The modelling approach uses these complementary feature types rather than relying on comment text alone. The meanings of `if_1` and `if_2` are not assumed beyond the dataset description.

## Approach

### 1. Exploratory data analysis

- Inspected dataset dimensions, data types, missing values, duplicates, and descriptive statistics.
- Examined the target-label distribution to understand class imbalance.
- Explored comment length and emoticon usage across labels.
- Used a correlation heatmap to inspect relationships among numeric features.

### 2. Data cleaning and preprocessing

- Converted numeric columns to numeric values and handled invalid or missing values.
- Filled missing comment text with an empty string and missing categorical values with a placeholder.
- Normalized comment text by lowercasing, removing URLs and punctuation, reducing repeated characters, and collapsing extra whitespace.
- Applied consistent preprocessing to training and test data.

### 3. Feature engineering

Created additional features from existing columns to capture engagement and writing style:

- **Engagement:** `net_votes`, `total_votes`, `vote_ratio`, `upvote_rate`, and `controversy_score`.
- **Count transformations:** `log_upvote` and `log_downvote`, using `log1p` to reduce skew in reaction counts.
- **Text structure:** `comment_length`, `word_count`, `avg_word_len`, and `unique_word_ratio`.
- **Punctuation and emphasis:** `has_question`, `has_exclamation`, and `caps_ratio`.

These engineered features supplement the original numeric and categorical columns.

### 4. Feature extraction and transformation

A `ColumnTransformer` combines multiple representations:

- **Word-level TF-IDF:** up to 20,000 features, using unigrams and bigrams to represent individual terms and short phrases.
- **Character-level TF-IDF:** character n-grams of length 3–5, up to 8,000 features, helping capture spelling patterns and word-form variation.
- **Numeric features:** standardized with `StandardScaler`.
- **Categorical features:** converted into indicator columns using `OneHotEncoder(handle_unknown='ignore')`.

This hybrid representation combines lexical information with structured metadata.

### 5. Model development

| Model / technique | Purpose |
|---|---|
| Logistic Regression | Linear baseline well suited to sparse, high-dimensional TF-IDF features |
| XGBoost | Gradient-boosted decision trees for nonlinear patterns and feature interactions |
| LightGBM | Efficient gradient boosting for structured and engineered features |
| GridSearchCV / RandomizedSearchCV | Search for hyperparameter configurations using cross-validation |
| Weighted soft voting | Combine model class probabilities into an ensemble prediction |

Class-balancing options are used in selected models to address the imbalanced target. The notebook uses a stratified 80:20 train-validation split and a fixed random seed (`42`) for reproducibility where supported.

## Evaluation

### Why Macro F1?

For each class, F1 is the harmonic mean of precision and recall. Macro F1 calculates the F1 score for each class and then averages the scores equally:

\[
F1_{\text{macro}} = \frac{1}{K}\sum_{k=1}^{K}F1_k
\]

This is useful for imbalanced multiclass classification because minority classes contribute equally to the final metric rather than being overwhelmed by the most frequent class.

### Recorded validation results

| Model / experiment | Validation Macro F1 |
|---|---:|
| Logistic Regression tuning experiment | `0.7960` |
| XGBoost tuning experiment | `0.7909` |
| LightGBM tuning experiment | **`0.8247`** |

The notebook's recorded LightGBM classification report shows the following per-class F1 scores:

| Class | F1-score |
|---:|---:|
| 0 | 0.9637 |
| 1 | 0.7826 |
| 2 | 0.8888 |
| 3 | 0.6638 |
| **Macro average** | **0.8247** |

The results indicate that performance varies by class, with class `3` being more challenging than the other classes. The best recorded individual-model validation result is LightGBM.

**Metric note:** `0.8247` is the notebook's recorded validation Macro F1 for LightGBM. The approximately `0.82` Kaggle score is the competition score reported for the project; validation and leaderboard scores may differ.

## Tech Stack

- **Language:** Python
- **Data processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine learning:** scikit-learn
- **Gradient boosting:** XGBoost, LightGBM
- **Environment:** Kaggle Notebooks / Jupyter

## Run the Notebook

### On Kaggle

1. Open the notebook in Kaggle.
2. Attach the competition dataset.
3. Run the notebook cells in order.
4. Review validation metrics and the selected model.
5. Confirm that the submission file is generated in the expected format.

The notebook currently reads the data from Kaggle input paths. Update these paths if the competition dataset is mounted elsewhere:

```python
train = pd.read_csv('/kaggle/input/comment-category-prediction-challenge/train.csv')
test = pd.read_csv('/kaggle/input/comment-category-prediction-challenge/test.csv')
sample_submission = pd.read_csv('/kaggle/input/comment-category-prediction-challenge/Sample.csv')
```

### Local environment

Install the main dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm jupyter
```

Download the competition data according to the competition rules, update the file paths, and run the notebook. The notebook uses `SEED = 42`, but exact results may vary with library versions and model configurations.

## Key Skills Demonstrated

- Exploratory data analysis and class-imbalance assessment
- Text cleaning and NLP feature extraction with word- and character-level TF-IDF
- Feature engineering from text, engagement, and metadata
- Preprocessing mixed data types with scikit-learn pipelines
- Training and evaluating linear and tree-based classifiers
- Hyperparameter search and model comparison
- Macro F1 evaluation for imbalanced multiclass classification
- Probability-based ensembling and Kaggle submission preparation

## Future Improvements

- Refit the selected model or ensemble on all labelled data before test prediction.
- Validate ensemble weights with cross-validation or out-of-fold predictions.
- Inspect a confusion matrix and per-class errors to improve minority-class performance.
- Compare text-only, metadata-only, and combined-feature baselines.
- Record exact final model parameters and library versions for reproducibility.

## Repository Structure

```text
comment-category-prediction/
├── README.md
└── comment_category_prediction.ipynb
```

The competition data files are not included in this repository. Add the notebook and this README to the root of your GitHub repository.

---

**Project type:** Academic machine learning / NLP classification  
**Objective:** Predict comment categories from text and structured metadata  
**Best recorded validation Macro F1:** `0.8247`
