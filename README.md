# comment-category-prediction

Comment Category Prediction Challenge
A machine learning project for predicting the category of a comment from its text, engagement signals, and additional metadata. The project was developed in a Kaggle notebook as part of an academic course project and follows an end-to-end classification workflow: data inspection, exploratory data analysis (EDA), preprocessing, feature engineering, model training, validation, model comparison, and submission generation.
Best recorded validation result: LightGBM achieved a Macro F1 score of 0.8247 on the 20% validation split. The project score reported on Kaggle was approximately 0.82.

Table of Contents
- Project Overview
- Problem Statement
- Dataset
- Workflow
- Exploratory Data Analysis
- Data Preprocessing
- Feature Engineering
- Feature Transformation
- Models and Training
- Evaluation Metric
- Validation Results
- Generating the Submission
- Tech Stack
- How to Run
- Key Learnings
- Limitations and Future Improvements
Project Overview
The goal is to classify each comment into one of four target classes: 0, 1, 2, or 3. The notebook combines natural language processing (NLP) techniques with structured metadata such as votes, emoticon counts, and categorical attributes.

Rather than relying on comment text alone, the model uses several complementary kinds of information:
- Text signals: words, short phrases, and character patterns.
- Engagement signals: upvotes and downvotes.
- Comment structure: length, word count, punctuation indicators, and capitalization.
- Categorical metadata: race, religion, gender, and disability fields.
- Ensemble learning: combines the probability predictions of multiple classifiers.
The notebook explores Logistic Regression, XGBoost, and LightGBM, then evaluates a weighted soft-voting ensemble.
Problem Statement
This is a supervised, multiclass text-classification task. Each training example contains a comment and its associated features, along with a target label. The model learns patterns from labelled examples and predicts labels for the unlabelled test set.

The target distribution is imbalanced, meaning some classes have substantially more examples than others. A model can achieve high accuracy by predicting common classes well while performing poorly on rare classes. For this reason, Macro F1 is the main evaluation metric.
Dataset
The notebook loads the competition files from Kaggle:
File	Purpose
train.csv	Training examples with input features and the target label
test.csv	Examples for which labels must be predicted
Sample.csv	Sample submission template


The notebook records these dataset dimensions:
Dataset	Shape	Meaning
Training set	(198000, 15)	198,000 rows and 15 columns, including the target
Test set	(102000, 14)	102,000 rows and 14 input columns


The data files are supplied by the Kaggle competition and are not included in this repository. The notebook uses Kaggle input paths, so update the paths if running in another environment.
Feature groups used
- Text: comment, transformed into clean_comment.
- Engagement and numeric signals: upvote, downvote, emoticon_1, emoticon_2, emoticon_3, if_1, and if_2.
- Categorical metadata: race, religion, gender, and disability.
- Target: label, available only in the training data.
Workflow
The notebook follows these stages:
1. Load the training, test, and sample-submission files.
2. Inspect dimensions, column types, descriptive statistics, duplicates, and missing values.
3. Explore label balance, comment length, emoticon usage, and numeric-feature correlations.
4. Clean missing or malformed values and normalize comment text.
5. Engineer features that capture engagement, comment structure, and text characteristics.
6. Split the training data into training and validation subsets using an 80:20 split, stratified by label.
7. Transform text and structured columns into model-ready numerical features.
8. Tune and train candidate classifiers.
9. Compare models using validation Macro F1 and inspect classification reports.
10. Combine model probabilities with weighted soft voting.
11. Select the best validation model, predict test labels, and save submission.csv.
Exploratory Data Analysis
EDA was used to understand the dataset and motivate downstream modelling choices.
1. Label distribution
The target labels are imbalanced. Class 0 has the most examples, while class 3 has comparatively few. This creates a risk that a model will favour majority classes.

Decision: evaluate with Macro F1 and use class-balancing options in selected classifiers rather than relying on accuracy alone.
2. Comment length by label
The notebook compares comment length across classes. Labels 1 and 2 tend to have longer comments, while labels 0 and 3 tend to have shorter comments. The distributions overlap, so length is treated as a supporting signal, not a standalone solution.
3. Emoticon usage
The mean counts of emoticon_1, emoticon_2, and emoticon_3 are compared across labels. The notebook observes some variation, but the differences are not large enough to treat emoticon counts as the primary predictor.
4. Numeric-feature correlations
A correlation heatmap is used to examine relationships among upvotes, downvotes, emoticon counts, and if_1/if_2. Most relationships are weak, with some association between upvotes and downvotes.

Why this matters: correlation analysis provides context about the structured data, while the modelling pipeline retains multiple signals because weak pairwise correlation does not necessarily mean a feature has no predictive value.
Data Preprocessing
Missing values and type consistency
The notebook applies the following cleaning steps to both training and test data:
- Converts numeric columns to numeric values with pd.to_numeric(..., errors='coerce').
- Replaces invalid or missing numeric values with 0.
- Replaces missing comments with an empty string.
- Fills missing categorical values with "none".
- Converts categorical values to strings, lowercases them, and strips surrounding whitespace.
These steps ensure that the training and test data use consistent representations.
Text cleaning
The clean_text() function applies the following operations:
1. Converts text to lowercase.
2. Removes URLs beginning with http or www.
3. Reduces repeated character sequences longer than two characters to two repeated characters. For example, loooove becomes loove.
4. Removes punctuation and special characters while retaining letters, digits, and whitespace.
5. Collapses repeated whitespace and trims leading or trailing spaces.
The cleaned text is stored in clean_comment. The original comment is retained for features such as raw comment length and punctuation indicators.

Why clean text? It reduces superficial variation and gives the vectorizers a more consistent representation of the comments.
Feature Engineering
Feature engineering creates additional signals from existing columns. The notebook adds the following features.
Feature	Definition	Why it may help
net_votes	upvote - downvote	Represents the net balance of positive and negative votes
total_votes	upvote + downvote	Captures total engagement
vote_ratio	upvote / (downvote + 1)	Represents the balance between upvotes and downvotes while avoiding division by zero
comment_length	Number of characters in the original comment	Captures comment length
word_count	Number of whitespace-separated words in the original comment	Captures comment size in word units
log_upvote	log1p(upvote)	Compresses the scale of highly skewed upvote counts
log_downvote	log1p(downvote)	Compresses the scale of highly skewed downvote counts
upvote_rate	upvote / (total_votes + 1)	Captures the proportion of positive engagement, with a stabilizing denominator
controversy_score	min(upvote, downvote) / (max(upvote, downvote) + 1)	Higher values indicate a more balanced mix of upvotes and downvotes
has_question	1 if the raw comment contains ?, otherwise 0	Captures question-style punctuation
has_exclamation	1 if the raw comment contains !, otherwise 0	Captures exclamatory punctuation
caps_ratio	Uppercase-character count divided by len(comment) + 1	Captures emphasis through capitalization
avg_word_len	Average word length in the cleaned comment	Captures a coarse characteristic of writing style
unique_word_ratio	Unique-word count divided by total-word count	Captures lexical repetition versus variety


log1p(x) means log(1 + x). It is useful for skewed non-negative counts because it compresses very large values while preserving zero as zero.

The engineered features are combined with the original numeric columns. These are hypotheses about useful predictive signals, not proof that any one feature causes a particular label.
Feature Transformation
The notebook uses a ColumnTransformer to process different feature types appropriately.
1. Word-level TF-IDF
TF-IDF stands for Term Frequency–Inverse Document Frequency. It gives higher weight to terms that are informative for a document relative to the collection, while down-weighting terms that occur very frequently across documents.

Configuration in the notebook:
- max_features=20000
- ngram_range=(1, 2), including individual words (unigrams) and two-word phrases (bigrams)
- stop_words='english'
- min_df=2, ignoring terms appearing in fewer than two documents
- sublinear_tf=True, applying logarithmic scaling to term frequency
- dtype=np.float32, reducing feature-storage memory compared with 64-bit floats
Why: word n-grams capture vocabulary and short phrases that can signal the comment's category.
2. Character-level TF-IDF
A second TF-IDF vectorizer uses:
- analyzer='char_wb'
- ngram_range=(3, 5)
- max_features=8000
- min_df=2
- sublinear_tf=True
Character n-grams represent short sequences of characters within word boundaries. They can capture spelling variations, word fragments, and morphology that exact word features may miss.

Why use both word and character features? Word features represent lexical meaning at the token level, while character features can provide robustness to spelling and word-form variation.
3. Numeric scaling
StandardScaler standardizes the numeric feature columns. Scaling is particularly useful for linear models such as Logistic Regression, where features with very different numeric ranges can otherwise influence optimization unevenly.
4. Categorical encoding
OneHotEncoder(handle_unknown='ignore') converts categorical fields into numerical indicator columns. The handle_unknown='ignore' option allows the test data to contain a category that was not present during fitting without causing an encoding error.
5. Combined representation
The ColumnTransformer joins the word TF-IDF, character TF-IDF, scaled numeric features, and one-hot encoded categorical features into a single model input matrix.
Models and Training
The notebook compares three classifiers and a weighted ensemble.
1. Logistic Regression
Logistic Regression is a supervised classification algorithm that learns a linear decision boundary in the transformed feature space. In multiclass classification, it can estimate probabilities for each class.

Why it is useful here:
- Works well with high-dimensional, sparse TF-IDF features.
- Provides a strong and efficient baseline for text classification.
- Produces class probabilities that can be used by the ensemble.
The notebook uses the saga solver, a high iteration limit, a fixed random seed, and class_weight='balanced' in the final manually specified model to account for the imbalanced target.

Tuning note: the notebook records a GridSearchCV result with C=10 and a validation Macro F1 of 0.7960 for that experiment. The later explicitly instantiated log_model uses C=4. These are separate configurations in the notebook and should not be treated as the same run.
2. XGBoost
XGBoost is a gradient-boosted decision-tree library. It builds trees sequentially, with later trees improving the model based on errors made by earlier trees.

Why it is included:
- Learns nonlinear relationships and feature interactions.
- Can complement the linear patterns learned by Logistic Regression.
- Supports regularization and row/feature subsampling to help control overfitting.
The explicitly instantiated model uses settings including:
- n_estimators=700
- learning_rate=0.05
- max_depth=4
- subsample=0.85
- colsample_bytree=0.7
- tree_method='hist'
- eval_metric='mlogloss'
- early_stopping_rounds=40
The notebook also records a GridSearchCV experiment with n_estimators=700, max_depth=4, and learning_rate=0.05, with a best cross-validation score of approximately 0.7865 and validation Macro F1 of 0.7909.
3. LightGBM
LightGBM is a gradient-boosting framework designed for efficient tree-based learning. It grows trees leaf-wise, choosing the leaf that gives the largest loss reduction. This can be effective on complex tabular patterns, but tree complexity needs to be controlled to reduce overfitting.

Why it is included:
- Captures nonlinear relationships between engagement, metadata, and engineered features.
- Efficiently handles many structured features.
- Provides a different modelling approach from linear Logistic Regression.
The explicitly instantiated model uses settings including:
- n_estimators=700
- learning_rate=0.05
- num_leaves=63
- subsample=0.85
- colsample_bytree=0.7
- reg_alpha=0.05
- reg_lambda=0.1
- min_child_samples=20
- class_weight='balanced'
The notebook records a hyperparameter-search result with num_leaves=63, n_estimators=300, and learning_rate=0.1, a best cross-validation score of approximately 0.8163, and validation Macro F1 of 0.8247. The later manually instantiated model uses a different estimator count and learning rate, so keep this distinction in mind when reproducing the result.
4. Hyperparameter search
The notebook imports both GridSearchCV and RandomizedSearchCV and records search results for candidate configurations.
- GridSearchCV evaluates every parameter combination in a specified grid. It is systematic but can become computationally expensive when the grid is large.
- RandomizedSearchCV evaluates a fixed number of sampled parameter combinations. It can explore a broader search space with a lower computational budget, but it does not guarantee finding the globally best combination.
Hyperparameter tuning helps choose model settings based on cross-validation performance rather than relying entirely on defaults.
5. Weighted soft-voting ensemble
Soft voting combines the predicted class probabilities from several models instead of simply taking a majority vote of their predicted labels.

The notebook combines probabilities from Logistic Regression, XGBoost, and LightGBM:
\[
P_{\text{ensemble}} =
w_{\text{LR}}P_{\text{LR}} +
w_{\text{XGB}}P_{\text{XGB}} +
w_{\text{LGB}}P_{\text{LGB}}
\]
The weights are initially made proportional to each model's validation Macro F1 score. The notebook then adds a small 0.05 boost to the LightGBM weight and renormalizes all weights so that they sum to 1. The class with the highest resulting probability is selected.

Why ensemble? The models can learn different patterns. Combining their probabilities may improve robustness, although an ensemble is not guaranteed to outperform the strongest individual model and must be validated empirically.
Evaluation Metric
Macro F1 score
The notebook uses f1_score(..., average='macro').

For each class, F1 is the harmonic mean of precision and recall:
\[
F1 = 2 \times \frac{\text{Precision}\times\text{Recall}}
{\text{Precision}+\text{Recall}}
\]
Macro F1 calculates F1 separately for every class and then takes the unweighted average:
\[
F1_{\text{macro}} = \frac{1}{K}\sum_{k=1}^{K}F1_k
\]
where \(K\) is the number of classes.
- Precision: of the examples predicted as a class, how many actually belong to it?
- Recall: of the examples belonging to a class, how many did the model identify?
- F1: balances precision and recall.
- Macro averaging: gives each class equal weight, including minority classes.
Accuracy is also printed in the classification reports, but it can hide poor performance on rare labels when the target distribution is imbalanced. Macro F1 is therefore the more informative headline metric for this project.
Validation Results
The notebook's saved outputs contain these validation results:
Model / experiment	Validation Macro F1	Notes
Logistic Regression search experiment	0.7960	Recorded search result used C=10
XGBoost search experiment	0.7909	Recorded search result used 700 estimators, depth 4, learning rate 0.05
LightGBM search experiment	0.8247	Highest explicitly recorded individual-model validation score
Weighted soft-voting ensemble	Check notebook output	The final ensemble score is calculated at runtime; confirm the printed value when reproducing


The recorded LightGBM classification report shows:
Class	Precision	Recall	F1-score
0	0.9823	0.9457	0.9637
1	0.7297	0.8439	0.7826
2	0.8801	0.8977	0.8888
3	0.6355	0.6947	0.6638
Macro average	0.8069	0.8455	0.8247


The validation split contains 39,600 examples, corresponding to 20% of the 198,000-row training dataset.

Important distinction: 0.8247 is the recorded validation Macro F1 for LightGBM in the notebook. The approximately 0.82 Kaggle score is the project score reported by the author. Validation performance and a competition leaderboard score are related but are not necessarily identical.
Generating the Submission
After comparing validation scores, the notebook selects the best model by Macro F1. It then predicts labels for the test data. If the ensemble is selected, it combines test-set probabilities using the same learned weights; otherwise, it uses the selected individual classifier.

Finally, it copies the sample-submission structure, fills its label column with predictions, and writes:
submission.csv
The generated file is intended to be uploaded to the Kaggle competition submission interface. The notebook contains a section describing final training on the full training data, but when reproducing the workflow, verify that the models and preprocessor are actually refitted on the combined training and validation data before final test prediction.
Tech Stack
- Python for the modelling workflow
- Pandas and NumPy for data manipulation and numerical operations
- Matplotlib and Seaborn for visual analysis
- scikit-learn for preprocessing, TF-IDF, splitting, scaling, encoding, Logistic Regression, hyperparameter search, and evaluation
- XGBoost for gradient-boosted trees
- LightGBM for efficient gradient boosting
- Kaggle Notebooks for the competition environment and data access
How to Run
Option A: Run on Kaggle
1. Open the relevant Kaggle competition or notebook environment.
2. Attach the competition dataset containing train.csv, test.csv, and Sample.csv.
3. Open the notebook (.ipynb) and run the cells in order.
4. Ensure the required libraries are available in the notebook environment.
5. Review the validation reports and selected model.
6. Confirm that submission.csv is generated and that its columns match the sample submission.
The notebook currently uses these Kaggle paths:
train = pd.read_csv('/kaggle/input/comment-category-prediction-challenge/train.csv')
test = pd.read_csv('/kaggle/input/comment-category-prediction-challenge/test.csv')
sample_submission = pd.read_csv('/kaggle/input/comment-category-prediction-challenge/Sample.csv')
If the competition dataset is mounted under a different directory, update the paths accordingly.
Option B: Run locally
1. Clone or download this repository.
2. Create and activate a Python virtual environment.
3. Install the libraries used by the notebook.
4. Download the competition data through Kaggle, subject to the competition's rules, and place it in a local data directory.
5. Update the three CSV paths in the notebook.
6. Run the notebook from top to bottom.
Example installation command:
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm jupyter
Reproducibility note: the notebook uses SEED = 42 for the train-validation split and model randomness where supported. Exact scores may still vary with library versions, model configuration, and execution environment. Some model-training and tuning cells are computationally intensive.
Key Learnings
Through this project, I practised:
- Performing structured EDA before model development.
- Identifying class imbalance and selecting a suitable evaluation metric.
- Cleaning text for an NLP classification pipeline.
- Combining word-level and character-level TF-IDF features.
- Engineering features from votes, text length, punctuation, capitalization, and lexical diversity.
- Using ColumnTransformer, StandardScaler, and OneHotEncoder for mixed data types.
- Training and comparing linear and gradient-boosting classifiers.
- Exploring hyperparameter tuning with GridSearchCV and RandomizedSearchCV.
- Combining model probabilities through a weighted soft-voting ensemble.
- Producing a competition-ready CSV submission.
Limitations and Future Improvements
Potential next steps include:
1. Make final refitting explicit: refit the selected model or all ensemble members on the complete labelled dataset before generating test predictions.
2. Use a consistent tuning workflow: ensure the exact hyperparameters reported by search are the ones used in the final model, or document why they were changed.
3. Avoid validation-weight overfitting: calculate ensemble weights using cross-validation or out-of-fold predictions instead of relying only on one validation split.
4. Inspect minority-class errors: review the confusion matrix and per-class precision/recall, especially for class 3.
5. Evaluate feature groups: compare text-only, structured-only, and combined-feature models to understand which signals add value.
6. Track experiments systematically: record the exact parameters, random seed, library versions, validation score, and Kaggle score for every run.
7. Check for data leakage: ensure that transformations are fitted only on training data during validation, and that test labels are never used.
Project Context
This project was completed as an academic machine learning project using a Kaggle competition workflow. It demonstrates an end-to-end approach to multiclass text classification, from raw data inspection through feature engineering and model evaluation to test prediction and submission creation.
Note: Dataset dimensions and metric values in this README are based on the uploaded notebook's recorded notes and outputs. Re-run the notebook to verify results after changing parameters or library versions.
