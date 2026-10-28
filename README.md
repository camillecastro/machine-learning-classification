# Machine Learning Classification & Model Evaluation

### Supervised Learning | Model Comparison | Python

**Computer Engineering | Marian University**

## Project Overview

This project explores supervised machine learning through the implementation and evaluation of three classification algorithms: Logistic Regression, Decision Tree, and Random Forest.

Inspired by concepts studied in my Machine Learning coursework, the project investigates how different algorithms perform on the same dataset and compares their predictive performance using accuracy, precision, recall, F1 score, and confusion matrices.

The objective is to connect machine learning theory with practical Python implementation, reproducible experimentation, and model evaluation.

## Project Objectives

- Implement multiple supervised classification algorithms.
- Explore data preprocessing and feature scaling.
- Divide data into training and testing sets.
- Evaluate models using multiple classification metrics.
- Apply cross-validation to estimate generalization performance.
- Visualize predictions using confusion matrices.
- Compare the strengths and limitations of different models.

## Dataset

**Breast Cancer Wisconsin Diagnostic Dataset**

The dataset is publicly available through scikit-learn and contains measurements derived from digitized images of breast cell samples.

Dataset characteristics:

- 569 observations
- 30 numerical input features
- Two classification categories: malignant and benign
- No personal patient identifiers included in the dataset

The dataset is loaded directly using:

`sklearn.datasets.load_breast_cancer()`

This project is intended for educational machine learning analysis only and is not a clinical diagnostic tool.

## Machine Learning Models

### 1. Logistic Regression

Logistic Regression is a supervised learning algorithm that estimates the probability of a binary classification outcome.

In this project, Logistic Regression is implemented with standardized features using scikit-learn's pipeline functionality.

### 2. Decision Tree

Decision Trees classify observations by recursively dividing data according to feature values.

The model uses a maximum tree depth of four to limit complexity and reduce the risk of overfitting.

### 3. Random Forest

Random Forest combines multiple decision trees to generate predictions using an ensemble learning approach.

The implementation uses 150 trees with a fixed random seed to support reproducibility.

## Technologies Used

**Programming Language**
- Python

**Libraries**
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

**Machine Learning Concepts**
- Supervised Learning
- Binary Classification
- Feature Scaling
- Train/Test Splitting
- Cross-Validation
- Model Evaluation
- Ensemble Learning

## Methodology

### Step 1: Data Loading

The Breast Cancer Wisconsin Diagnostic dataset is loaded using scikit-learn.

Input features and classification labels are separated.

### Step 2: Data Splitting

The dataset is divided into:

- 80% training data
- 20% testing data

Stratified splitting is used to preserve the approximate class distribution in both sets.

### Step 3: Data Preprocessing

StandardScaler is applied to the Logistic Regression model through a pipeline.

The scaler is fitted using training data, helping prevent information leakage from the test set.

### Step 4: Model Training

Three classification algorithms are trained on the same training dataset.

A fixed random state supports reproducible comparisons.

### Step 5: Cross-Validation

Five-fold stratified cross-validation is performed on the training data.

The average F1 score is calculated for each model to provide an additional estimate of classification performance.

### Step 6: Model Evaluation

The trained models generate predictions on the held-out test set.

Performance is measured using accuracy, precision, recall, and F1 score.

## Evaluation Metrics

### Accuracy

The proportion of correctly classified observations.

$$
Accuracy = \frac{TP+TN}{TP+TN+FP+FN}
$$

### Precision

The proportion of predicted positive cases that are actually positive.

$$
Precision = \frac{TP}{TP+FP}
$$

### Recall

The proportion of actual positive cases correctly identified.

$$
Recall = \frac{TP}{TP+FN}
$$

### F1 Score

The harmonic mean of precision and recall.

$$
F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision+Recall}
$$

**Important:** In this implementation, scikit-learn uses the benign class as the positive class (label 1). Therefore, precision, recall, and F1 are calculated for benign classification. Evaluating malignant-case recall separately would be important for any clinically oriented analysis.

## Model Evaluation Results

The Python program generates a CSV file containing the evaluation metrics for all three models.

The results are saved to:

`results/model_metrics.csv`

The program also generates confusion matrices for visual comparison.

### Confusion Matrix Visualization

![Classification Confusion Matrices](results/confusion_matrices.png)

*Figure 1: Confusion matrices for Logistic Regression, Decision Tree, and Random Forest.*

The image will appear after the simulation has been executed and the generated PNG has been uploaded to this repository.

## Engineering Analysis

### Logistic Regression

Logistic Regression provides a relatively interpretable baseline for binary classification.

Feature scaling supports the optimization process, particularly when input measurements have different numerical ranges.

### Decision Tree

Decision Trees provide interpretable decision rules and can capture nonlinear relationships.

However, their performance may depend strongly on tree depth and the characteristics of the training data.

### Random Forest

Random Forest combines predictions from multiple decision trees.

Ensemble learning can improve predictive stability and reduce the variance associated with individual trees.

### Model Comparison

No single algorithm is universally superior.

Model selection should consider predictive performance, interpretability, computational requirements, and the consequences of classification errors.

For this educational dataset, the project compares models using the same data split and evaluation metrics.

## Running the Project

### Requirements

- Python 3.9 or newer
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

### Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

### Execution

Run the Python program:

```bash
python main.py
```

The program will:

1. Load the dataset.
2. Split the observations into training and testing sets.
3. Train and evaluate three classification models.
4. Perform cross-validation.
5. Print model performance metrics.
6. Save the results to a CSV file.
7. Generate confusion matrix visualizations.

## Project Structure

```text
machine-learning-classification/
|
|-- main.py
|-- requirements.txt
|-- README.md
|
|-- results/
    |-- confusion_matrices.png
    |-- model_metrics.csv
```

The `results` directory is automatically created when the program runs.

## Limitations

- The project uses a relatively small educational dataset.
- Results may vary with data splitting and model configuration.
- No hyperparameter optimization is currently implemented.
- Model performance on this dataset does not establish clinical reliability.
- The classification metrics use the benign class as the positive class by default.
- Additional external validation would be necessary for real-world applications.

## Future Development

- [x] Implement Logistic Regression.
- [x] Implement Decision Tree classification.
- [x] Implement Random Forest classification.
- [x] Add stratified training and testing splits.
- [x] Implement cross-validation.
- [x] Calculate classification performance metrics.
- [x] Generate confusion matrix visualization code.
- [ ] Perform hyperparameter tuning.
- [ ] Investigate feature importance.
- [ ] Compare ROC curves and AUC scores.
- [ ] Evaluate class-specific error tradeoffs.
- [ ] Experiment with additional datasets.

## Academic Context

This project was developed as an educational portfolio example inspired by my Machine Learning coursework in Computer Engineering at Marian University.

It demonstrates foundational supervised learning concepts, Python programming, model evaluation, and reproducible experimental methodology.
