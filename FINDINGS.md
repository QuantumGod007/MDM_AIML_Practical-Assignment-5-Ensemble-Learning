# Findings — Practical Assignment 5

## 1. Aim and data
I compared three individual classifiers with three ensemble methods. The **Breast Cancer Wisconsin (Diagnostic)** dataset contains 569 rows and 30 numerical input features. It is built into scikit-learn. I used a **stratified 80/20 split** (455 training, 114 test cases), random seed 42. There are 42 malignant cases in the test set.

For the evaluation, the positive class is **malignant (1)**; the original scikit-learn label is remapped to make recall easier to interpret. A StandardScaler is fitted *inside* the Logistic Regression and KNN pipelines, using training data only. I also ran five-fold stratified cross-validation on the training set. The test set was held out until the final evaluation.

## 2. Models and reasons for choosing them
- **Logistic Regression:** simple linear baseline; works well when the features give a fairly clear separation.
- **Decision Tree:** captures nonlinear decisions and is easy to inspect; can overfit.
- **KNN:** a different, distance-based approach, so it adds model diversity. Scaling matters here.
- **Voting:** three base models vote on a class, and the majority wins. No extra model is trained to combine them.
- **Stacking:** the three base models produce cross-validated predictions, which train a Logistic Regression meta-classifier. It can learn when to trust different models.
- **AdaBoost:** combines 100 shallow decision trees (stumps). It puts more weight on examples previous stumps classified incorrectly.

## 3. Test-set results

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.965 | 0.975 | 0.929 | 0.951 | 0.996 |
| Decision Tree | 0.904 | 0.943 | 0.786 | 0.857 | 0.947 |
| KNN | 0.956 | 0.974 | 0.905 | 0.938 | 0.982 |
| Voting (Hard) | 0.965 | 1.000 | 0.905 | 0.950 | — |
| Stacking | 0.974 | 1.000 | 0.929 | 0.963 | 0.995 |
| AdaBoost | 0.974 | 1.000 | 0.929 | 0.963 | 0.986 |

These figures were produced by running `python main.py`; the exact values are also saved in `results/model_scores.csv`. ROC-AUC is left as N/A for **hard voting** because it outputs class labels rather than continuous probabilities. The ROC chart therefore includes the other five models.

## 4. What I noticed
**Stacking and AdaBoost tied on this particular test split** with an F1-score of 0.963. Each detected 39 of the 42 malignant cases and missed 3. Hard voting detected 38 of 42 and missed 4. The plain decision tree was weakest on malignant recall (33 of 42), although its structure is easier to explain.

The improvements are not huge. For example, Logistic Regression was already strong, and stacking only increased test accuracy by one correct prediction (from 110 to 111 out of 114). This is a good reminder that a more complex ensemble is not automatically worth it.

Cross-validation gives an additional view: stacking had the highest mean training-fold F1 (about 0.973), while AdaBoost scored about 0.961. Those are **validation** results, not a second independent test set, so I would not claim one model is always best.

## 5. When I would use each method
- **Voting:** when I have several reasonably strong and different classifiers, and I want a simple way to combine them without training a second-level model.
- **Stacking:** when base models make different mistakes and there is enough data to train the meta-classifier without overfitting. It costs more training time.
- **AdaBoost:** when simple weak learners can be improved step by step, especially on structured numerical data. Noisy or mislabeled cases may receive too much attention.

## 6. Improvements and responsible use
Next, I would try repeated stratified cross-validation, hyperparameter tuning **inside** cross-validation, probability calibration, and threshold selection to reduce false negatives. I would also test on an independent dataset before trusting the results. The class counts are unequal, so looking only at accuracy would hide important errors.

This is a medical-themed public learning dataset, not a clinical validation study. The models could make harmful mistakes if used for real diagnosis. Generalization, data quality, possible demographic bias and the impact of false negatives would need much more testing. For the assignment, the goal is to understand ensemble methods and evaluate them fairly.