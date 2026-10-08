Project Overview
    This project builds an end-to-end machine learning workflow to identify customers at risk of churn. The objective is to help a retention team prioritize customers for proactive intervention before they leave.
    The project applies reproducible data cleaning, model training, hyperparameter tuning, threshold tuning, experiment tracking, artifact logging, and model-selection documentation using MLflow.


Business Objective
    The business goal is to identify as many true churners as possible.
    Recall is prioritized because a false negative means a customer who will churn is not identified and may leave without receiving a retention intervention. 


Models to be tested:
    - linear model: logistic regression
    - bagged trees: random forest 
    - boosted trees: gradient boosting 


Data Pipeline

The dataset is cleaned using a deterministic cleaning script: "scripts/data/deterministic_cleaning.py"

The training scripts split the cleaned data into:
    - Training data: used for model fitting, hyperparameter tuning, and cross-validated threshold selection.
    - Test data: used to evaluate the finalized models.

Numeric columns are processed with median imputation and standard scaling. Categorical columns are processed with most-frequent-value imputation and one-hot encoding. All preprocessing is included in a scikit-learn pipeline to ensure transformations are learned only from training folds during cross-validation.


Models Evaluated

Two classifiers were evaluated:
    - Gradient Boosting Classifier
    - Random Forest Classifier

For each model, the workflow performs:
1. Hyperparameter tuning with `RandomizedSearchCV`.
2. Five-fold stratified cross-validation.
3. Selection based on churn recall.
4. Threshold tuning using out-of-fold churn probabilities.
5. Final fitting on the full training set.
6. One final evaluation on the held-out test set.
7. Logging of metrics, parameters, artifacts, model lineage, and the serialized pipeline to MLflow.


metrics to be logged (according to priority):
    - PR-AUC: 
    - Churn recall:
    - Churn precision
    - Churn F1
    - ROC-AUC

from threshold tuning:
    - the gradient boosting searched the threshold 0.25, it provides 0.8048 test set churn recall
    - the random forest searched the threshold 0.50, it provides 0.7834 test set churn recall


    | Model             | Threshold | Test recall | Test precision | ROC-AUC | PR-AUC |
    | ----------------- | --------- | ----------- | -------------- | ------- | ------ |
    | Gradient Boosting | 0.25      | 0.8048      | 0.5059         | 0.8466  | 0.6668 |
    | Random Forest     | 0.50      | 0.7834      | 0.5026         | 0.8379  | 0.6355 |

    - Gradient Boosting appears to be a better model after the threshold tuning, it translate to the following operational policy:
        If predicted churn probability >= 0.25:
            send customer to retention intervention workflow
        Else:
            do not intervene

