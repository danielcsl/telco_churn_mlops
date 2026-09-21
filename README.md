Models to be tested:
    - linear model: logistic regression
    - bagged trees: random forest 
    - boosted trees: gradient boosting 

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