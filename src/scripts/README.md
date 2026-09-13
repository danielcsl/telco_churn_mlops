# Scripts

# data scripts
 - download_data.py : downloads the dataset from kaggle : https://www.kaggle.com/datasets/blastchar/telco-customer-churn

 - make_splits.py : read the telcochurn csv and split the original data into training and testing data with a ratio of 8:2 and stratification, then save into train and test data set in 'data/processed'

 - determinisit_cleaning.py : read the train.csv and test.csv and drop the customer ID column and change the data type of the Total Charge into float. returning cleaned train and test data frame.

 - train_baseline.py : train a logistic regressoin model and log it with mflow
 - train_gradient_bosting.py : train a gradient boosting model and log it with mlflow
 - train_random_forest.py : train a random forest modeal and log it with mlflow

 # model determination and comparision
 - recall appears to be the most important 
 
 -> out of the 3 base model logged, the random forest appears to be the most competitive regarding the recall and F1 score, hence we proceeded to hyper-parameter tuning
    - hyper-parameter tuning on random forest was performed, the resulting best hyper-parameters are:

     ===== Best Cross-Validation Result =====
        Best CV churn recall: 0.7967
        Best parameters: {
            'model__n_estimators': 500,
            'model__min_samples_split': 10, 
            'model__min_samples_leaf': 2, 
            'model__max_features': 'log2', 
            'model__max_depth': 5
            }
yet the recall and F1 score didnt improve, hence the model was not promoted to champion

-> Also tried to tune hyperparameters on gradient boosting model
    ===== Best Cross-Validation Result =====
    Best CV churn recall: 0.5271
    Best parameters: {
        'model__subsample': 0.8, 
        'model__n_estimators': 500, 
        'model__min_samples_split': 2, 
        'model__min_samples_leaf': 1, 
        'model__max_features': 'log2', 
        'model__max_depth': 3, 
        'model__learning_rate': 0.03
        }
    yet the recall and F1 score didnt improved better than base logistic regression model, hence not promoted to champion



 