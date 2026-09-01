# Scripts

# data scripts
 - download_data.py : downloads the dataset from kaggle : https://www.kaggle.com/datasets/blastchar/telco-customer-churn

 - make_splits.py : read the telcochurn csv and split the original data into training and testing data with a ratio of 8:2 and stratification, then save into train and test data set in 'data/processed'

 - determinisit_cleaning.py : read the train.csv and test.csv and drop the customer ID column and change the data type of the Total Charge into float. returning cleaned train and test data frame.

 - train.py : import the load_and_clean() (reading the train.csv -> splited train_df and test_df) function from deterministic_cleaning.py, 1: create the preprocessor (cleaning numeric and categorical columns), 2: assemble the preprocessor and logistic regression training, 3: train the base model and evaluate

 