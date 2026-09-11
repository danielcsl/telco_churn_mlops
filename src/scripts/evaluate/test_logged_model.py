from pathlib import Path
import sys

import mlflow
import mlflow.sklearn

# Find the root directory of the project.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Allow this script to import files from src/.
sys.path.append(str(PROJECT_ROOT / "src"))

#import the cleaning function used
from scripts.data.deterministic_cleaning import load_and_clean

# Connect to the same MLflow database as train.py.
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH.as_posix()}")

MODEL_URI = "models:/m-42ef0943b09946d18c3b7627a4655a19"

def main():

    # load the testing set with load_and_clean()
    _, test_df = load_and_clean()

    X_test = test_df.drop(columns=['Churn'])
    y_test = test_df['Churn']

    sample_feature = X_test.head(3)
    sample_result = y_test.head(3)

    # load the model from MLflow
    loaded_pipeline = mlflow.sklearn.load_model(MODEL_URI)

    # predict with the model
    predictions = loaded_pipeline.predict(sample_feature)
    probabilities = loaded_pipeline.predict_proba(sample_feature)

    class_names = list(loaded_pipeline.named_steps['model'].classes_)
    churn_class_index = class_names.index('Yes')
    churn_probabilities = probabilities[:,churn_class_index]

    # Display the validation results.
    print("\n===== Logged Model Validation =====")

    for row_number, (actual, prediction, probability) in enumerate(
        zip(sample_result, predictions, churn_probabilities),
        start=1,
    ):
        print(f"\nCustomer {row_number}")
        print(f"Actual Churn: {actual}")
        print(f"Predicted Churn: {prediction}")
        print(f"Predicted Probability of Churn: {probability:.4f}")


if __name__ == "__main__":
    main()