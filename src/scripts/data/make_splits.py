from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
FILE = DATA_DIR/"WA_Fn-UseC_-Telco-Customer-Churn.csv"

def make_splits(test_size = 0.2, random_state = 42):
    df = pd.read_csv(FILE)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['Churn'] # churn is imbalanced
    )

    processed_dir = DATA_DIR/"processed"
    processed_dir.mkdir(exist_ok=True)
    train_df.to_csv(processed_dir / "train.csv", index = False)
    test_df.to_csv(processed_dir / "test.csv", index = False)
    print(f'Train : {len(train_df)} rows, Test : {len(test_df)} rows')

if __name__ == "__main__":
    make_splits()