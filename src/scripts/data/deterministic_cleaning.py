from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / 'data'
PROCESS_DIR = DATA_DIR / 'processed'

TRAIN_PATH = PROCESS_DIR / 'train.csv'
TEST_PATH = PROCESS_DIR / 'test.csv'

def determinisitc_cleaning(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df['TotalCharges'] = pd.to_numeric(
        df['TotalCharges'],
        errors = 'coerce'
    )

    df = df.drop(columns = ['customerID'])

    return df

def load_and_clean() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    train_df = determinisitc_cleaning(train_df)
    test_df = determinisitc_cleaning(test_df)

    return train_df, test_df

if __name__ == '__main__':
    train_df, test_df = load_and_clean()

    print(train_df.info())
    print("\nMissing values in training data:")
    print(train_df.isna().sum()[train_df.isna().sum() > 0])