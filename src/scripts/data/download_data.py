from pathlib import Path
import shutil
import kagglehub

def download_data():
    # download to default kaggle cache dir
    cache = kagglehub.dataset_download("blastchar/telco-customer-churn")

    # define the desired destionation folder
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    DATA_DIR = PROJECT_ROOT / "data"

    # move with shutil
    shutil.copytree(cache, DATA_DIR, dirs_exist_ok=True)
    print(f'Dataset files in {DATA_DIR}')

if __name__ == '__main__':
    download_data()
