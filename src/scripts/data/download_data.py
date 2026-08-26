from pathlib import Path
import shutil
import kagglehub

DATASET_HANDLE = "blastchar/telco-customer-churn"
SOURCE_FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
DESTINATION_FILENAME = "telco_customer_churn.csv"

def download_data():
    # download to default kaggle cache dir
    cache_dir = Path(kagglehub.dataset_download(DATASET_HANDLE))

    # define the desired destionation folder
    project_root = Path(__file__).resolve().parents[3]
    data_dir = project_root / "data" / "raw"
    data_dir.mkdir(parents = True, exist_ok = True)

    source_path = cache_dir / SOURCE_FILENAME

    destionation_path = data_dir / DESTINATION_FILENAME

    if not source_path.exists():
        raise FileNotFoundError(
            f'Expected source file was not found: {source_path}'
        )

    # copy csv and rename it in one opeartion
    shutil.copy2(source_path, destionation_path)

    print(f'Downloaded from: {source_path}')
    print(f'Saved as: {destionation_path}')
    
if __name__ == '__main__':
    download_data()
