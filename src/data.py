import pandas as pd
from pathlib import Path

def load_data(file_path):
    """
    Load data from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    BASE_DIR = Path(__file__).parent.parent
    file_path = BASE_DIR / file_path

    df = pd.read_csv(file_path)

    return df