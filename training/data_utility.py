import os
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split
import seaborn as sns
import logging
import logging.config

def download_penguins():
    """Download the penguins dataset and store it in a directory called
    data"""
    # Check for existing data directory
    logging.info(msg="Checking for existing data directory...")
    if not os.path.isdir("/data"):
        logging.info(msg="Downloading dataset..")

        # Download and save penguins dataset csv
        penguins = sns.load_dataset('penguins')

        # Create directory and file path for dataset
        logging.info(msg="Creating directory...")
        data_dir = 'data'
        file_path = os.path.join(data_dir, 'penguins.csv')
        os.makedirs(data_dir, exist_ok=True)

        # Save data to directory
        logging.info(msg="Saving csv to directory")
        penguins.to_csv(file_path, index=False)
    else:
        logging.info(msg="Data already downloaded")


def load_data(directory, filename):
    """
    Loads a file from a given directory and returns its content as a string.
    """
    file_path = os.path.join(directory, filename)

    if os.path.isfile(file_path):
        with open(file_path, "r") as file:
            data = file.read()
        return data
    else:
        print(f"File '{filename}' not found in directory '{directory}'.")
        return None

def split_data():
    """Create features and labels of train and test datasets"""

    # Create a data from dataset
    logging.info(msg="Creating DataFrame from csv..")
    penguins = pd.read_csv('data/penguins.csv')

    # Create training and test splits
    penguins = penguins.dropna(subset=['species'])
    logging.info(msg="Creating features")
    X = penguins.drop('species', axis=1)
    logging.info(msg="Creating target")
    y = penguins['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    logging.info(msg="Test and train splits created")

    return X_train, X_test, y_train, y_test

def configure_logging():
    """Configure logging handlers and return a logger instance."""
    if Path("logging.conf").exists():
        logging.config.fileConfig("logging.conf")
    else:
        logging.basicConfig(
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
            level=logging.INFO,
        )