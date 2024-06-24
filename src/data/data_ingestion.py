import numpy as np
import pandas as pd
import yaml
import os
from sklearn.model_selection import train_test_split

#import logging
import logging

logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

#console & File handler
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler =logging.FileHandler('errors.log')
file_handler.setLevel('ERROR')

formatter =  logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(messages)s')

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> float:
    try:
        with open(params_path, 'r') as file:
            test_size = yaml.safe_load(file)['data_ingestion']['test_size']
            logger.debug('test size retrived')
        return test_size
    except FileNotFoundError:
        logger.error('File not found')
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error')
        raise
    except KeyError as e:
        logger.error('Some error occured')
        raise

def read_data(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
        return df
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty.")
        raise
    except pd.errors.ParserError:
        print("Error: The CSV file contains parsing errors.")
        raise
    except Exception as e:
        print(f"Unexpected error while reading data: {e}")
        raise

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.drop(columns=['tweet_id'], inplace=True)
        final_df = df[df['sentiment'].isin(['neutral', 'sadness'])]
        final_df['sentiment'].replace({'neutral': 1, 'sadness': 0}, inplace=True)
        return final_df
    except KeyError as e:
        print(f"Error: Missing column in DataFrame: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error while processing data: {e}")
        raise

def save_data(data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    try:
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(data_path, 'test.csv'), index=False)
    except OSError as e:
        print(f"Error creating directory {data_path}: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error while saving data: {e}")
        raise

def main():
    try:
        test_size = load_params('params.yaml')

        df = read_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')

        final_df = process_data(df)

        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)

        data_path = os.path.join('data', 'raw')

        save_data(data_path, train_data, test_data)
    except Exception as e:
        print(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()
