# main.py

import pandas as pd
from src.data_preparation import load_data

def main():
    # Load and clean data
    train_data = load_data('data/TrainingDatasets/TREC-2017-LiveQA-Medical-Train-1.xml')
    
    # Exploratory data analysis
    print(train_data[["subject", "focus", "type", "answer"]].head(15))
    print(train_data[["subject", "message", "focus", "type", "answer"]].head(15))


if __name__ == "__main__":
    main()
