# main.py

import pandas as pd
from src.data_preparation import load_data

def main():
    # Load and clean data
    data_paths = ['data/TrainingDatasets/TREC-2017-LiveQA-Medical-Train-1.xml', 'data/TrainingDatasets/TREC-2017-LiveQA-Medical-Train-2.xml']
    train_data = load_data(data_paths)
    
    # Exploratory data analysis
    print(train_data[["question_id","focus", "answer"]].head(45))

if __name__ == "__main__":
    main()
