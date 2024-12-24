import pandas as pd
from cleaning import CLEAN  
def main():
    print("Loading dataset...")
    try:
        df = pd.read_csv("test.csv")  
        print("Dataset loaded successfully.")
    except FileNotFoundError:
        print("Dataset file not found. Ensure the file is in the correct path.")
        return

    cleaner = CLEAN()
    cleaned_data = cleaner.clean_data(df)

    print("Sample of cleaned data:")
    print(cleaned_data.head())

if __name__ == "__main__":
    main()