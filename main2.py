import pandas as pd
from single import CLEAN1

def main():
    try:
        file = pd.read_csv('single.csv')  
        print("Input file loaded successfully.")
        print(file.head())  
    except FileNotFoundError:
        print("Error: 'single.csv' not found.")
        return

    cleaner = CLEAN1()
    
    try:
        cleaned_data = cleaner.clean_data(file)
        print("Cleaned DataFrame:")
        print(cleaned_data.head())  
        
        cleaned_data.to_csv('cleaned_data.csv', index=False)
        print("Cleaned data saved to 'cleaned_data.csv'.")
    except Exception as e:
        print(f"Error during data cleaning: {e}")

if __name__ == "__main__":
    main()