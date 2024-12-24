import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

class CLEAN:

    def drop_loan_id(self, file):
        print("Checking for Loan_ID column to drop...")
        file = file.drop(columns=['Loan_ID'])
        return file

    def find_mode(self, file, col):
        """Finds the mode of a column."""
        print(f"Finding mode for column {col}...")
        return file[col].mode()[0]  

    def data_type_con(self, file):
        print("Converting data types...")
        file["Loan_Amount_Term"] = file["Loan_Amount_Term"].astype("str")
        file["Credit_History"].fillna(0.0, inplace=True)
        file["Credit_History"] = file["Credit_History"].apply(lambda x: 1 if x > 0 else 0)
        file["Credit_History"] = file["Credit_History"].astype('str')
        return file

    def data_con(self, file):
        print("Replacing '3+' with '3' in Dependents...")
        file["Dependents"] = file["Dependents"].replace("3+", "3")
        return file

    def caping_num_cols(self, file, col):
        print(f"Capping outliers for {col}...")
        data = file[col]
        Q1 = np.percentile(data, q=25)
        Q3 = np.percentile(data, q=75)
        IQR = Q3 - Q1
        lb = Q1 - 1.5 * IQR
        ub = Q3 + 1.5 * IQR

        # Apply capping
        data = np.where(data < lb, lb, data)
        data = np.where(data > ub, ub, data)
        file[col] = data
        return file

    def outlier_filling(self, file):
        print("Handling outliers...")
        _, num_cols = self.cat_num_split(file)
        for col in num_cols:
            file = self.caping_num_cols(file, col)
        return file

    def cat_num_split(self, file):
        print("Splitting columns into categorical and numerical...")
        cat_cols = file.select_dtypes(include="object").columns
        num_cols = file.select_dtypes(exclude="object").columns
        return cat_cols, num_cols

    def fill_null(self, file):
        print("Filling missing values...")
        cat_cols, num_cols = self.cat_num_split(file)

        for col in file:
            if col in cat_cols:
                mode = self.find_mode(file, col)
                file[col] = file[col].fillna(mode)
            elif col in num_cols:
                median = file[col].median()
                file[col] = file[col].fillna(median)
        return file

    def stand_data(self, file):
        print("Standardizing and encoding data...")
        cat_cols, num_cols = self.cat_num_split(file)

        # Initialize the scaler and encoder
        ss = StandardScaler()
        le = LabelEncoder()

        # Standardize numerical column
        for col in num_cols:
            file[[col]] = ss.fit_transform(file[[col]])

        # Encode categorical column
        for col in cat_cols:
            file[col] = le.fit_transform(file[col].astype(str))

       
        joblib.dump(ss, 'standard_scaler.joblib')
        joblib.dump(le, 'label_encoder.joblib')

        return file

    def clean_data(self, file):
        print("Starting data cleaning process...")
        file = self.drop_loan_id(file)  
        file = self.data_type_con(file)
        file = self.data_con(file)
        file = self.fill_null(file)
        file = self.outlier_filling(file)
        file = self.stand_data(file)
        
        for col in file.columns:
            print(f"{col}: {file[col].isnull().sum()} missing values")

        file.to_csv("preprocess3.csv", index=False)
        print("Data cleaning process completed.")
        return file