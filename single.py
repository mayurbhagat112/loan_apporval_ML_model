import numpy as np
import pandas as pd
import joblib

encoders = joblib.load("label_encoders.joblib")
scaler = joblib.load("scaler.joblib")

class CLEAN1():

    def find_mode(self, file, i):
        print('Entering mode computation')
        fmode = file[i].mode()[0] 
        print('Mode computation done')
        return fmode

    def data_type_con(self, file):
        print('Entering data type conversion')
        file['Loan_Amount_Term'] = file['Loan_Amount_Term'].astype('str')
        file['Credit_History'] = file['Credit_History'].astype('str')
        print('Data type conversion done')
        return file

    def data_con(self, file):
        print('Replacing 3+ with 3 in Dependents')
        file['Dependents'] = file['Dependents'].replace('3+', '3').astype('str')
        print('Replacement done')
        return file

    def caping_num_cols(self, file, i):
        print(f'Entering capping for numerical column: {i}')
        data = file[i]
        Q1 = np.percentile(data, q=25)
        Q3 = np.percentile(data, q=75)

        IQR = Q3 - Q1
        lb = Q1 - 1.5 * IQR
        ub = Q3 + 1.5 * IQR

        data = np.where(data < lb, lb, data)
        data = np.where(data > ub, ub, data)
        file[i] = data
        print(f'Capping done for {i}')
        return file

    def outlier_filling(self, file):
        _, num_cols = self.cat_num_split(file)
        for i in num_cols:
            file = self.caping_num_cols(file, i)
        return file

    def cat_num_split(self, file):
        print('Splitting columns into categorical and numerical')
        cat_cols = file.select_dtypes(include='object').columns
        num_cols = file.select_dtypes(exclude='object').columns
        print('Column splitting done')
        return cat_cols, num_cols

    def fill_null(self, file):
        print('Filling null values')
        cat_cols, num_cols = self.cat_num_split(file)

        for i in file.columns:
            if i in cat_cols:
                mode = self.find_mode(file, i)
                file[i] = file[i].fillna(mode)
                print(f'Null values filled for categorical column: {i}')
            elif i in num_cols:
                median = file[i].median()
                file[i] = file[i].fillna(median)
                print(f'Null values filled for numerical column: {i}')

        print('Null value filling complete')
        return file

    def stand_data(self, file):
        print('Entering standardization')
        cat_cols, num_cols = self.cat_num_split(file)

        # Standardize numerical columns
        file[num_cols] = scaler.transform(file[num_cols])

        # Encode categorical columns
        for i in cat_cols:
            if i in encoders:
                file[i] = encoders[i].transform(file[i])
            else:
                print(f'Warning: No encoder found for {i}. Skipping this column.')

        print('Standardization done')
        return file

    def clean_data(self, file):
        if 'Loan_ID' in file:
            file = file.drop('Loan_ID', axis=1)
            print("'Loan_ID' column dropped.")

    

        con_data = self.data_type_con(file)
        data_3to3 = self.data_con(con_data)
        fill_data = self.fill_null(data_3to3)
        data_out = self.outlier_filling(fill_data)
        file = self.stand_data(data_out)

        # Check for null values after processing
        for i in file.columns:
            print(f'{i} has {file[i].isnull().sum()} null values')

        file.to_csv('single1.csv', index=False)
        print('Data cleaning completed')

        return file
    
