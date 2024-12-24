import pickle
import streamlit as st
import os
import numpy as np
import pandas as pd
from cleaning import CLEAN
from single import CLEAN1

# Load the model dynamically based on user selection
def load_model(model_name):
    try:
        with open(f"model/{model_name}.pkl", "rb") as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error(f"⚠️ The model file '{model_name}.pkl' was not found. Please ensure it's in the correct directory.")
        st.stop()

# Get a list of model names (assuming they are .pkl files in the 'model' directory)
model_files = [f.replace(".pkl", "") for f in os.listdir("model") if f.endswith(".pkl")]

# Prediction function for single input
def predict_loan_status(model, gender, married, dependents, education, self_employed, applicant_income,
                        coapplicant_income, loan_amount, loan_amount_term, credit_history, property_area):
    # Prepare the input data correctly according to the model's expected format
    input_data = np.array([[gender, married, dependents, education, self_employed, applicant_income,
                            coapplicant_income, loan_amount, loan_amount_term, credit_history, property_area]])
    prediction = model.predict(input_data)
    return "YES" if prediction == 1 else "NO"

# Function to process and predict from CSV
def process_and_predict_csv(model, file):
    # Read the uploaded CSV file
    data = pd.read_csv(file)

    # Check if required columns are present
    required_columns = [
        'Loan_ID', 'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 
        'Credit_History', 'Property_Area'
    ]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        st.error(f"⚠️ Missing required columns: {', '.join(missing_columns)}")
        return

    # Extract the Loan_ID column for the final output
    loan_ids = data['Loan_ID']

    # Apply preprocessing steps (from your CLEAN class)
    cleaner = CLEAN()
    data = cleaner.data_type_con(data)
    data = cleaner.data_con(data)
    data = cleaner.fill_null(data)
    data = cleaner.outlier_filling(data)
    data = cleaner.stand_data(data)

    # Check the feature names after preprocessing
    st.write("Feature names after preprocessing:", data.columns.tolist())

    # Prepare data for prediction (exclude Loan_ID)
    input_data = data[required_columns[1:]]  # Exclude Loan_ID column
    predictions = model.predict(input_data)

    # Create a DataFrame for results
    results = pd.DataFrame({
        'Loan_ID': loan_ids,
        'Loan_Status': ['YES' if pred == 1 else 'NO' for pred in predictions]
    })

    # Display the results
    st.dataframe(results)

    # Allow the user to download the result
    csv = results.to_csv(index=False)
    st.download_button(
        label="Download Prediction Results",
        data=csv,
        file_name="loan_predictions.csv",
        mime="text/csv"
    )

# Main function
def main():
    # Application title and description
    st.markdown(
        """
        <div style="text-align: center;">
            <h1 style="color: #4CAF50;">🏦 Predict Loan Eligibility</h1>
            <p style="font-size: 18px;">Enter the details below to check if your loan will be approved!</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Model selection
    model_name = st.selectbox("Select Model", model_files, help="Choose a model to make predictions.")
    model = load_model(model_name)

    # Tab selection for form or file upload
    option = st.radio("Choose Input Type", ["Single Input", "CSV File Upload"])

    if option == "Single Input":
        # Form layout for single input
        with st.form("loan_form"):
            # Organize fields in columns
            col1, col2 = st.columns(2)

            with col1:
                gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
                married = st.selectbox("Marital Status", [0, 1], format_func=lambda x: "Married" if x == 1 else "Single")
                dependents = st.number_input("Number of Dependents", min_value=0, step=1)
                education = st.selectbox("Education Level", [0, 1], format_func=lambda x: "Graduate" if x == 1 else "Not Graduate")
                self_employed = st.selectbox("Self Employed", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

            with col2:
                applicant_income = st.number_input("Applicant Income (₹)", min_value=0, step=1000)
                coapplicant_income = st.number_input("Co-applicant Income (₹)", min_value=0, step=1000)
                loan_amount = st.number_input("Loan Amount (₹)", min_value=0, step=1000)
                loan_amount_term = st.selectbox("Loan Term (months)", [360, 120, 240, 180, 60, 300, 480, 36, 84, 12])
                credit_history = st.selectbox("Credit History", [0, 1], format_func=lambda x: "Good" if x == 1 else "Bad")
                property_area = st.selectbox("Property Area", [0, 1, 2], format_func=lambda x: ["Urban", "Semiurban", "Rural"][x])

            # Submit button
            submitted = st.form_submit_button("Predict Loan Status")
            if submitted:
                with st.spinner("Calculating..."):
                    result = predict_loan_status(model, gender, married, dependents, education, self_employed,
                                                 applicant_income, coapplicant_income, loan_amount,
                                                 loan_amount_term, credit_history, property_area)
                st.success(f"**Result:** The loan status is: {result}")

    elif option == "CSV File Upload":
        # Upload CSV file
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            process_and_predict_csv(model, uploaded_file)

    # About section
    with st.expander("About this App"):
        st.markdown(
            """
            - **Purpose:** Predicts loan approval status using a trained model.
            - **Features:** User-friendly input form for individual loan predictions and CSV file upload for batch predictions.
            - **Built with:** Streamlit, NumPy, and Python's machine learning libraries.
            """
        )
        st.markdown("✨ Created with love by MAYUR.")

# Run the app
if __name__ == "__main__":
    main()
