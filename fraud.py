import streamlit as st
import pandas as pd
import joblib

# Load saved files
scaler = joblib.load('scaler.pkl')
model = joblib.load('logistic_model.pkl')
columns = joblib.load('columns.pkl')

st.title('Fraud Detection')
st.markdown('Enter the transaction details to predict if it is fraudulent or not.')
st.divider()

# Input fields
transaction_type = st.selectbox(
    'Transaction Type',
    ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']
)
amount = st.number_input(
    'Amount',
    min_value=0.0,
    value=1000.0,
    step=0.01
)
oldbalanceOrg = st.number_input(
    'Old Balance Origin',
    min_value=0.0,
    value=0.0,
    step=0.01
)
newbalanceOrig = st.number_input(
    'New Balance Origin',
    min_value=0.0,
    value=0.0,
    step=0.01
)
oldbalanceDest = st.number_input(
    'Old Balance Destination',
    min_value=0.0,
    value=0.0,
    step=0.01
)
newbalanceDest = st.number_input(
    'New Balance Destination',
    min_value=0.0,
    value=0.0,
    step=0.01
)

if st.button('Predict'):
    # Create dataframe with ALL features (including engineered ones)
    input_data = pd.DataFrame({
        'type': [transaction_type],
        'amount': [amount],
        'oldbalanceOrg': [oldbalanceOrg],
        'newbalanceOrig': [newbalanceOrig],
        'oldbalanceDest': [oldbalanceDest],
        'newbalanceDest': [newbalanceDest]
    })
    
    # CREATE ENGINEERED FEATURES (must match training)
    input_data['balanceDiffOrigin'] = input_data['newbalanceOrig'] - input_data['oldbalanceOrg']
    input_data['balanceDiffDest'] = input_data['newbalanceDest'] - input_data['oldbalanceDest']
    
    # One-hot encode categorical column
    input_data = pd.get_dummies(
        input_data,
        columns=['type'],
        drop_first=True
    )
    
    # Add missing columns with 0 (for one-hot encoded categories not present)
    for col in columns:
        if col not in input_data.columns:
            input_data[col] = 0
    
    # Keep column order same as training
    input_data = input_data[columns]
    
    # Scale numeric columns
    numeric = [
        'amount',
        'oldbalanceOrg',
        'newbalanceOrig',
        'oldbalanceDest',
        'newbalanceDest',
        'balanceDiffOrigin',
        'balanceDiffDest'
    ]
    input_data[numeric] = scaler.transform(input_data[numeric])
    
    # Prediction
    prediction = model.predict(input_data)[0]
    st.subheader(f'Prediction Result: {int(prediction)}')
    if prediction == 1:
        st.error('The transaction is predicted to be fraudulent.')
    else:
        st.success('The transaction is predicted to be legitimate.')