
import streamlit as st
import pandas as pd
import numpy as np
import boto3
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

st.set_page_config(page_title='Loan Default Predictor', layout='wide')
st.title('Loan Default Risk Predictor')
st.write('Powered by XGBoost pipeline deployed on AWS SageMaker.')

ENDPOINT_NAME = 'loan-default-xgb-endpoint'
runtime = boto3.client('sagemaker-runtime')

st.sidebar.header('Borrower Input Features')
loan_amnt       = st.sidebar.number_input('Loan Amount ($)', 500, 40000, 10000, 500)
funded_amnt     = st.sidebar.number_input('Funded Amount ($)', 500, 40000, 10000, 500)
funded_amnt_inv = st.sidebar.number_input('Funded Amount (Investors) ($)', 0, 40000, 9800, 500)
term            = st.sidebar.selectbox('Loan Term (months)', [36, 60])
int_rate        = st.sidebar.slider('Interest Rate (%)', 5.0, 30.0, 13.0, 0.1)
installment     = st.sidebar.number_input('Monthly Installment ($)', 10.0, 2000.0, 330.0, 10.0)
grade_encoded   = st.sidebar.selectbox('Credit Grade', [1,2,3,4,5,6,7], format_func=lambda x: 'ABCDEFG'[x-1])
annual_inc      = st.sidebar.number_input('Annual Income ($)', 10000, 500000, 65000, 1000)
dti             = st.sidebar.slider('Debt-to-Income Ratio', 0.0, 50.0, 18.0, 0.1)
delinq_2yrs     = st.sidebar.number_input('Delinquencies (last 2 yrs)', 0, 20, 0)
fico_high       = st.sidebar.slider('FICO Score (High)', 580, 850, 710)
fico_low        = st.sidebar.slider('FICO Score (Low)', 575, 845, 705)
open_acc        = st.sidebar.number_input('Open Credit Accounts', 0, 60, 10)
pub_rec         = st.sidebar.number_input('Public Records', 0, 10, 0)
revol_bal       = st.sidebar.number_input('Revolving Balance ($)', 0, 200000, 15000, 500)
revol_util      = st.sidebar.slider('Revolving Utilization (%)', 0.0, 100.0, 45.0, 0.1)
total_acc       = st.sidebar.number_input('Total Credit Accounts', 1, 100, 22)
mort_acc        = st.sidebar.number_input('Mortgage Accounts', 0, 20, 1)
pub_rec_bankruptcies = st.sidebar.number_input('Public Record Bankruptcies', 0, 5, 0)

input_df = pd.DataFrame([{
    'loan_amnt': loan_amnt, 'funded_amnt': funded_amnt,
    'funded_amnt_inv': funded_amnt_inv, 'term': term,
    'int_rate': int_rate, 'installment': installment,
    'grade_encoded': grade_encoded, 'annual_inc': annual_inc,
    'dti': dti, 'delinq_2yrs': delinq_2yrs,
    'fico_range_high': fico_high, 'fico_range_low': fico_low,
    'last_fico_range_high': fico_high, 'last_fico_range_low': fico_low,
    'open_acc': open_acc, 'pub_rec': pub_rec,
    'revol_bal': revol_bal, 'revol_util': revol_util,
    'total_acc': total_acc, 'mort_acc': mort_acc,
    'pub_rec_bankruptcies': pub_rec_bankruptcies,
    'out_prncp': 0.0, 'out_prncp_inv': 0.0,
    'total_rec_late_fee': 0.0, 'last_pymnt_amnt': 0.0,
}])

if st.button('Predict Default Risk'):
    try:
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='application/json',
            Body=input_df.to_json()
        )
        result = json.loads(response['Body'].read().decode())
        prob = result['probability'][0]
        pred = result['prediction'][0]

        st.metric('Default Probability', f'{prob:.1%}')
        if pred == 1:
            st.error('HIGH RISK — Model predicts this loan will DEFAULT')
        else:
            st.success('LOW RISK — Model predicts this loan will be FULLY PAID')

        fig, ax = plt.subplots(figsize=(5, 1.2))
        color = '#2ecc71' if prob < 0.4 else '#e67e22' if prob < 0.65 else '#e74c3c'
        ax.barh(['Risk'], [prob], color=color, height=0.5)
        ax.barh(['Risk'], [1-prob], left=[prob], color='#ecf0f1', height=0.5)
        ax.set_xlim(0, 1)
        ax.axvline(0.5, color='gray', linestyle='--', linewidth=1)
        ax.set_title(f'Default Risk: {prob:.1%}')
        st.pyplot(fig)
        plt.close()

    except Exception as e:
        st.error(f'Prediction error: {e}')

st.markdown('---')
st.caption('LendingClub Loan Default Predictor — XGBoost | Final Project')
