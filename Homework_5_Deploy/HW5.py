# StreamlitApp — AMZN Return Predictor

import streamlit as st
import numpy as np
import pandas as pd
import boto3
import json

st.set_page_config(page_title='AMZN Return Predictor', layout='wide')
st.title('Amazon (AMZN) 5-Day Cumulative Return Predictor')
st.write('This app uses a Kernel PCA + Lasso pipeline deployed on AWS SageMaker.')

# ── AWS credentials from Streamlit secrets ─────────────────────────────────────
boto3_session = boto3.Session(
    aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
    aws_session_token=st.secrets["AWS_SESSION_TOKEN"],
    region_name="us-east-1"
)

# ── Load SP500 data from S3 ────────────────────────────────────────────────────
@st.cache_data
def load_data():
    s3 = boto3_session.client('s3')
    obj = s3.get_object(Bucket=st.secrets["AWS_BUCKET"], Key="SP500Data.csv")
    df = pd.read_csv(obj['Body'], index_col=0)
    return df

dataset = load_data()
return_period = 5

X = np.log(dataset.drop(['AMZN'], axis=1)).diff(return_period)
X = np.exp(X).cumsum()
X.columns = [name + '_CR_Cum' for name in X.columns]
X = X.dropna()

# ── SageMaker endpoint ─────────────────────────────────────────────────────────
runtime = boto3_session.client('sagemaker-runtime')
ENDPOINT_NAME = st.secrets["AWS_ENDPOINT"]

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header('Select Prediction Date')
date_options = X.index.tolist()
selected_date = st.sidebar.selectbox('Date', date_options, index=len(date_options)-1)

# ── Predict ────────────────────────────────────────────────────────────────────
if st.button('Predict AMZN Return'):
    row = X.loc[[selected_date]]
    payload = row.to_json()

    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType='application/json',
        Body=payload
    )
    result = json.loads(response['Body'].read().decode())
    predicted_return = result[0]

    st.success(f'Predicted AMZN 5-Day Cumulative Return: **{predicted_return:.4f}**')
    st.write(f'Date: {selected_date}')

# ── Show raw features ──────────────────────────────────────────────────────────
with st.expander('View Feature Data (X) for Selected Date'):
    st.dataframe(X.loc[[selected_date]])
