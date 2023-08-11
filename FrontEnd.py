import streamlit as st
import json
import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from make_pred import make_predict 

# Setup data from csv
train = pd.read_csv('store-sales-time-series-forecasting/train.csv')
test = pd.read_csv('store-sales-time-series-forecasting/test.csv')

st.title("Basic Prediction")

st.write("Using XGBOOST")
x1 = st.sidebar.number_input("ID",step=1)
x2 = st.sidebar.text_input("Date (YYYY-MM-DD)")
x3 = st.sidebar.number_input("Store NBR",step=1)
x4 = st.sidebar.text_input("Family")
x5 = st.sidebar.number_input("On Promotion",step=1)

inputs = {
  "x1": x1,
  "x2": x2,
  "x3": x3,
  "x4": x4,
  "x5": x5
}


if st.button('Predict'):
    res = requests.post(url="http://127.0.0.1:8000/predict",data=json.dumps(inputs))
    st.subheader(f"Response from API = {res.text}")
    
    train['new_date']=pd.to_datetime(train['date'],format='%Y-%m-%d',errors='coerce')

    fig = plt.figure(figsize=(10, 4))
    sns.lineplot(x='new_date',y='sales',data=train,errorbar=None,estimator='mean')
    st.pyplot(fig)
    
    pred = make_predict(test)
    test['new_date']=pd.to_datetime(test['date'],format='%Y-%m-%d',errors='coerce')
    test['sales'] = pred
    fig = plt.figure(figsize=(10, 4))
    sns.lineplot(x='new_date',y='sales',data=test,errorbar=None,estimator='mean')
    st.pyplot(fig)