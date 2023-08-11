import streamlit as st
from make_pred import make_prediction, make_predict 
import pandas as pd
import seaborn as sns
#import plotly.express as px
import matplotlib.pyplot as plt

# Setup data from csv
train = pd.read_csv('store-sales-time-series-forecasting/train.csv')
test = pd.read_csv('store-sales-time-series-forecasting/test.csv')


# Setup title page
st.set_page_config(page_title="Prediction")
st.header("Prediction - Stores Sales Forecasting")
st.markdown("Using XGBoost, The prediction will appear on the graphs below to intuit how the prediction was made.")
st.sidebar.header("Make Prediction")

x1 = st.sidebar.number_input("ID",step=1)
x2 = st.sidebar.text_input("Date (YYYY-MM-DD)")
x3 = st.sidebar.number_input("Store NBR",step=1)
x4 = st.sidebar.text_input("Family")
x5 = st.sidebar.number_input("On Promotion",step=1)
predict = st.sidebar.button("Predict")

if predict:
    p1 = [x1, x2, x3, x4, x5]
    x = pd.DataFrame([p1],columns=['id','date','store_nbr','family','onpromotion'])
    res = make_prediction(x)
    st.subheader(f"Predicted Value : {res}")
    
    
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

# Managing input data
#df['date']=pd.to_datetime(df['date'],format='%Y-%m-%d',errors='coerce')






#df = train.groupby('new_date')['sales'].mean().reset_index()
#plot1 = px.line(df,x="new_date",y="sales",title="Date vs Sales")
        

#st.plotly_chart(plot2)