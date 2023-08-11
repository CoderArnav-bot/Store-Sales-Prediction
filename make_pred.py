import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib
import json

def make_prediction(x):
    train = pd.read_csv('store-sales-time-series-forecasting/train.csv')
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(train.loc[:, train.columns != 'sales'], train['sales'], test_size=0.33, random_state=42)
    
    
    cat_attribs = ['id','date','family']
    pipe = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), cat_attribs)], remainder='passthrough')

    encoder = pipe.fit(X_train)
    x = encoder.transform(x)
    
    loaded_model = joblib.load('model.jb')
    pred_out = loaded_model.predict(x)
    
    with open("encoder.json", "w") as write_file:
       json.dump({'Predict':str(pred_out)}, write_file, indent=4)
    
    
    with open('encoder.json') as json_file:
        data = json.load(json_file)
        
    return float(data['Predict'].lstrip('[').rstrip(']'))

def make_predict(x):
    train = pd.read_csv('store-sales-time-series-forecasting/train.csv')
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(train.loc[:, train.columns != 'sales'], train['sales'], test_size=0.33, random_state=42)
    
    
    cat_attribs = ['id','date','family']
    pipe = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), cat_attribs)], remainder='passthrough')

    encoder = pipe.fit(X_train)
    x = encoder.transform(x)
    
    loaded_model = joblib.load('model.jb')
    pred_out = loaded_model.predict(x)
    return pred_out
    
    