### Importing Libraries ###
import pandas as pd
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib



def make_model_save():
    test = pd.read_csv('store-sales-time-series-forecasting/test.csv')
    train = pd.read_csv('store-sales-time-series-forecasting/train.csv')
    
    ### Prediction ###
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(train.loc[:, train.columns != 'sales'], train['sales'], test_size=0.33, random_state=42)
    
    
    cat_attribs = ['id','date','family']
    pipe = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), cat_attribs)], remainder='passthrough')

    encoder = pipe.fit(X_train)
    X_train = encoder.transform(X_train)
    X_test = encoder.transform(X_test)
    test = encoder.transform(test)
        
    # train the model
    model = XGBRegressor(n_estimators=10, max_depth=20, verbosity=2)
    model.fit(X_train, y_train)            
    joblib.dump(model,'model.jb')
    
    