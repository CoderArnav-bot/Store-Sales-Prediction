### Importing Libraries ###
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


### Importing Data ###
holidays = pd.read_csv('store-sales-time-series-forecasting/holidays_events.csv')
oil = pd.read_csv('store-sales-time-series-forecasting/oil.csv')
stores = pd.read_csv('store-sales-time-series-forecasting/stores.csv')
test = pd.read_csv('store-sales-time-series-forecasting/test.csv')
train = pd.read_csv('store-sales-time-series-forecasting/train.csv')
transactions = pd.read_csv('store-sales-time-series-forecasting/transactions.csv')


### Preparing Data ###
m1=pd.merge(holidays,oil)
m2=pd.merge(train,stores, on='store_nbr', how = 'left')
m3=pd.merge(m2,transactions)
df=pd.merge(m3,m1,on="date")

### EDA ###
des = df.describe()
df.loc[(df.dcoilwtico.isnull()),'dcoilwtico'] = df.dcoilwtico.mean()
info = df.info()

#Converting the date column from string to datetime dtype.
df['new_date']=pd.to_datetime(df['date'],format='%Y-%m-%d',errors='coerce')

#Time Series plot of the sales data
plot1 = sns.lineplot(x='new_date',y='sales',data=df,ci=None,estimator='mean')


df[["year", "month", "day"]] = df["date"].str.split("-", expand = True)
df['month'].replace(['01','02','03','04','05','06','07','08','09','10','11','12'],['Jan','Feb','Mar','Apr','May','June','July','Aug','Sept','Oct','Nov','Dec'],inplace=True)
df['month'] = pd.Categorical(df['month'],
                                   categories=['Jan','Feb','Mar','Apr','May','June','July','Aug','Sept','Oct','Nov','Dec'],
                                   ordered=True)
df['day'] = pd.Categorical(df['day'],categories=['01','02','03','04','05','06','07','08','09','10','11', '12', '14','15','16','17','18','19','20','21','22','23', '24', '25', '26', '27','28','29','30','31'],ordered=True)

#plotting the monthwise sales trend
f, ax = plt.subplots(1,2,figsize=(25,15))
plot2 = sns.lineplot(x='year',y='sales',data=df,ci=None,estimator='mean',ax=ax[0])
plot3 = sns.lineplot(x='month',y='sales',data=df,ci=None,estimator='mean',ax=ax[1])


s=df.groupby('store_nbr')['sales'].mean().sort_values(ascending=False)
s=pd.DataFrame(s)
ax,f=plt.subplots(figsize=(25,15))
sns.barplot(x=s.index,y='sales',data=s,ax=None,ci=None,order=s.index)

### Predictive Analytics ###
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.loc[:, train.columns != 'sales'], train['sales'], test_size=0.33, random_state=42)


from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

cat_attribs = ['id','date','family']
pipe = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), cat_attribs)], remainder='passthrough')

encoder = pipe.fit(X_train)
X_train = encoder.transform(X_train)
X_test = encoder.transform(X_test)
test = encoder.transform(test)

# train the model
model = XGBRegressor(n_estimators=10, max_depth=20, verbosity=2)
model.fit(X_train, y_train)

score = model.score(X_test, y_test) #87.4088
    
