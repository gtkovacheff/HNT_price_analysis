import pandas as pd
import numpy as np
from sklearn import linear_model
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

#load_the_data
HNT_hotspot_Data = pd.read_csv('Data/HNT_data.csv', sep=';').drop('Price', axis=1) #price column Irrelevant since we join with historical data
HNT_historical_price = pd.read_csv('Data/HNT_historical_data.csv', sep=';')
HNT_historical_price.rename(columns={'Open*': 'Open', 'Close**': 'Close'}, inplace=True)

#Some data wrangling
for i in HNT_historical_price.columns:
    if i == 'Date':
        HNT_historical_price['Date'] = pd.to_datetime(HNT_historical_price['Date'], format='%b %d, %Y')
    else:
        HNT_historical_price[i] = HNT_historical_price[i].map(lambda x: x.replace('€ ', '').replace('€', '').replace(',', ''))

#to pd_date_time
HNT_hotspot_Data['Date'] = pd.to_datetime(HNT_hotspot_Data['Date'], format='%d.%m.%Y')

#remove the first 6 points and the last one
HNT_hotspot_Data = HNT_hotspot_Data[6:-1].reset_index(drop=True) # because of missing data

#join HNT_hotspot_Data with HNT_historical_price['Price']
HNT_hotspot_Data = pd.merge(HNT_hotspot_Data, HNT_historical_price[['Date', 'Close']], how='inner', on='Date').rename(columns={'Close': 'Price'})

#convert to numeric
for i in HNT_hotspot_Data.columns[1:]:
    HNT_hotspot_Data[i]=pd.to_numeric(HNT_hotspot_Data[i])

#find corr between the variables --> 0.884
HNT_hotspot_Data.corr()

#line plot NumberHotspots VS Price
HNT_hotspot_Data.plot(x='Number of Hotspots', y='Price', title='NumberOfHotspots vs Price')

#X_train, y_train
X_train, y_train = HNT_hotspot_Data.loc[:, 'Number of Hotspots'].values.reshape(-1, 1), HNT_hotspot_Data.Price.values.reshape(-1, 1)

#Linear reggression
reg = linear_model.LinearRegression()

#fit the reggression
reg.fit(X_train, y_train)

#find the coefficients
reg.coef_
reg.intercept_


print('Variance score: {}'.format(reg.score(X_train, y_train)))
#78 percent of the variation could be explained with the number of hotspots

#predict new price when hotspots reach 100 000
reg.predict([[100000]]) #price would be 37.33

plt.style.use('fivethirtyeight')

#residual errors in the training data
plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train, color="green", s=10, label='Train data')
plt.title('Residuals vs fitted')

# The mean squared error
print(f'Mean squared error: {mean_squared_error(reg.predict(X_train), y_train)}') #4.609435727152511

#In summary: Regression analysis impies that the equation for the future price would be
# -2.52768419 + 0.00039858 * X, which results in: if the number of hotspots doubles (100000)
# we can observe price around 37.33 euro (3x than today ~ 12euro) , variation explained from the number of hotspots is 78% which is higher
# than expectation and the mean squared error is around 4.60 which could be less but we should optimize the regression

#Approach 2
# train stochastic gradient descent
#define param grid
param_grid = {
    'penalty': ['l1', 'l2'],
    'alpha': [0.0001, 0.001, 0.001, 0.01],
}
# create pipeline
SDG = make_pipeline(StandardScaler(), linear_model.SGDRegressor())

#search from the grid space
search = GridSearchCV(SDG, param_grid, n_jobs=-1)

#fit data
search.fit(X_train, y_train.ravel())
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)

SDG.fit(X_train, y_train.ravel())
SDG.score(X_train, y_train.ravel())
SDG['sgdregressor'].coef_
SDG['sgdregressor'].intercept_

SDG.predict([[100000]])
print(f'Mean squared error: {mean_squared_error(SDG.predict(X_train), y_train.ravel())}') #4.609435727152511