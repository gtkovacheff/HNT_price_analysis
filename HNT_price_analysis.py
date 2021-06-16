import pandas as pd
import numpy as np
from sklearn import linear_model
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

#define date parser
date_parser = lambda x: pd.to_datetime(x, format='%d.%m.%Y')

date_parser_2=lambda x: pd.to_datetime(x, format='%b %d, %Y')
column_parser_clean=lambda x: x.replace('€ ', '').replace('€', '').replace(',', '')
column_parser_to_numeric=lambda x: pd.to_numeric(x)
#load_the_data
# HNT_hotspot_Data = pd.read_csv('Data/HNT_hotspot_data.csv', sep=';', dtype={'Date': np.datetime64, 'Number of Hotspots': np.int64})
HNT_hotspot_Data = pd.read_csv('Data/HNT_hotspot_data.csv', sep=';', dtype={'Number of Hotspots': np.int64}, parse_dates=['Date'], date_parser=date_parser)
HNT_historical_price = pd.read_csv('Data/HNT_historical_data.csv', sep=';', parse_dates=['Date'],  date_parser=date_parser_2)
HNT_historical_price.loc[:, ['Open*', 'High', 'Low', 'Close**', 'Volume', 'Market Cap']] = HNT_historical_price.loc[:, ['Open*', 'High', 'Low', 'Close**', 'Volume', 'Market Cap']].\
    applymap(column_parser_clean).applymap(column_parser_to_numeric)

HNT_historical_price.rename(columns={'Open*': 'Open', 'Close**': 'Close'}, inplace=True)

#remove the first 6 points and the last one
# HNT_hotspot_Data = HNT_hotspot_Data[6:-1].reset_index(drop=True) #because of missing data

#join HNT_hotspot_Data with HNT_historical_price['Price']
prepped_data = pd.merge(HNT_historical_price[['Date', 'Close']], HNT_hotspot_Data, how='left', on='Date')

#find corr between the variables --> 0.884
prepped_data.corr()

#line plot NumberHotspots VS Price
prepped_data.plot(x='Number of Hotspots', y='Price', title='NumberOfHotspots vs Price')

#X_train, y_train
X_train, y_train = prepped_data.loc[:, 'Number of Hotspots'].values.reshape(-1, 1), HNT_hotspot_Data.Price.values.reshape(-1, 1)

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