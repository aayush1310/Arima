#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')
# rcParams['figure.figsize'] = 10,6


# In[3]:


df = pd.read_csv('./perrin-freres-monthly-champagne.csv')


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.columns=['Month','Sales']


# In[7]:


df


# In[8]:


df = df.dropna()


# In[9]:


df


# In[10]:


type(df['Month'][0])


# In[11]:


# Convering String to Timestamp
df['Month'] = pd.to_datetime(df['Month'])
indexedDataset = df.copy()
indexedDataset.set_index(['Month'], inplace=True)


# In[12]:


indexedDataset.index.year


# In[79]:


indexedDataset


# In[271]:


df_all = indexedDataset['Sales']


# In[272]:


df_all


# In[158]:


df_train = indexedDataset[indexedDataset.index.year != 1972]['Sales']
df_test = indexedDataset[indexedDataset.index.year == 1972]['Sales']

print('train examples: ', df_train.shape[0])
print('test examples: ', df_test.shape[0])


# In[159]:


df_train


# In[160]:


indexedDataset.describe()


# In[161]:


# Visualization


# In[162]:


# Plotting Graph
df_train.plot(figsize=(10,8))
plt.xlabel('Year')
plt.legend()
plt.ylabel('Sales in Millions')
plt.plot(df_train)


# The graph clearly says thet it is seasonal. Repeating Peaks and geting higher and higher. 
# 

# Let's plot the graph into trend, seasonality and residuals.

# In[163]:


import statsmodels.api as sm


# In[164]:


decomposition = sm.tsa.seasonal_decompose(df_train, model='additive', freq = 12)
fig = decomposition.plot()
fig.set_size_inches(15,10)
fig.suptitle('Decomposition plot')
plt.show()


# In[165]:


from IPython.display import display, HTML


# In[166]:


# Determining Rolling Statistics and Plotting data
def make_plot(series,window, title = ''):
    rollmean = series.rolling(window=12).mean()
    rollstd = series.rolling(window =12).std()
    plt.figure(figsize = (8,5))
    series.plot()
    rollmean.plot(label = 'rolling mean')
    rollstd.plot(label = 'rolling std')
    plt.ylabel('Sales in millions')
    plt.title(title)
    plt.legend()
    plt.show()


# In[167]:


# Performing Augmented Dicky Fuller(ADFC) Test
from statsmodels.tsa.stattools import adfuller


# In[168]:


def stationarity(timeseries):
    print('Dickey-Fuller Test: ')
    dftest=adfuller(timeseries, autolag='AIC')
    dfoutput=pd.Series(dftest[0:4], index=['Test Statistic','p-value','Lags Used','No. of Obs'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)


# In[169]:


make_plot(df_train, 12, 'Time series with rolling mean and std')


# The plot shows that the rolling mean as well as the standard deviation is not constant, this gives us a hint of non-stationarity. Let us confirm it by using the Dicky Fuller Test.

# In[170]:


stationarity(df_train)


# In[171]:


# Here p-value is not less than 0.05. Also, Critical Value is also not greater that test statistic
# Making the data stationary and then again checking its stationarity
# First we will try with differencing


# In[172]:


seasonal_diff_df = df_train - df_train.shift(12)
seasonal_diff_df.dropna(inplace=True)


# In[173]:


decomposition = sm.tsa.seasonal_decompose(seasonal_diff_df, model='additive', freq = 12)
fig = decomposition.plot()
fig.set_size_inches(15,10)
fig.suptitle('Decomposition plot')
plt.show()


# In[174]:


make_plot(seasonal_diff_df,12, 'Time series without seasonality')


# In[175]:


stationarity(seasonal_diff_df)


# In[176]:


df_acf = acf(seasonal_diff_df,nlags=40)
df_pacf = pacf(seasonal_diff_df,nlags =40)


# In[177]:


plt.figure(figsize = (15,8))
plt.subplot(211)
plot_acf(seasonal_diff_df, ax=plt.gca(), color ='blue')
plt.subplot(212)
plot_pacf(seasonal_diff_df, ax=plt.gca(),  color ='blue')
plt.show()


# As you can see, that there are only a few significant values in both the ACF and the PACF. Both the PACF and ACF drop off suddenly, perhaps suggesting a mix of AR and MA models. All this is insignificant for us as we will try a bunch of different parameters and choose the one with lower RMSE(root mean square error)

# We define a function to take the seasonal difference for a series to remove the seasonality.

# As the model will predict values for non-seasonal data we need to add back the seasonal components we subtracted to get the true predictions. We define a function called inverse_seas_diff to inverse the seasonal difference of the series.

# In[305]:


def inverse_seas_diff(original, value, interval):
#     print("Modified Original===> {} Value===> {} result===> {}".format(original.iloc[-interval], value, value + original.iloc[-interval]))
    return value + original.iloc[-interval]

def make_seas_diff(original, interval):
    seasonal_diff_df = original - original.shift(interval)
    seasonal_diff_df.dropna(inplace  =True)
    return list(seasonal_diff_df)


# In[306]:


from statsmodels.tsa.arima_model import ARIMA


# In[309]:


def rmse(order):
    
    train_list = list(df_train)
    predictions = []
    for index,true_value in df_test.iteritems(): 
        seasonal_diff_df = make_seas_diff(pd.Series(train_list),12)
        model = ARIMA(seasonal_diff_df, order=order)
        results_AR = model.fit(trend = 'nc',disp=0)
        pred = results_AR.forecast()[0][0]
        pred = inverse_seas_diff(pd.Series(train_list),pred,12)
        train_list.append(true_value)
        predictions.append(pred)
        
    return ('RMSE: ', mean_squared_error(list(df_test), predictions) ** 0.5)


# In[310]:


import itertools
p = range(0,5)
d = range(0,3)
q = range(0,5)
combinations = list(itertools.product(p,d,q))

for parameter in combinations:
    try:
        order = parameter
        print(order,':', rmse(order))
    except:
        continue


# In[311]:


train_list = list(df_train)
predictions = []
for index,true_value in df_test.iteritems(): 
    seasonal_diff_df = make_seas_diff(pd.Series(train_list),12)
    model = ARIMA(seasonal_diff_df, order= (0,1,1))
    results_AR = model.fit(trend = 'nc',disp=0)
    pred = results_AR.forecast()[0][0]
    pred = inverse_seas_diff(pd.Series(train_list),pred,12)
    train_list.append(true_value)
    predictions.append(pred)

print ('RMSE: ', mean_squared_error(list(df_test), predictions) ** 0.5)

pred_series =  pd.Series(predictions, index = df_test.index)
plt.plot(pred_series, label = 'predicted sales')
plt.plot(df_test, label = 'actual sales')
plt.xticks(rotation = 'vertical')
plt.ylabel('Sales in millions')
plt.title('Predicted and Actual sales')
plt.legend()


# In[312]:


train_list = list(df_train)
seasonal_diff_df = make_seas_diff(pd.Series(train_list),12)
model = ARIMA(seasonal_diff_df, order= (0,1,1))
model.fit = model.fit(trend = 'nc',disp=0)


# In[313]:


model.fit.summary()


# ## Predicting for next 2 Years

# In[288]:


from pandas.tseries.offsets import DateOffset
future_dates=[indexedDataset.index[-1]+ DateOffset(months=x)for x in range(0,24)]


# In[289]:


future_dates_df=pd.DataFrame({"Month":future_dates,"Sales":np.nan})


# In[290]:


future_dates_df


# In[291]:


future_dates_df['Month'] = pd.to_datetime(future_dates_df['Month'])
future_indexedDataset = future_dates_df.copy()
future_indexedDataset.set_index(['Month'], inplace=True)


# In[292]:


future_indexedDataset


# In[293]:


df_future=future_indexedDataset['Sales']


# In[294]:


len(df_future)


# In[295]:


len(list(df_all))


# In[314]:


train_list = list(df_all)
predictions = []
for index,true_value in df_future.iteritems(): 
    seasonal_diff_df = make_seas_diff(pd.Series(train_list),12)
    model = ARIMA(seasonal_diff_df, order= (0,1,1))
    results_AR = model.fit(trend = 'nc',disp=0)
    pred = results_AR.forecast()[0][0]
    pred = inverse_seas_diff(pd.Series(train_list),pred,12)
    train_list.append(pred)
    predictions.append(pred)

pred_series =  pd.Series(predictions, index = df_future.index)
plt.plot(pred_series, label = 'predicted sales')
plt.plot(df_all, label = 'actual sales')
plt.xticks(rotation = 'vertical')
plt.ylabel('Sales in millions')
plt.title('Predicted and Actual sales')
plt.legend()

