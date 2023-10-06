from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.stattools import kpss,adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

from scipy.stats import gaussian_kde, norm

def checkresiduals(fitted_model, plot:bool=True, lb_lag=20):
    
    d = np.maximum(fitted_model.loglikelihood_burn, fitted_model.nobs_diffuse)
    
    if hasattr(fitted_model.data, 'dates') and fitted_model.data.dates is not None:
        ix = fitted_model.data.dates[d:]
    else:
        ix = np.arange(fitted_model.nobs - d) 
        
    residuals = pd.Series(
            fitted_model.filter_results.standardized_forecasts_error[0, d:],
            index=ix)
    
    ljungbox = acorr_ljungbox(residuals, lags=lb_lag)    
    jarquebera = jarque_bera(resids=residuals)

    print()
    if plot:
        fig, ax = plt.subplot_mosaic([['a)', 'a)'], ['b)', 'c)']],
                              layout='constrained', figsize=(10,5))
        
        # Residuals vs. Time plot
        residuals.plot(ax = ax['a)'], color='k', linewidth=1.0)
        ax['a)'].set_title(f'ARIMA Residuals')
        ax['a)'].hlines(0,ix[0],ix[-1], alpha=0.4, color='k',)
        
        # Residuals ACF
        plot_acf(residuals, lags=40, ax=ax['b)'],auto_ylims=True, marker='.')
        ax['b)'].set_title('ACF of Residuals')

        # Histogram of residuals
        ax['c)'].hist(residuals, bins=20, density=True, alpha=0.7, color='g')
        ax['c)'].set_title('Histogram of Residuals')
        
        kde = gaussian_kde(residuals)
        xlim = (-1.96*2, 1.96*2)
        x = np.linspace(xlim[0], xlim[1])
        ax['c)'].plot(x, kde(x), label='KDE')
        ax['c)'].plot(x, norm.pdf(x), label='N(0,1)', color ='darkorange', linestyle = '-')
        ax['c)'].set_xlim(xlim)
        ax['c)'].legend()
        ax['c)'].set_title('Histogram plus estimated density')


        plt.tight_layout()
    
    print('=========== Ljung-Box ===========')
    print('Ljung-Box Test Statistic:', ljungbox.lb_stat[lb_lag])
    print('Ljung-Box p-value:', ljungbox.lb_pvalue[lb_lag])
    print('=========== Jarque-Bera ===========')
    print('Jarque Bera Test Statistic:', jarquebera[0])
    print('Jarque Bera p-value:', jarquebera[1])
    

def one_step_arima(arima_fitted, outsample):
    
    forecasts = []
    model = arima_fitted
    
    for t in range(outsample.shape[0]):     
        yhat = model.forecast(steps=1)
        forecasts.append(yhat[0])    
        model = model.append(outsample.iloc[t:t+1], refit=False)
    
    return forecasts

def metrics(outsample, predictions):

    mae = mean_absolute_error(outsample, predictions)
    mse = mean_squared_error(outsample, predictions)
    mape = mean_absolute_percentage_error(outsample, predictions)
    rmse = np.sqrt(mse)

    return pd.DataFrame([
        {"Metric":"RMSE","Value":rmse},
        {"Metric":"MAE","Value":mae},       
        {"Metric":"MAPE","Value":mape},
        ])
    
def adf_test(y_vector):
    regression_types = ['n', 'c', 'ct']  # List of regression types
    labels = ['None', 'Constant', 'Constant/Trend']  # Labels for regression types

    for regression_type, label in zip(regression_types, labels):
        adf_test = adfuller(y_vector, regression=regression_type, autolag='BIC')
        print('-------------{}--------------'.format(label))
        print(f'ADF Statistic:{adf_test[0]}')
        print(f'p-value: {adf_test[1]}')
        print(f'Used Lags: {adf_test[2]}')
        print(f'Critical Values: {adf_test[4]}')
        if adf_test[0] < adf_test[4].get('5%'): 
            print('--> Stationary')
        else:
            print('--> Non-Stationary')
            

def kpss_test(y_vector):
    
    regression_types = ['c', 'ct']  # List of regression types
    labels = ['Constant', 'Constant/Trend']  # Labels for regression types

    for regression_type, label in zip(regression_types, labels):
        kpss_test = kpss(y_vector, regression=regression_type)
        print('-------------{}--------------'.format(label))
        print(f'KPSS Statistic:{kpss_test[0]}')
        print(f'p-value: {kpss_test[1]}')
        print(f'Used Lags: {kpss_test[2]}')
        print(f'Critical Values: {kpss_test[3]}')
        if kpss_test[0] < kpss_test[3].get('5%'): 
            print('--> Stationary')
        else:
            print('--> Non-Stationary')
    