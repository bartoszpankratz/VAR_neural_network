import zipfile
import pandas as pd
import numpy as np
import scipy.stats as sps

from statsmodels.tsa.arima_process import ArmaProcess
from arch import arch_model
from arch.univariate import EGARCH
from arch.univariate import HARCH

#Main class with Data objects used in the model

class Data:
    def __init__(self, metadata, data):
        self.metadata = metadata
        self.data = data
		
#Auxilliary functions for generating different simulation models

def Gaussian_Noise(process,iters):
#GWS
    metadata ={"process":process,"number_of_iterations":iters}
    y = np.random.normal(0, 1, size= iters)
    true_var = np.repeat([sps.norm.ppf(0.01)], iters)
    data = pd.DataFrame(data = {'Returns' : y, 'True_VAR_0.01' : true_var})
    return metadata, data
	
def get_AR(process, iters, params):
# AR
    if params == None:          
        params = {'AR': [-0.999]}
        
    metadata ={"process":process,"number_of_iterations":iters}
    metadata.update(params)
    
    ar = np.array([1] + params['AR'])
    ar_object = ArmaProcess(ar, [1])
    y = ar_object.generate_sample(nsample = iters)
    true_var = sum([y[i:-(len(ar)-i)] * (-1*ar[len(ar)-i]) for i in range(1, len(ar))]) +  sps.norm.ppf(0.01)
    y = y[len(ar): ]
        
    data =  pd.DataFrame(data = {'Returns' : y, 'True_VAR_0.01' : true_var})
    return metadata, data

def get_ARCH(process,iters,params):
# ARCH
    if params == None:          
        params = {'INIT': [0, 0.1], 'P_PARAMS': [0.2, 0.1]}
        
    metadata ={"process":process,"number_of_iterations":iters}
    metadata.update(params)
    
    model = arch_model(None, p = len(params['P_PARAMS']), o = 0, q = 0)
    param_vect = params['INIT'] + params['P_PARAMS'] 
    y = model.simulate(nobs= iters, burn= 100, params = param_vect)
    true_var  = y['volatility'] * sps.norm.ppf(0.01)
    y = y['data'] 
    
    data =  pd.DataFrame(data = {'Returns' : y, 'True_VAR_0.01' : true_var})
    return metadata, data
	
def get_GARCH(process,iters,params):
# GARCH
    if params == None:          
        params = {'INIT': [0, 0.1], 'P_PARAMS': [0.2, 0.1], 'Q_PARAMS': [0.2, 0.1]}
        
    metadata ={"process":process,"number_of_iterations":iters}
    metadata.update(params)
    
    model = arch_model(None, p = len(params['P_PARAMS']), o = 0, q = len(params['Q_PARAMS']))
    param_vect = params['INIT'] + params['P_PARAMS'] + params['Q_PARAMS']
    y = model.simulate(nobs= iters, burn= 100, params = param_vect)
    true_var  = y['volatility'] * sps.norm.ppf(0.01)
    y = y['data']  
    
    data =  pd.DataFrame(data = {'Returns' : y, 'True_VAR_0.01' : true_var})
    return metadata, data
	
def get_GJR_GARCH(process,iters,params):
# GJR GARCH
    if params == None:          
        params = {'INIT': [0, 0.1], 'P_PARAMS': [0.2, 0.1], 
                  'O_PARAMS': [0.5], 'Q_PARAMS': [0.2, 0.1]}
        
    metadata ={"process":process,"number_of_iterations":iters}
    metadata.update(params)
    
    model = arch_model(None, p = len(params['P_PARAMS']),
                               o = len(params['O_PARAMS']),
                               q = len(params['Q_PARAMS']))
    param_vect = params['INIT'] + params['P_PARAMS'] + params['O_PARAMS'] + params['Q_PARAMS']
    y = model.simulate(nobs= iters, burn= 100, params = param_vect)
    true_var  = y['volatility'] * sps.norm.ppf(0.01)
    y = y['data']
    
    data =  pd.DataFrame(data = {'Returns' : y, 'True_VAR_0.01' : true_var})
    return metadata, data
	
def get_EGARCH(process,iters,params):
# EGARCH
    if params == None:          
        params =  {'INIT': [0, 0.1], 'P_PARAMS': [0.4, 0.1], 'Q_PARAMS': [0.7, 0.1]}
        
    metadata ={"process":process,"number_of_iterations":iters}
    metadata.update(params)
    
    model = EGARCH(p = len(params['P_PARAMS']), q = len(params['Q_PARAMS']))
    param_vect = params['INIT'] + params['P_PARAMS'] + params['Q_PARAMS']
    y = model.simulate(nobs= iters, burn= 1000, parameters = param_vect, rng=np.random.standard_normal)
    true_var  = y[1] * sps.norm.ppf(0.01)
    y = y[0]
    
    data =  pd.DataFrame(data = {'Returns' : y, 'True_VAR_0.01' : true_var})
    return metadata, data
	
def get_HARCH(process,iters,params):
# HARCH
    if params == None:          
        params =  {'LAGS': [1, 5, 17], 'INIT': [1], 'PARAMS': [0.2, 0.05, 0.01]}
        
    metadata ={"process":process,"number_of_iterations":iters}
    metadata.update(params)
    
    model = HARCH(lags = params['LAGS'])
    param_vect = params['INIT'] + params['PARAMS']
    y = model.simulate(nobs = iters, burn= 1000, parameters = param_vect, 
                       rng=np.random.standard_normal)
    true_var  = y[1] * sps.norm.ppf(0.01)
    y = y[0]
    
    data =  pd.DataFrame(data = {'Returns' : y, 'True_VAR_0.01' : true_var})
    return metadata, data
	
#function for generating simulation data
def sim_data(process, iters = 1000, params = None):
    implemented = ['GWS','ARCH','AR','GARCH','GJR_GARCH','EGARCH','HARCH']
    if process == 'GWS':
        metadata, data = Gaussian_Noise(process,iters)
    elif process == 'AR':
        metadata, data = get_AR(process,iters,params)
    elif process == 'ARCH':
        metadata, data = get_ARCH(process,iters,params)
    elif process == 'GARCH':
        metadata, data = get_GARCH(process,iters,params)
    elif process == 'GJR_GARCH':
        metadata, data = get_GJR_GARCH(process,iters,params)
    elif process == 'EGARCH':
        metadata, data = get_EGARCH(process,iters,params)
    elif process == 'HARCH':
        metadata, data = get_HARCH(process,iters,params)
    return Data(metadata, data)
	
#function for extracting data from zipfile
def get_data(zip, dict, file = None):
    try:
        df = pd.read_csv(zf.open(dict))
        if file != None:
            try:
                dt = pd.read_csv(zf.open(file))
                ticker = [ticker for ticker in df["TICKER"] if ticker in file.upper()][0]
                row = df[df["TICKER"] == ticker].iloc[:,0:5]
                metadata = {col:row[col].values[0] for col in row.columns}
                data = pd.DataFrame(data = {
                'Date'    : dt['Date'][1:],
                'Prices'  : dt['Close'][1:],
                'Returns' : dt['Close'][1:].values 
                            / dt['Close'][:-1].values - 1
                })
                data = data.set_index('Date')
                return Data(metadata,data)
            except:
                print(f" {file} - no such file in directory!")
        else:
            return [get_data(zip,dict,name) for name in zf.namelist() if name != dict]
    except:
        print(f" {dict} - no such file in directory!")