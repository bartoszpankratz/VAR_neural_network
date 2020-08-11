import scipy.stats as sps
import pandas as pd
import numpy as np

        #if data.name in ['GARCH', 'GARCH_ARCH', 'GARCH_ARCH_T']:
        #    for i in range(len(params['ALPHA'])):
        #        name = ['TrueVaR_' + str(params['ALPHA'][i])]
        #        self.vars[name] = data.data[name]
        
        #self.backtest = {
        #    'Kupiec'         : [],
        #    'Bernoulli'      : [],
        #    'Christoffersen' : []
        #}
		
class VaR:
    def __init__(self, data, methods, alpha, look_back):
        self.name  = data.metadata["NAME"] + "_" + "VAR"
        self.vars = pd.DataFrame({
            'ID'    : data.data.index,
            'Returns' : data.data['Returns']})
        self.vars = self.vars.set_index('ID')
        if "True_VAR_0.01" in data.data.columns:
            self.vars['True_VAR_0.01'] = data.data["True_VAR_0.01"]
		for method in methods:
			if method == "VCA":
				self.calc_VCA(alpha,look_back)
			elif method == "HIST":
				self.calc_Hist(alpha,look_back)
            
    def calc_VCA(self,alpha,look_back):
        INV_NORM   = sps.norm.ppf(alpha)
        var_titles = ['VCA_' + str(alpha[i]) for i in range(len(alpha))]
        y = self.vars['Returns']
        y_hat = [[np.mean(y[i-look_back: i]) + np.std(y[i-look_back: i]) * INV_NORM[j]
                 for i in range(look_back, len(y))] for j in range(len(INV_NORM))]
        for i,name in enumerate(var_titles):
            self.vars[name] = ""
            self.vars.loc[self.vars.index[look_back:len(y)], name] = y_hat[i]
    
    def calc_Hist(self,alpha,look_back):
        var_titles = ['HIST_' + str(alpha[i]) for i in range(len(alpha))]
        y  = self.vars['Returns']
        y_hat = [[np.percentile(y[i-look_back: i], q = a*100) 
                  for i in range(look_back, len(y))] for a in alpha]
        for i,name in enumerate(var_titles):
            self.vars[name] = ""
            self.vars.loc[self.vars.index[look_back:len(y)], name] = y_hat[i]   
       

    