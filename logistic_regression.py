import pandas as pd
import statsmodels.api as sm
import numpy as np
loansData = pd.read_csv('/Users/yishusong/projects/linear_regression/loansData_clean.csv')
# I have the cleaned data file locally
loansData['IR_TF'] = loansData['Interest.Rate'] >= 0.12
loansData['IR_TF'] = loansData['IR_TF'].astype(int)
loansData['intercept'] = 1.0
# Create an all 0 list for example: x=[0 for i in range(len(loansData.index))]
ind_vars = ['FICO.Score', 'Amount.Requested', 'intercept']
logit = sm.Logit(loansData['IR_TF'], loansData[ind_vars])
result=logit.fit()
coeff = result.params
#FICO.Score          -0.087423
#Amount.Requested     0.000174
#intercept           60.125045
#dtype: float64
FICOScore = 0 
LoanAmount = 0
def logistic_function (FicoScore, LoanAmount):
	return 1/(1+np.exp(-60.125045 + 0.087423*FicoScore - 0.000174*LoanAmount))

print(logistic_function (720, 10000))