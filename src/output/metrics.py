import pandas as pd
import numpy as np

def mdd(portfolios):
    mdd = 0
    for i in range(1, len(portfolios)):
        mai = max(portfolios[:i])
        mddi = (mai - portfolios[i]) / mai
        mdd = max(mdd, mddi)
    return mdd

def alpha(portfolios):
    pass

history = pd.read_csv( './output/summary.csv', index_col=0)

print(history.iloc[301:, :])

mdds = []
# print(history.iloc[301:,0].values.tolist())
# print(mdd(history.iloc[301:,0].values.tolist()))

for i in range(10):
    mdds.append(mdd(history.iloc[301:,i].values.tolist()))

print('mdds:', mdds)