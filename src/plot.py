import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd
import numpy as np

# sns.set()

# Load the trading history
# ---------------------------------------------------------------------------------------
history = pd.read_csv( 'prices-2022-Aug-raw.csv', index_col=0)

# Make a pretty plot
# ---------------------------------------------------------------------------------------
history.iloc[301-2:, :].plot(y=['AAPL', 'AMZN',	'ASML',	'DXCM',	'INTC',	'MSFT',	'NVDA',	'NFLX',	'PEP',	'SGEN'], use_index=True, figsize=(10, 6))

# font = {
# 'weight' : 'normal',
# 'size'   : 10,
# }
# plt.xlabel("Date",font)         
# plt.ylabel("Close Prices",font)

# plt.axis([0, 3289, 0, 900])
# plt.grid()
# plt.show()
# plt.savefig('price.png', dpi=200)
# plt.close()

rel = np.load('NASDAQ_industry_relation.npy')
print(rel.sum())
print(rel.shape)
# print(rel[:,:,0])
# print(rel[:,:,1])
# print(rel[:,:,2])
# print(rel[:,:,3])
# print(rel[:,:,4])
# print(rel[:,:,5])
# print(rel[:,:,6])
# print(rel[:,:,7])
# print(rel[:,:,8])
# print(rel[:,:,9])
# print(rel)