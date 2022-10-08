import matplotlib.pyplot as plt
import pandas as pd


summary = pd.read_csv( './output/summary.csv', index_col=0)

# summary.iloc[301-2:, :].plot(y=['Simple R-GCN',	'LSTM-driven R-GCN', 'CNN-driven R-GCN', 'CNN-driven Simple R-GCN', 'CNN Stacked Simple R-GCN',	'CNN',	'sRNN',	'LSTM',	'GRU', 'Market'], use_index=True, figsize=(10, 6))
# summary.iloc[301-2:, :].plot(y=['Simple R-GCN',	'LSTM-driven R-GCN', 'CNN-driven Simple R-GCN', 'CNN Stacked Simple R-GCN', 'Market'], use_index=True, figsize=(10, 6))
summary.iloc[301-2:, :].plot(y=['LSTM-driven R-GCN', 'CNN', 'sRNN', 'LSTM', 'GRU', 'Market'], use_index=True, figsize=(10, 6))

font = {
'weight' : 'normal',
'size'   : 10,
}
plt.xlabel("Date",font)         
plt.ylabel("Performance (fAPV)",font)

# plt.axis([0, 3286, 0, 100])
plt.axis([0, 2513, 0, 40])
plt.grid()
plt.show()
# plt.savefig('fAPV.png', dpi=200)
plt.close()