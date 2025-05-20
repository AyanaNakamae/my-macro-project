import pandas as pd
import matplotlib.pyplot as plt
from fredapi import Fred
from statsmodels.tsa.filters.hp_filter import hpfilter
import numpy as np

# 1. 国の選択とGDPデータの取得
# 例としてアメリカ合衆国 (USA) を選択します。FREDのシリーズIDを使用します。
# 実質GDPの四半期データ (GDPC1) を取得します。
fred = Fred(api_key='8dfaf30a8f87d5731d37b87fb46c4839') # 'YOUR_FRED_API_KEY' を実際のAPIキーに置き換えてください。
gdp_data = fred.get_series('NGDPRSAXDCCAQ')

# データをPandas Seriesとして確認
print(gdp_data.head())

# 2. 対数変換
log_gdp = np.log(gdp_data)

# 3. HPフィルターの適用とλの検討
lambdas = [10, 100, 1600]
trends = {}
cycles = {}

for lam in lambdas:
  # hpfilterはタプルでトレンドと循環成分を返します
  trend, cycle = hpfilter(log_gdp, lamb=lam)
  trends[lam] = trend
  cycles[lam] = cycle

# 4. 可視化

# グラフ1：元のデータとトレンド成分の比較
plt.figure(figsize=(12, 6))
plt.plot(log_gdp.index, log_gdp, label='Original Log GDP', color='blue')
for lam in lambdas:
  plt.plot(trends[lam].index, trends[lam], label=f'Trend (λ={lam})')

plt.title('Original Log GDP vs. HP Filter Trends with Different λ')
plt.xlabel('Date')
plt.ylabel('Log GDP')
plt.legend()
plt.grid(True)
plt.show()

# グラフ2：循環成分の比較
plt.figure(figsize=(12, 6))
for lam in lambdas:
  plt.plot(cycles[lam].index, cycles[lam], label=f'Cycle (λ={lam})')

plt.title('HP Filter Cycles with Different λ')
plt.xlabel('Date')
plt.ylabel('Log GDP Cycle Component')
plt.legend()
plt.grid(True)
plt.show()