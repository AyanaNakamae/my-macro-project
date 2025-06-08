import matplotlib
matplotlib.rcParams['font.family'] = ['Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fredapi import Fred
from statsmodels.tsa.filters.hp_filter import hpfilter

# 1. データの取得
fred = Fred(api_key='8dfaf30a8f87d5731d37b87fb46c4839')

# カナダのGDPデータ
print("カナダのデータを取得中...")
canada_gdp = fred.get_series('NGDPRSAXDCCAQ')
canada_gdp = canada_gdp.dropna()
print(f"カナダのデータ期間: {canada_gdp.index[0]} から {canada_gdp.index[-1]}")

# 日本のGDPデータ
print("日本のデータを取得中...")
japan_gdp = fred.get_series('JPNRGDPEXP')
japan_gdp = japan_gdp.dropna()
print(f"日本のデータ期間: {japan_gdp.index[0]} から {japan_gdp.index[-1]}")

# 2. データ期間を合わせる
common_start = max(canada_gdp.index.min(), japan_gdp.index.min())
common_end = min(canada_gdp.index.max(), japan_gdp.index.max())

canada_gdp_aligned = canada_gdp[common_start:common_end]
japan_gdp_aligned = japan_gdp[common_start:common_end]

print(f"\n共通分析期間: {common_start} から {common_end}")
print(f"カナダのデータ点数: {len(canada_gdp_aligned)}")
print(f"日本のデータ点数: {len(japan_gdp_aligned)}")

# 3. 対数変換
log_canada_gdp = np.log(canada_gdp_aligned)
log_japan_gdp = np.log(japan_gdp_aligned)

# 4. HP-filterの適用（λ=1600を使用）
print("\nHP-filterを適用中...")
canada_trend, canada_cycle = hpfilter(log_canada_gdp, lamb=1600)
japan_trend, japan_cycle = hpfilter(log_japan_gdp, lamb=1600)

# 5. ステップ4：統計分析
print("\n" + "="*50)
print("ステップ4：統計分析")
print("="*50)

# 標準偏差の計算
canada_cycle_std = canada_cycle.std()
japan_cycle_std = japan_cycle.std()

print(f"カナダの循環変動成分の標準偏差: {canada_cycle_std:.4f}")
print(f"日本の循環変動成分の標準偏差: {japan_cycle_std:.4f}")

# 標準偏差の比較
std_ratio = canada_cycle_std / japan_cycle_std
print(f"\n標準偏差の比率（カナダ/日本）: {std_ratio:.3f}")

if std_ratio > 1:
    print(f"→ カナダの景気変動は日本より {std_ratio:.2f}倍大きい")
else:
    print(f"→ 日本の景気変動はカナダより {1/std_ratio:.2f}倍大きい")

# 相関係数の計算
correlation = canada_cycle.corr(japan_cycle)
print(f"\nカナダと日本の循環変動成分の相関係数: {correlation:.4f}")

# 相関の解釈
if correlation > 0.7:
    correlation_strength = "非常に強い正の相関"
elif correlation > 0.5:
    correlation_strength = "強い正の相関"
elif correlation > 0.3:
    correlation_strength = "中程度の正の相関"
elif correlation > 0.1:
    correlation_strength = "弱い正の相関"
elif correlation > -0.1:
    correlation_strength = "ほとんど相関なし"
elif correlation > -0.3:
    correlation_strength = "弱い負の相関"
else:
    correlation_strength = "中程度以上の負の相関"

print(f"→ {correlation_strength}")

# 6. ステップ5：循環変動成分の比較グラフ
print("\n" + "="*50)
print("ステップ5：グラフ作成")
print("="*50)

plt.figure(figsize=(15, 10))

# 上段：循環変動成分の時系列比較
plt.subplot(2, 2, 1)
plt.plot(canada_cycle.index, canada_cycle, label='カナダ', color='red', linewidth=2)
plt.plot(japan_cycle.index, japan_cycle, label='日本', color='blue', linewidth=2)
plt.title('循環変動成分の比較（カナダ vs 日本）', fontsize=12, fontweight='bold')
plt.ylabel('循環変動成分')
plt.xlabel('年')
plt.legend()
plt.grid(True, alpha=0.3)

# 右上：散布図（相関関係の可視化）
plt.subplot(2, 2, 2)
plt.scatter(canada_cycle, japan_cycle, alpha=0.6, color='purple', s=30)
plt.xlabel('カナダの循環変動成分')
plt.ylabel('日本の循環変動成分')
plt.title(f'相関関係\n(相関係数: {correlation:.3f})', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

# 回帰直線を追加
z = np.polyfit(canada_cycle, japan_cycle, 1)
p = np.poly1d(z)
plt.plot(canada_cycle, p(canada_cycle), "r--", alpha=0.8, linewidth=2)

# 下段左：トレンド成分の比較（参考）
plt.subplot(2, 2, 3)
plt.plot(canada_trend.index, canada_trend, label='カナダ（トレンド）', color='lightcoral', linewidth=2)
plt.plot(japan_trend.index, japan_trend, label='日本（トレンド）', color='lightblue', linewidth=2)
plt.title('トレンド成分の比較（参考）', fontsize=12, fontweight='bold')
plt.ylabel('対数実質GDP（トレンド）')
plt.xlabel('年')
plt.legend()
plt.grid(True, alpha=0.3)

# 下段右：標準偏差の比較（棒グラフ）
plt.subplot(2, 2, 4)
countries = ['カナダ', '日本']
std_values = [canada_cycle_std, japan_cycle_std]
bars = plt.bar(countries, std_values, color=['red', 'blue'], alpha=0.7)
plt.title('循環変動成分の標準偏差比較', fontsize=12, fontweight='bold')
plt.ylabel('標準偏差')

# 棒グラフに数値を表示
for bar, value in zip(bars, std_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
             f'{value:.4f}', ha='center', va='bottom', fontsize=10)

plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# 7. 分析結果のサマリー
print("\n" + "="*60)
print("                   分析結果サマリー")
print("="*60)
print(f"分析対象: カナダ vs 日本")
print(f"分析期間: {common_start.strftime('%Y-%m')} ～ {common_end.strftime('%Y-%m')}")
print(f"データ点数: {len(canada_cycle)}期間")
print()
print("【標準偏差による変動の大きさ比較】")
print(f"  カナダ: {canada_cycle_std:.4f}")
print(f"  日本:   {japan_cycle_std:.4f}")
print(f"  比率:   {std_ratio:.3f} (カナダ/日本)")
print()
print("【相関分析】")
print(f"  相関係数: {correlation:.4f}")
print(f"  相関の強さ: {correlation_strength}")
print()
print("【経済学的解釈】")
if abs(correlation) > 0.5:
    print("  ✓ 両国の景気循環は高い同調性を示している")
    print("  ✓ 一方の国の景気が良くなると、他方も良くなる傾向")
elif abs(correlation) > 0.3:
    print("  ✓ 両国の景気循環は中程度の同調性を示している")
    print("  ✓ ある程度連動するが、独立した要因も存在")
else:
    print("  ✓ 両国の景気循環の同調性は低い")
    print("  ✓ それぞれ独立した景気変動パターンを持つ")

if std_ratio > 1.2:
    print(f"  ✓ カナダの方が景気変動が大きく、より不安定")
elif std_ratio < 0.8:
    print(f"  ✓ 日本の方が景気変動が大きく、より不安定")
else:
    print(f"  ✓ 両国の景気変動の大きさは似ている")

print(f"\nプログラム完了：統計分析とグラフ作成が正常に実行されました。")
