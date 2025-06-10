# 統合GDP分析プログラム：カナダと日本の景気循環比較
# Step 1.2, Step 3, Step 4.5を統合

import matplotlib
matplotlib.rcParams['font.family'] = ['Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fredapi import Fred
from statsmodels.tsa.filters.hp_filter import hpfilter

# ===== 設定 =====
FRED_API_KEY = '8dfaf30a8f87d5731d37b87fb46c4839'
LAMBDAS = [10, 100, 1600]  # HPフィルターのパラメータ

print("="*70)
print("     統合GDP分析プログラム：カナダと日本の景気循環比較")
print("="*70)

# ===== STEP 1: データ取得 =====
print("\nSTEP 1: データ取得")
print("-" * 30)

fred = Fred(api_key=FRED_API_KEY)

# カナダのGDPデータ取得
print("カナダのGDPデータを取得中...")
canada_gdp = fred.get_series('NGDPRSAXDCCAQ')
canada_gdp = canada_gdp.dropna()
print(f"カナダのデータ期間: {canada_gdp.index[0]} から {canada_gdp.index[-1]}")
print(f"カナダのデータ点数: {len(canada_gdp)}")

# 日本のGDPデータ取得
print("日本のGDPデータを取得中...")
japan_gdp = fred.get_series('JPNRGDPEXP')
japan_gdp = japan_gdp.dropna()
print(f"日本のデータ期間: {japan_gdp.index[0]} から {japan_gdp.index[-1]}")
print(f"日本のデータ点数: {len(japan_gdp)}")

# データ期間を合わせる
common_start = max(canada_gdp.index.min(), japan_gdp.index.min())
common_end = min(canada_gdp.index.max(), japan_gdp.index.max())

canada_gdp_aligned = canada_gdp[common_start:common_end]
japan_gdp_aligned = japan_gdp[common_start:common_end]

print(f"\n共通分析期間: {common_start} から {common_end}")
print(f"共通期間でのデータ点数: {len(canada_gdp_aligned)}")

# ===== STEP 2: 対数変換 =====
print("\nSTEP 2: 対数変換")
print("-" * 30)

log_canada_gdp = np.log(canada_gdp_aligned)
log_japan_gdp = np.log(japan_gdp_aligned)

print("対数変換完了")
print(f"カナダ対数GDP - 最初の5データ:")
print(log_canada_gdp.head())
print(f"日本対数GDP - 最初の5データ:")
print(log_japan_gdp.head())

# ===== STEP 3: HPフィルター適用（複数のλ値） =====
print("\nSTEP 3: HPフィルター適用")
print("-" * 30)

# カナダ用の辞書
canada_trends = {}
canada_cycles = {}

# 日本用の辞書
japan_trends = {}
japan_cycles = {}

print("HPフィルターを適用中...")
for lam in LAMBDAS:
    print(f"  λ = {lam} で処理中...")
    
    # カナダ
    canada_trend, canada_cycle = hpfilter(log_canada_gdp, lamb=lam)
    canada_trends[lam] = canada_trend
    canada_cycles[lam] = canada_cycle
    
    # 日本
    japan_trend, japan_cycle = hpfilter(log_japan_gdp, lamb=lam)
    japan_trends[lam] = japan_trend
    japan_cycles[lam] = japan_cycle

print("HPフィルター適用完了")

# ===== STEP 4: 可視化（個別国分析） =====
print("\nSTEP 4: 個別国分析の可視化")
print("-" * 30)

# カナダの分析
print("カナダの分析グラフを作成中...")
plt.figure(figsize=(15, 10))

# グラフ1：カナダの元データとトレンド成分
plt.subplot(2, 2, 1)
plt.plot(log_canada_gdp.index, log_canada_gdp, label='Original Log GDP', color='blue', linewidth=2)
for lam in LAMBDAS:
    plt.plot(canada_trends[lam].index, canada_trends[lam], label=f'Trend (λ={lam})', linewidth=2)
plt.title('カナダ：元データ vs HPフィルタートレンド', fontsize=12, fontweight='bold')
plt.xlabel('年')
plt.ylabel('対数GDP')
plt.legend()
plt.grid(True, alpha=0.3)

# グラフ2：カナダの循環成分
plt.subplot(2, 2, 2)
for lam in LAMBDAS:
    plt.plot(canada_cycles[lam].index, canada_cycles[lam], label=f'Cycle (λ={lam})', linewidth=2)
plt.title('カナダ：HPフィルター循環成分', fontsize=12, fontweight='bold')
plt.xlabel('年')
plt.ylabel('循環成分')
plt.legend()
plt.grid(True, alpha=0.3)

# グラフ3：日本の元データとトレンド成分
plt.subplot(2, 2, 3)
plt.plot(log_japan_gdp.index, log_japan_gdp, label='Original Log GDP', color='blue', linewidth=2)
for lam in LAMBDAS:
    plt.plot(japan_trends[lam].index, japan_trends[lam], label=f'Trend (λ={lam})', linewidth=2)
plt.title('日本：元データ vs HPフィルタートレンド', fontsize=12, fontweight='bold')
plt.xlabel('年')
plt.ylabel('対数GDP')
plt.legend()
plt.grid(True, alpha=0.3)

# グラフ4：日本の循環成分
plt.subplot(2, 2, 4)
for lam in LAMBDAS:
    plt.plot(japan_cycles[lam].index, japan_cycles[lam], label=f'Cycle (λ={lam})', linewidth=2)
plt.title('日本：HPフィルター循環成分', fontsize=12, fontweight='bold')
plt.xlabel('年')
plt.ylabel('循環成分')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ===== STEP 5: 統計分析（λ=1600を使用） =====
print("\nSTEP 5: 統計分析（λ=1600）")
print("-" * 30)

# λ=1600の結果を使用（四半期データの標準値）
canada_cycle_main = canada_cycles[1600]
japan_cycle_main = japan_cycles[1600]
canada_trend_main = canada_trends[1600]
japan_trend_main = japan_trends[1600]

# 標準偏差の計算
canada_cycle_std = canada_cycle_main.std()
japan_cycle_std = japan_cycle_main.std()

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
correlation = canada_cycle_main.corr(japan_cycle_main)
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

# ===== STEP 6: 比較分析の可視化 =====
print("\nSTEP 6: カナダ・日本比較分析の可視化")
print("-" * 30)

plt.figure(figsize=(15, 10))

# グラフ1：循環変動成分の時系列比較
plt.subplot(2, 2, 1)
plt.plot(canada_cycle_main.index, canada_cycle_main, label='カナダ', color='red', linewidth=2)
plt.plot(japan_cycle_main.index, japan_cycle_main, label='日本', color='blue', linewidth=2)
plt.title('循環変動成分の比較（カナダ vs 日本）', fontsize=12, fontweight='bold')
plt.ylabel('循環変動成分')
plt.xlabel('年')
plt.legend()
plt.grid(True, alpha=0.3)

# 重要な景気後退期にマークを追加
plt.axvspan('2008-01-01', '2009-12-31', alpha=0.2, color='gray', label='金融危機')
plt.axvspan('2020-01-01', '2020-12-31', alpha=0.2, color='yellow', label='コロナ危機')

# グラフ2：散布図（相関関係）
plt.subplot(2, 2, 2)
plt.scatter(canada_cycle_main, japan_cycle_main, alpha=0.6, color='purple', s=30)
plt.xlabel('カナダの循環変動成分')
plt.ylabel('日本の循環変動成分')
plt.title(f'相関関係\n(相関係数: {correlation:.3f})', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

# 回帰直線を追加
z = np.polyfit(canada_cycle_main, japan_cycle_main, 1)
p = np.poly1d(z)
plt.plot(canada_cycle_main, p(canada_cycle_main), "r--", alpha=0.8, linewidth=2)

# グラフ3：トレンド成分の比較
plt.subplot(2, 2, 3)
plt.plot(canada_trend_main.index, canada_trend_main, label='カナダ（トレンド）', color='lightcoral', linewidth=2)
plt.plot(japan_trend_main.index, japan_trend_main, label='日本（トレンド）', color='lightblue', linewidth=2)
plt.title('トレンド成分の比較', fontsize=12, fontweight='bold')
plt.ylabel('対数実質GDP（トレンド）')
plt.xlabel('年')
plt.legend()
plt.grid(True, alpha=0.3)

# グラフ4：標準偏差の比較
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

# ===== STEP 7: 単独の時系列比較グラフ =====
print("\nSTEP 7: 詳細な時系列比較グラフ")
print("-" * 30)

plt.figure(figsize=(12, 6))
plt.plot(canada_cycle_main.index, canada_cycle_main, label='カナダ', color='red', linewidth=2)
plt.plot(japan_cycle_main.index, japan_cycle_main, label='日本', color='blue', linewidth=2)

plt.title('循環変動成分の時系列比較（カナダ vs 日本）', fontsize=14, fontweight='bold')
plt.xlabel('年', fontsize=12)
plt.ylabel('循環変動成分', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# 重要な景気後退期にマークを追加
plt.axvspan('2008-01-01', '2009-12-31', alpha=0.2, color='gray', label='金融危機')
plt.axvspan('2020-01-01', '2020-12-31', alpha=0.2, color='yellow', label='コロナ危機')

plt.tight_layout()
plt.show()

# ===== STEP 8: 分析結果のサマリー =====
print("\n" + "="*70)
print("                      最終分析結果サマリー")
print("="*70)
print(f"分析対象: カナダ vs 日本")
print(f"分析期間: {common_start.strftime('%Y-%m')} ～ {common_end.strftime('%Y-%m')}")
print(f"データ点数: {len(canada_cycle_main)}期間")
print(f"使用したλ値: {LAMBDAS}")
print(f"主要分析に使用したλ: 1600（四半期データの標準値）")
print()

print("【HPフィルター分析結果】")
print("複数のλ値での分析:")
for lam in LAMBDAS:
    print(f"  λ={lam}: カナダ標準偏差={canada_cycles[lam].std():.4f}, 日本標準偏差={japan_cycles[lam].std():.4f}")

print()
print("【メイン分析（λ=1600）】")
print("標準偏差による変動の大きさ比較:")
print(f"  カナダ: {canada_cycle_std:.4f}")
print(f"  日本:   {japan_cycle_std:.4f}")
print(f"  比率:   {std_ratio:.3f} (カナダ/日本)")
print()
print("相関分析:")
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

print()
print("【実行されたステップ】")
print("  Step 1: データ取得とアライメント")
print("  Step 2: 対数変換")
print("  Step 3: 複数λ値でのHPフィルター適用")
print("  Step 4: 個別国分析の可視化")
print("  Step 5: 統計分析（λ=1600）")
print("  Step 6: 比較分析の可視化")
print("  Step 7: 詳細時系列比較")
print("  Step 8: 統合分析サマリー")

print(f"\n{'='*70}")
print("統合GDP分析プログラム完了：すべてのステップが正常に実行されました。")
print(f"{'='*70}")
