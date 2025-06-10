import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# OECD諸国のリスト
countries = [
    'Australia', 'Austria', 'Belgium', 'Canada', 'Denmark', 'Finland',
    'France', 'Germany', 'Greece', 'Iceland', 'Ireland', 'Italy',
    'Japan', 'Netherlands', 'New Zealand', 'Norway', 'Portugal',
    'Spain', 'Sweden', 'Switzerland', 'United Kingdom', 'United States'
]

# サンプルデータ生成（実際の分析では、OECD StatやPenn World Tableなどのデータを使用）
def generate_sample_data():
    """
    サンプルデータを生成
    実際の分析では、以下のデータが必要：
    - 実質GDP (Y)
    - 資本ストック (K) 
    - 労働投入 (L)
    - 投資 (I)
    """
    np.random.seed(42)
    data = {}
    
    years = list(range(1990, 2020))
    
    for country in countries:
        # 各国の特性を反映した初期値設定
        base_gdp = np.random.uniform(20000, 60000)
        base_capital = base_gdp * np.random.uniform(2.5, 4.0)
        base_labor = np.random.uniform(5, 80) * 1000000  # millions
        
        # 成長率のパラメータ（国によって異なる）
        gdp_growth = np.random.uniform(0.015, 0.035)
        capital_growth = np.random.uniform(0.02, 0.045)
        labor_growth = np.random.uniform(0.005, 0.015)
        
        # 時系列データ生成
        gdp_series = [base_gdp * (1 + gdp_growth + np.random.normal(0, 0.01)) ** i for i in range(30)]
        capital_series = [base_capital * (1 + capital_growth + np.random.normal(0, 0.015)) ** i for i in range(30)]
        labor_series = [base_labor * (1 + labor_growth + np.random.normal(0, 0.005)) ** i for i in range(30)]
        
        data[country] = pd.DataFrame({
            'year': years,
            'gdp': gdp_series,
            'capital': capital_series,
            'labor': labor_series
        })
    
    return data

def calculate_growth_rates(series):
    """成長率を計算"""
    return [(series[i+1] - series[i]) / series[i] for i in range(len(series)-1)]

def estimate_capital_share(gdp, capital, labor):
    """
    資本分配率αを推定
    コブ・ダグラス生産関数 Y = A * K^α * L^(1-α) を用いて
    対数線形回帰により推定
    """
    # 対数変換
    log_gdp = np.log(gdp)
    log_capital = np.log(capital)
    log_labor = np.log(labor)
    
    # 回帰分析: log(Y) = log(A) + α*log(K) + (1-α)*log(L)
    # これを log(Y) = log(A) + α*log(K) + β*log(L) として推定し、α + β ≈ 1 を仮定
    
    X = np.column_stack([np.ones(len(log_gdp)), log_capital, log_labor])
    try:
        # 最小二乗法
        coefficients = np.linalg.lstsq(X, log_gdp, rcond=None)[0]
        alpha = coefficients[1]
        beta = coefficients[2]
        
        # 規模の収益一定を仮定してαを調整
        alpha_adjusted = alpha / (alpha + beta)
        return max(0.2, min(0.5, alpha_adjusted))  # 現実的な範囲に制限
    except:
        return 0.35  # デフォルト値

def growth_accounting_analysis(data):
    """成長会計分析を実行"""
    results = []
    
    for country, df in data.items():
        # データの期間を1990-2019に限定
        df_period = df[(df['year'] >= 1990) & (df['year'] <= 2019)].copy()
        
        if len(df_period) < 2:
            continue
            
        # 成長率計算
        gdp_growth = calculate_growth_rates(df_period['gdp'].values)
        capital_growth = calculate_growth_rates(df_period['capital'].values)
        labor_growth = calculate_growth_rates(df_period['labor'].values)
        
        # 平均成長率
        avg_gdp_growth = np.mean(gdp_growth) * 100  # パーセント表示
        avg_capital_growth = np.mean(capital_growth) * 100
        avg_labor_growth = np.mean(labor_growth) * 100
        
        # 資本分配率の推定
        alpha = estimate_capital_share(df_period['gdp'].values, 
                                     df_period['capital'].values, 
                                     df_period['labor'].values)
        
        # 成長会計の分解
        # ΔY/Y = ΔA/A + α(ΔK/K) + (1-α)(ΔL/L)
        # TFP成長率 = GDP成長率 - α*資本成長率 - (1-α)*労働成長率
        
        capital_contribution = alpha * avg_capital_growth
        labor_contribution = (1 - alpha) * avg_labor_growth
        tfp_growth = avg_gdp_growth - capital_contribution - labor_contribution
        
        # Capital Deepening = α * (資本成長率 - 労働成長率)
        capital_deepening = alpha * (avg_capital_growth - avg_labor_growth)
        
        # シェア計算
        tfp_share = tfp_growth / avg_gdp_growth if avg_gdp_growth != 0 else 0
        capital_share = capital_contribution / avg_gdp_growth if avg_gdp_growth != 0 else 0
        
        results.append({
            'Country': country,
            'Growth Rate': round(avg_gdp_growth, 2),
            'TFP Growth': round(tfp_growth, 2),
            'Capital Deepening': round(capital_deepening, 2),
            'TFP Share': round(tfp_share, 2),
            'Capital Share': round(capital_share, 2),
            'Alpha': round(alpha, 2)
        })
    
    return pd.DataFrame(results)

def format_table(df):
    """表の体裁を整える"""
    # 結果をソート（国名のアルファベット順）
    df_sorted = df.sort_values('Country').copy()
    
    # 平均行を追加
    avg_row = {
        'Country': 'Average',
        'Growth Rate': round(df_sorted['Growth Rate'].mean(), 2),
        'TFP Growth': round(df_sorted['TFP Growth'].mean(), 2),
        'Capital Deepening': round(df_sorted['Capital Deepening'].mean(), 2),
        'TFP Share': round(df_sorted['TFP Share'].mean(), 2),
        'Capital Share': round(df_sorted['Capital Share'].mean(), 2),
        'Alpha': round(df_sorted['Alpha'].mean(), 2)
    }
    
    df_final = pd.concat([df_sorted, pd.DataFrame([avg_row])], ignore_index=True)
    
    return df_final

def print_formatted_table(df):
    """元のTable 5.1形式で表を出力"""
    print("\nTable 5.1")
    print("Growth Accounting in OECD Countries: 1990-2019")
    print("-" * 85)
    
    # ヘッダー行
    print(f"{'Country':<15} {'Growth Rate':<12} {'TFP Growth':<12} {'Capital Deepening':<17} {'TFP Share':<10} {'Capital Share':<13}")
    print("-" * 85)
    
    # データ行
    for _, row in df.iterrows():
        if row['Country'] == 'Average':
            print("-" * 85)
        
        print(f"{row['Country']:<15} {row['Growth Rate']:<12.2f} {row['TFP Growth']:<12.2f} "
              f"{row['Capital Deepening']:<17.2f} {row['TFP Share']:<10.2f} {row['Capital Share']:<13.2f}")

def main():
    """メイン関数"""
    print("Growth Accounting Analysis (1990-2019)")
    print("=" * 60)
    
    # データ生成
    print("1. Generating sample data...")
    data = generate_sample_data()
    
    # 成長会計分析
    print("2. Performing growth accounting analysis...")
    results = growth_accounting_analysis(data)
    
    # 表の整形
    print("3. Formatting results...")
    final_table = format_table(results)
    
    # 結果表示（Table 5.1形式）
    print_formatted_table(final_table)
    
    # 統計サマリー
    print(f"\n統計サマリー:")
    print(f"対象国数: {len(final_table) - 1}")  # 平均行を除く
    print(f"平均GDP成長率: {final_table[final_table['Country'] == 'Average']['Growth Rate'].iloc[0]:.2f}%")
    print(f"平均TFP成長率: {final_table[final_table['Country'] == 'Average']['TFP Growth'].iloc[0]:.2f}%")
    print(f"平均資本分配率(α): {final_table[final_table['Country'] == 'Average']['Alpha'].iloc[0]:.2f}")
    
    # CSVファイルに保存
    final_table.to_csv('growth_accounting_1990_2019.csv', index=False)
    print(f"\n結果を 'growth_accounting_1990_2019.csv' に保存しました。")
    
    return final_table

# 実行
if __name__ == "__main__":
    results_table = main()
    
    # 元のTable 5.1との比較のため、データを表示
    print("\n" + "="*60)
    print("参考: 元のTable 5.1の構造")
    print("="*60)
    print("各列の意味:")
    print("- Growth Rate: 実質GDP成長率 (%)")
    print("- TFP Growth: 全要素生産性成長率 (%)")
    print("- Capital Deepening: 資本深化の寄与 (%)")
    print("- TFP Share: 成長に占めるTFPの割合")
    print("- Capital Share: 成長に占める資本の割合")
    print("\n成長会計の恒等式:")
    print("GDP成長率 = TFP成長率 + α×資本成長率 + (1-α)×労働成長率")
    print("ここで、α は資本分配率（通常0.3-0.4）")
    
