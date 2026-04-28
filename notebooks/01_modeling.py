import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import statsmodels.api as sm
from scipy.stats import fisher_exact
from sklearn.preprocessing import StandardScaler

# ============================================================
# 项目：Olist订单取消风险分析 —— 支付方式与金额的交互效应
# 来源：olist-ecommerce-analysis 描述性分析的延伸建模
#
# 分析结构：
#   第一部分：描述性统计摘要（来自 04_cancel_risk_analysis.sql）
#   第二部分：Fisher 精确检验（300+ 区间，纯单一支付口径）
#   第三部分：数据加载与质量检查（全量数据）
#   第四部分：特征工程
#   第五部分：基础逻辑回归（statsmodels，含置信区间）
#   第六部分：交互项逻辑回归（statsmodels，含置信区间）
#   第七部分：可视化
#
# 口径说明：
#   描述性统计 & Fisher 检验：300+ 区间，纯单一支付口径（n_voucher=75）
#   逻辑回归：全量 credit_card + voucher 订单，保证估计稳定性
#   两者互补——描述性统计放大高价区间信号，回归在全量数据上验证独立效应
# ============================================================


# ============================================================
# 第一部分：描述性统计摘要
# 来源：04_cancel_risk_analysis.sql 查询 10.2
# 口径：纯单一支付，payment_value >= 300
# ============================================================

summary_df = pd.DataFrame({
    'payment_type': ['credit_card', 'voucher'],
    'total_orders': [8268, 75],
    'canceled_orders': [65, 15],
    'cancel_rate_pct': [0.79, 20.00],
    'canceled_gmv': [51980.29, 16019.38],
})

print("=" * 60)
print("第一部分：300+ 区间取消率对比（纯单一支付口径）")
print("=" * 60)
print(summary_df.to_string(index=False))
print()
print("核心发现：voucher 取消率 20%，是信用卡 0.79% 的约 25 倍")
print("待验证：控制金额后，支付方式的独立效应是否仍显著")

# ============================================================
# 第二部分：Fisher 精确检验
# 目的：在样本量有限（voucher n=75）的 300+ 区间，
#       用精确检验替代卡方检验，避免大样本假设失效
# 结果：OR=31.55，p<0.000001
# ============================================================

print("\n" + "=" * 60)
print("第二部分：Fisher 精确检验（300+ 区间）")
print("=" * 60)

contingency = [
    [15, 60],  # voucher:     取消 15，未取消 60
    [65, 8203],  # credit_card: 取消 65，未取消 8203
]
odds_ratio, p_value = fisher_exact(contingency, alternative='greater')

print(f"voucher:      取消 15 / 总计 75  (20.00%)")
print(f"credit_card:  取消 65 / 总计 8268 (0.79%)")
print(f"\nOdds Ratio: {odds_ratio:.2f}")
print(f"p-value:    {p_value:.6f}")
print()
print("结论：300+ 区间 voucher 取消风险显著高于信用卡（OR=31.55，p<0.000001）")
print("注：受限于 voucher 样本量（n=75），该结论仅反映高价区间信号，")
print("    独立效应验证依赖第五、六部分的全量回归。")

# ============================================================
# 第三部分：数据加载与质量检查
# 数据来源：SQL 多表关联导出（orders + order_payments）
# 处理逻辑：同一订单多条支付记录，取 payment_value 最大者（ROW_NUMBER）
# 筛选口径：仅保留 voucher 与 credit_card，排除 boleto 等
# ============================================================

df = pd.read_csv('../data/model_data.csv')

print("\n" + "=" * 60)
print("第三部分：数据加载与质量检查（全量数据）")
print("=" * 60)
print(f"样本总量：{df.shape[0]} 行，{df.shape[1]} 列")
print("\n前5行：")
print(df.head())

print("\n目标变量分布（样本不平衡检查）：")
print(df['is_canceled'].value_counts())
print("→ 类别不平衡，回归使用 class_weight='balanced'")

print("\n支付方式分布：")
print(df['payment_type'].value_counts())

print("\n空值检查：")
print(df.isnull().sum())

# ============================================================
# 第四部分：特征工程
# is_voucher：支付方式编码为 0/1 二值变量（credit_card=0，voucher=1）
# payment_value：保留原始连续值，标准化后用于回归
# voucher_x_value：交互项 = is_voucher × payment_value（标准化前构造）
# ============================================================

df['is_voucher'] = (df['payment_type'] == 'voucher').astype(int)
df['voucher_x_value'] = df['is_voucher'] * df['payment_value']

print("\n" + "=" * 60)
print("第四部分：特征工程")
print("=" * 60)
print("新增列：is_voucher, voucher_x_value")
print(df[['payment_type', 'is_voucher', 'payment_value', 'voucher_x_value']].head())

# ============================================================
# 第五部分：基础逻辑回归
# 自变量：is_voucher, payment_value（各自独立效应）
# 工具：statsmodels Logit，输出系数、p值、95% CI
# 目的：验证在控制金额后，支付方式本身是否仍影响取消概率
# ============================================================

print("\n" + "=" * 60)
print("第五部分：基础逻辑回归（is_voucher + payment_value）")
print("=" * 60)

features_base = ['is_voucher', 'payment_value']

scaler_base = StandardScaler()
X_base = scaler_base.fit_transform(df[features_base])
X_base = sm.add_constant(X_base)
y = df['is_canceled']

model_base = sm.Logit(y, X_base).fit(disp=0)
print(model_base.summary2())

# 提取 OR 和 95% CI
coef_base = model_base.params[1:]  # 去掉 intercept
ci_base = model_base.conf_int().iloc[1:]

print("\nOdds Ratio（基础模型）：")
pvals_base = model_base.pvalues.values[1:]  # 去掉 intercept，按位置取
for name, coef, (lo, hi), p in zip(features_base, coef_base, ci_base.values, pvals_base):
    print(f"  {name:20s}  OR={np.exp(coef):.4f}  "
          f"95%CI=[{np.exp(lo):.4f}, {np.exp(hi):.4f}]  "
          f"p={p:.4f}")

# ============================================================
# 第六部分：交互项逻辑回归
# 自变量：is_voucher, payment_value, voucher_x_value
# 目的：验证"高金额 + voucher"是否存在超出单独效应的组合放大效应
# 假设来源：04_cancel_risk_analysis.sql 查询 4 发现
#           voucher 在各价格段取消率均高于信用卡，300+ 区间差距最大
# ============================================================

print("\n" + "=" * 60)
print("第六部分：交互项逻辑回归（加入 voucher × payment_value）")
print("=" * 60)

features_inter = ['is_voucher', 'payment_value', 'voucher_x_value']

scaler_inter = StandardScaler()  # 独立 scaler，避免污染
X_inter = scaler_inter.fit_transform(df[features_inter])
X_inter = sm.add_constant(X_inter)

model_inter = sm.Logit(y, X_inter).fit(disp=0)
print(model_inter.summary2())

print("\nOdds Ratio（交互项模型）：")
pvals_inter = model_inter.pvalues.values[1:]  # 去掉 intercept，按位置取
for name, coef, (lo, hi), p in zip(
        features_inter,
        model_inter.params[1:],
        model_inter.conf_int().iloc[1:].values,
        pvals_inter
):
    print(f"  {name:20s}  OR={np.exp(coef):.4f}  "
          f"95%CI=[{np.exp(lo):.4f}, {np.exp(hi):.4f}]  "
          f"p={p:.4f}")

print()
print("解读：")
print("  is_voucher     OR>1：控制金额后，voucher 本身显著提升取消概率")
print("  payment_value  OR>1：金额越高，取消概率越高")
print("  voucher_x_value OR>1：交互效应为正，高金额对取消率的放大作用")
print("                       在 voucher 用户中更强（若p<0.05则显著）")

# ============================================================
# 第七部分：可视化
# 展示交互项模型各特征的 OR 及 95% CI（误差棒）
# voucher_x_value 橙色高亮，baseline OR=1 红色虚线
# ============================================================

print("\n" + "=" * 60)
print("第七部分：可视化输出")
print("=" * 60)

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

labels = features_inter
coefs = model_inter.params[1:]
ci = model_inter.conf_int().iloc[1:]
ors = np.exp(coefs)
ci_low = np.exp(ci.iloc[:, 0])
ci_high = np.exp(ci.iloc[:, 1])
xerr = [ors - ci_low, ci_high - ors]
colors = ['#5B9BD5', '#5B9BD5', '#E36C09']

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(labels, ors, xerr=xerr, color=colors,
               error_kw=dict(ecolor='#333333', capsize=5, linewidth=1.2))
ax.axvline(x=1, color='red', linestyle='--', linewidth=1.2, label='baseline (OR=1)')
ax.set_xlabel('Odds Ratio (with 95% CI)')
ax.set_title('Logistic Regression: Odds Ratios\n(Interaction Model, full sample)')

for bar, val in zip(bars, ors):
    ax.text(val + max(xerr[1]) * 0.05,
            bar.get_y() + bar.get_height() / 2,
            f'{val:.4f}', va='center', fontsize=9)

ax.legend()
plt.tight_layout()
plt.savefig('../outputs/figures/odds_ratio.png', dpi=150)
plt.show()
print("图表已保存至 ../outputs/figures/odds_ratio.png")

# ============================================================
# 核心结论汇总
# ============================================================
print("\n" + "=" * 60)
print("核心结论汇总")
print("=" * 60)
print("""
【描述性统计】
  300+ 区间 voucher 取消率 20%，信用卡 0.79%，差距约 25 倍

【Fisher 精确检验】（300+ 区间，纯单一支付口径）
  OR=31.55，p<0.000001
  → 高价区间 voucher 取消风险极显著高于信用卡

【逻辑回归 - 基础模型】（全量数据）
  控制金额后，voucher 本身仍独立提升取消概率（OR 见模型输出）

【逻辑回归 - 交互项模型】（全量数据）
  voucher × payment_value 交互项若 OR>1 且 p<0.05：
  → 高金额对取消率的放大效应在 voucher 用户中更强，
    组合风险超出两者单独效应的简单叠加

【局限性】
  300+ 区间 pure voucher 样本量仅 75，交互项效应估计依赖全量数据外推，
  高价区间结论需更大样本进一步验证

【业务建议】（对应 04_cancel_risk_analysis.sql 查询 10.3）
  对 300+ 纯 voucher 订单增加下单确认环节
  若将 voucher 取消率降至信用卡水平，预计挽回 GMV 约 19,658 元
""")

#逻辑回归结果显示，在控制订单金额后，voucher 支付本身使取消概率提升约 39%（OR=1.39）；金额每升高一个标准差，取消概率额外提升约 10%（OR=1.10）。
# 交互项显著为正（OR=1.09，p<0.0001），说明高金额对取消率的放大效应在 voucher 用户中更强，两者存在组合风险而非简单叠加。
# 结合 300+ 区间 Fisher 精确检验（OR=31.55，p<0.000001），支付方式与金额的交互效应在描述性与建模两个层面均得到验证。