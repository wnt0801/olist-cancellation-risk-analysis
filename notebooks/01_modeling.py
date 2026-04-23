import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 描述性统计背景（来自第一个项目的发现）
summary = {
    'payment_type': ['credit_card', 'voucher'],
    'total_orders': [8268, 75],
    'canceled_orders': [65, 15],
    'cancel_rate': [0.79, 20.00],
    'canceled_gmv': [51980.29, 16019.38]
}


summary_df = pd.DataFrame(summary)
print("=== 300+区间支付方式取消率对比 ===")
print(summary_df.to_string(index=False))
print("\n以上描述性统计无法排除金额本身的影响，以下用逻辑回归验证交互效应。\n")

df = pd.read_csv('../data/model_data.csv')
print(df.shape)
print(df.head())                                                                                          # 78126 行数据


# 看一下正负样本分布
print(df['is_canceled'].value_counts()) #
print(df['payment_type'].value_counts())

# 检查有没有空值
print(df.isnull().sum())                                                                                         # 0 空值


# 构建特征
df['is_voucher'] = (df['payment_type'] == 'voucher').astype(int)

X = df[['is_voucher', 'payment_value']]
y = df['is_canceled']

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 建模，处理样本不平衡
model = LogisticRegression(class_weight='balanced')
model.fit(X_scaled, y)

# 输出系数
for name, coef in zip(['is_voucher', 'payment_value'], model.coef_[0]):
    print(f'{name}: {coef:.4f}  odds_ratio: {np.exp(coef):.4f}')


# 加交互项
df['voucher_x_value'] = df['is_voucher'] * df['payment_value']

X2 = df[['is_voucher', 'payment_value', 'voucher_x_value']]
X2_scaled = scaler.fit_transform(X2)

model2 = LogisticRegression(class_weight='balanced')
model2.fit(X2_scaled, y)

for name, coef in zip(['is_voucher', 'payment_value', 'voucher_x_value'], model2.coef_[0]):
    print(f'{name}: {coef:.4f}  odds_ratio: {np.exp(coef):.4f}')



matplotlib.rcParams['font.family'] = 'DejaVu Sans'

labels = ['is_voucher', 'payment_value', 'voucher_x_value']
odds_ratios = [1.1932, 1.1595, 1.2514]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(labels, odds_ratios, color=['#5B9BD5', '#5B9BD5', '#E36C09'])
ax.axvline(x=1, color='red', linestyle='--', linewidth=1, label='baseline (OR=1)')
ax.set_xlabel('Odds Ratio')
ax.set_title('Logistic Regression: Odds Ratios\n(with interaction term)')

for bar, val in zip(bars, odds_ratios):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center')

ax.legend()
plt.tight_layout()
plt.savefig('../outputs/figures/odds_ratio.png', dpi=150)
plt.show()
