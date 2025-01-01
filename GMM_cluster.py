import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


# 数据准备
def generate_data():
    np.random.seed(42)
    data1 = np.random.normal(loc=0, scale=1, size=300)
    data2 = np.random.normal(loc=5, scale=1.5, size=300)
    data3 = np.random.normal(loc=10, scale=1, size=300)
    data = np.concatenate([data1, data2, data3])
    data = data.reshape(-1, 1)
    return data


data = generate_data()

# 数据标准化
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# 选择初始成分数
n_components = np.arange(1, 10)

# 使用 BIC 和 AIC 选择最佳成分数
bic_scores = []
aic_scores = []
gmm_models = []

for n in n_components:
    gmm = GaussianMixture(n_components=n, random_state=42)
    gmm.fit(data_scaled)
    bic_scores.append(gmm.bic(data_scaled))
    aic_scores.append(gmm.aic(data_scaled))
    gmm_models.append(gmm)

# 找到最优成分数
best_n_bic = n_components[np.argmin(bic_scores)]
best_n_aic = n_components[np.argmin(aic_scores)]

print(f'Optimal number of components by BIC: {best_n_bic}')
print(f'Optimal number of components by AIC: {best_n_aic}')

# 训练最佳模型
best_gmm = gmm_models[np.argmin(bic_scores)]

# 可视化结果
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(n_components, bic_scores, label='BIC')
plt.xlabel('Number of components')
plt.ylabel('BIC Score')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(n_components, aic_scores, label='AIC')
plt.xlabel('Number of components')
plt.ylabel('AIC Score')
plt.legend()

plt.tight_layout()
plt.show()


# 可视化 GMM 拟合结果
def plot_gmm(data, gmm, scaler):
    plt.figure(figsize=(10, 5))
    x = np.linspace(np.min(data), np.max(data), 1000).reshape(-1, 1)
    x_scaled = scaler.transform(x)
    log_prob = gmm.score_samples(x_scaled)
    pdf = np.exp(log_prob)

    plt.hist(data, bins=30, density=True, alpha=0.6, color='g', label='Data histogram')
    plt.plot(x, pdf, '-k', label='GMM fit')

    for i in range(gmm.n_components):
        mean = gmm.means_[i][0]
        variance = gmm.covariances_[i][0][0]
        component_pdf = np.exp(-(x_scaled - mean) ** 2 / (2 * variance)) / np.sqrt(2 * np.pi * variance)
        plt.plot(x, component_pdf, '--', label=f'Component {i + 1}')

    plt.xlabel('Data values')
    plt.ylabel('Density')
    plt.legend()
    plt.title('GMM Fit and Components')
    plt.show()


plot_gmm(data, best_gmm, scaler)
