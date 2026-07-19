import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans

n_samples = 500

# 生成31个簇的中心
cluster_centers = np.random.randn(31, 2)

# 使用KMeans算法生成聚类数据
kmeans = KMeans(n_clusters=31, init=cluster_centers, n_init=1)
data, labels = kmeans.fit_transform(np.random.randn(n_samples, 2)), kmeans.labels_

figure = plt.figure(dpi=100)
sns.set_style("white")
# 获取当前坐标轴对象
ax = plt.gca()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.spines['top'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['right'].set_visible(True)

col = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF", "#FFA07A", "#800080", "#32CD32", "#000080",
       "#FFD700", "#A0522D", "#008080", "#9ACD32", "#FFC0CB", "#20B2AA", "#D2691E", "#7B68EE", "#87CEEB", "#6A5ACD",
       "#C71585", "#40E0D0", "#F08080", "#DA70D6", "#FF1493", "#1E90FF", "#8B0000", "#00FF7F", "#3CB371", "#B22222",
       "#00CED1", "#E6E6FA"]

sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=labels, palette=sns.color_palette(col, 31),
                s=5, legend=False)

# 显示图表
plt.show()
