import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import scipy.cluster.hierarchy as sch

df = pd.read_csv('Mall_Customers.csv')

X = df.iloc[:, [3, 4]].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- K-MEANS CLUSTERING ---

# A. Tìm K tối ưu bằng phương pháp Elbow
wcss = [] # Within-Cluster Sum of Square
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Vẽ biểu đồ Elbow
plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), wcss, marker='o', color='red')
plt.title('Phương pháp Elbow (Tìm điểm khuỷu tay)')
plt.xlabel('Số lượng cụm (K)')
plt.ylabel('WCSS')
plt.show()

# B. Giả sử điểm khuỷu tay là K=5
k_opt = 5
kmeans = KMeans(n_clusters=k_opt, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# C. Vẽ kết quả phân cụm
plt.figure(figsize=(8, 6))
# Vẽ các điểm dữ liệu theo màu cụm
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_kmeans, s=50, cmap='viridis', label='Data Points')
# Vẽ tâm cụm (Centroids)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X', label='Centroids')
plt.title('Phân cụm khách hàng (K-Means)')
plt.xlabel('Annual Income (Scaled)')
plt.ylabel('Spending Score (Scaled)')
plt.legend()
plt.show()
