import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
from mpl_toolkits.mplot3d import Axes3D

#region Importazione
file_path = r"C:\Users\JoaquimFrancalanci\OneDrive - ITS Angelo Rizzoli\Desktop\MachineLearning\shopping_behavior_updated.csv"
df = pd.read_csv(file_path)

df.info()

#region ==========
# Clustering Informazioni Cliente
columns_for_clustering = ['Age', 'Gender', 'Purchase Amount (USD)', 'Size']

# Divisione Variabili Categoriche e Numeriche
numerical_features = [col for col in columns_for_clustering if df[col].dtype in ["int64", "float64"]]
categorical_features = [col for col in columns_for_clustering if col not in numerical_features]

# Conversione variabili categoriche in numeriche
encoder = OneHotEncoder(sparse_output=False)
encoded_categorical_features = encoder.fit_transform(df[categorical_features])

# Unione Varibili numeriche con quelle trasformate
data_for_clustering = pd.concat([df[numerical_features], pd.DataFrame(encoded_categorical_features, columns=encoder.get_feature_names_out(categorical_features))], axis=1)

data_for_clustering.info()

data_for_clustering.head()

scaler = StandardScaler()
scaler.fit(data_for_clustering)
data_for_clustering = scaler.transform(data_for_clustering)

#region kmeans Client
def perform_kmeans(data, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data)
    silhouette_avg = silhouette_score(data, cluster_labels)
    return silhouette_avg, cluster_labels

# Numeri di custer che voglio ottenere
num_clusters = 4

# "Accuratezza" KMeans
kmeans_silhouette, kmeans_labels = perform_kmeans(data_for_clustering, num_clusters)
df['KMeans_Cluster'] = kmeans_labels
print(f"KMeans clustering silhouette score: {kmeans_silhouette}")

#region Hierarchical client
def perform_hierarchical_clustering(data, num_clusters):
    hierarchical = AgglomerativeClustering(n_clusters=num_clusters)
    cluster_labels = hierarchical.fit_predict(data)
    silhouette_avg = silhouette_score(data, cluster_labels)
    return silhouette_avg, cluster_labels

# "Accuratezza" hierarchical
hierarchical_silhouette, hierarchical_labels = perform_hierarchical_clustering(data_for_clustering, num_clusters)
df['Hierarchical_Cluster'] = hierarchical_labels
print(f"Hierarchical clustering silhouette score: {hierarchical_silhouette}")

#region DBSCAN client
def perform_dbscan(data, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(data)
    silhouette_avg = silhouette_score(data, cluster_labels)
    return silhouette_avg, cluster_labels

# "Accuratezza" DBSCAN
dbscan_silhouette, dbscan_labels = perform_dbscan(data_for_clustering, eps=0.5, min_samples=5)
df['DBSCAN_Cluster'] = dbscan_labels
print(f"DBSCA clustering silhouette score: {dbscan_silhouette}")

#region Client 2D
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.scatter(df['Age'], df['Purchase Amount (USD)'], c=df['KMeans_Cluster'], cmap='viridis')
plt.title('K-Means Clustering')

plt.subplot(132)
plt.scatter(df['Age'], df['Purchase Amount (USD)'], c=df['Hierarchical_Cluster'], cmap='viridis')
plt.title('Hierarchical Clustering')

plt.subplot(133)
plt.scatter(df['Age'], df['Purchase Amount (USD)'], c=df['DBSCAN_Cluster'], cmap='viridis')
plt.title('DBSCAN Clustering')

plt.show()

#region Client 3D
fig = plt.figure(figsize=(15, 5))

ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(df['Age'], df['Purchase Amount (USD)'], df['KMeans_Cluster'], c=df['KMeans_Cluster'], cmap='viridis')
ax1.set_title('K-Means Clustering')
ax1.set_xlabel('Age')
ax1.set_ylabel('Purchase Amount (USD)')
ax1.set_zlabel('Cluster')

ax2 = fig.add_subplot(132, projection='3d')
ax2.scatter(df['Age'], df['Purchase Amount (USD)'], df['Hierarchical_Cluster'], c=df['Hierarchical_Cluster'], cmap='viridis')
ax2.set_title('Hierarchical Clustering')
ax2.set_xlabel('Age')
ax2.set_ylabel('Purchase Amount (USD)')
ax2.set_zlabel('Cluster')

ax3 = fig.add_subplot(133, projection='3d')
ax3.scatter(df['Age'], df['Purchase Amount (USD)'], df['DBSCAN_Cluster'], c=df['DBSCAN_Cluster'], cmap='viridis')
ax3.set_title('DBSCAN Clustering')
ax3.set_xlabel('Age')
ax3.set_ylabel('Purchase Amount (USD)')
ax3.set_zlabel('Cluster')

plt.show()

#region ==========
# Clustering Rispetto all'acquisto
columns_for_clustering = ['Payment Method', 'Frequency of Purchases','Purchase Amount (USD)', 'Discount Applied', 'Promo Code Used']

# Divisione Variabili Categoriche e Numeriche
numerical_features = [col for col in columns_for_clustering if df[col].dtype in ["int64", "float64"]]
categorical_features = [col for col in columns_for_clustering if col not in numerical_features]

# conversione categoriche in numeriche
encoder = OneHotEncoder(sparse_output=False)
encoded_categorical_features = encoder.fit_transform(df[categorical_features])

# Unione Varibili numeriche con quelle trasformate
data_for_clustering = pd.concat([df[numerical_features], pd.DataFrame(encoded_categorical_features, columns=encoder.get_feature_names_out(categorical_features))], axis=1)

data_for_clustering.info()

data_for_clustering.head()

scaler = StandardScaler()
scaler.fit(data_for_clustering)
data_for_clustering = scaler.transform(data_for_clustering)

#region Kmeans purchase
def perform_kmeans(data, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data)
    silhouette_avg = silhouette_score(data, cluster_labels)
    return silhouette_avg, cluster_labels

# Numeri di custer che voglio ottenere
num_clusters = 4

# "Accuratezza" KMeans
kmeans_silhouette, kmeans_labels = perform_kmeans(data_for_clustering, num_clusters)
df['KMeans_Cluster'] = kmeans_labels
print(f"KMeans clustering silhouette score: {kmeans_silhouette}")

#region Hierachical purchase
def perform_hierarchical_clustering(data, num_clusters):
    hierarchical = AgglomerativeClustering(n_clusters=num_clusters)
    cluster_labels = hierarchical.fit_predict(data)
    silhouette_avg = silhouette_score(data, cluster_labels)
    return silhouette_avg, cluster_labels

# "Accuratezza" hierarchical
hierarchical_silhouette, hierarchical_labels = perform_hierarchical_clustering(data_for_clustering, num_clusters)
df['Hierarchical_Cluster'] = hierarchical_labels
print(f"Hierarchical clustering silhouette score: {hierarchical_silhouette}")

#region DBSCAN purchase
def perform_dbscan(data, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(data)
    silhouette_avg = silhouette_score(data, cluster_labels)
    return silhouette_avg, cluster_labels

# "Accuratezza" DBSCAN
dbscan_silhouette, dbscan_labels = perform_dbscan(data_for_clustering, eps=0.5, min_samples=5)
df['DBSCAN_Cluster'] = dbscan_labels
print(f"DBSCA clustering silhouette score: {dbscan_silhouette}")

#region Purchase 2D
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.scatter(df['Age'], df['Purchase Amount (USD)'], c=df['KMeans_Cluster'], cmap='viridis')
plt.title('K-Means Clustering')

plt.subplot(132)
plt.scatter(df['Age'], df['Purchase Amount (USD)'], c=df['Hierarchical_Cluster'], cmap='viridis')
plt.title('Hierarchical Clustering')

plt.subplot(133)
plt.scatter(df['Age'], df['Purchase Amount (USD)'], c=df['DBSCAN_Cluster'], cmap='viridis')
plt.title('DBSCAN Clustering')

plt.show()

#region Purchase 3D
fig = plt.figure(figsize=(15, 5))

ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(df['Age'], df['Purchase Amount (USD)'], df['KMeans_Cluster'], c=df['KMeans_Cluster'], cmap='viridis')
ax1.set_title('K-Means Clustering')
ax1.set_xlabel('Age')
ax1.set_ylabel('Purchase Amount (USD)')
ax1.set_zlabel('Cluster')

ax2 = fig.add_subplot(132, projection='3d')
ax2.scatter(df['Age'], df['Purchase Amount (USD)'], df['Hierarchical_Cluster'], c=df['Hierarchical_Cluster'], cmap='viridis')
ax2.set_title('Hierarchical Clustering')
ax2.set_xlabel('Age')
ax2.set_ylabel('Purchase Amount (USD)')
ax2.set_zlabel('Cluster')

ax3 = fig.add_subplot(133, projection='3d')
ax3.scatter(df['Age'], df['Purchase Amount (USD)'], df['DBSCAN_Cluster'], c=df['DBSCAN_Cluster'], cmap='viridis')
ax3.set_title('DBSCAN Clustering')
ax3.set_xlabel('Age')
ax3.set_ylabel('Purchase Amount (USD)')
ax3.set_zlabel('Cluster')

plt.show()