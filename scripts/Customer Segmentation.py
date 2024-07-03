import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Importazione del dataset
file_path = r"C:\Users\JoaquimFrancalanci\OneDrive - ITS Angelo Rizzoli\Desktop\MachineLearning\shopping_behavior_updated.csv"
df = pd.read_csv(file_path)

print(df.head())

# Definizione delle colonne da usare per la clusterizzazione
columns_for_clustering = ['Age', 'Purchase Amount (USD)', 'Review Rating', 
                          'Previous Purchases', 'Frequency of Purchases']

# Separazione delle colonne numeriche e categoriche
numerical_features = [col for col in columns_for_clustering if df[col].dtype in ["int64", "float64"]]
categorical_features = [col for col in columns_for_clustering if col not in numerical_features]

# One-Hot Encoding delle variabili categoriche
encoder = OneHotEncoder(sparse_output=False)
encoded_categorical_features = encoder.fit_transform(df[categorical_features])

# Unione delle colonne numeriche e delle colonne codificate
data_for_clustering = pd.concat([df[numerical_features], pd.DataFrame(encoded_categorical_features, columns=encoder.get_feature_names_out(categorical_features))], axis=1)

# Standardizzazione dei dati
scaler = StandardScaler()
data_for_clustering = scaler.fit_transform(data_for_clustering)

# Funzione per eseguire il K-Means
def perform_kmeans(data, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data)
    silhouette_avg = silhouette_score(data, cluster_labels)
    return silhouette_avg, cluster_labels

# K-Means Clustering
num_clusters = 3
kmeans_silhouette, kmeans_labels = perform_kmeans(data_for_clustering, num_clusters)
df['KMeans_Cluster'] = kmeans_labels

print(f"K-Means clustering silhouette score: {kmeans_silhouette}")

# Funzione per eseguire il clustering gerarchico
def perform_hierarchical_clustering(data, num_clusters):
    hierarchical = AgglomerativeClustering(n_clusters=num_clusters)
    cluster_labels = hierarchical.fit_predict(data)
    silhouette_avg = silhouette_score(data, cluster_labels)
    return silhouette_avg, cluster_labels

# Agglomerative Clustering
hierarchical_silhouette, hierarchical_labels = perform_hierarchical_clustering(data_for_clustering, num_clusters)
df['Hierarchical_Cluster'] = hierarchical_labels

print(f"Hierarchical clustering silhouette score: {hierarchical_silhouette}")

# Funzione per eseguire DBSCAN
def perform_dbscan(data, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(data)
    # Calcolare il silhouette score solo se ci sono almeno due cluster validi
    if len(set(cluster_labels)) > 1 and -1 in cluster_labels:
        silhouette_avg = silhouette_score(data[cluster_labels != -1], cluster_labels[cluster_labels != -1])
    else:
        silhouette_avg = silhouette_score(data, cluster_labels)
    return silhouette_avg, cluster_labels

# DBSCAN Clustering
dbscan_silhouette, dbscan_labels = perform_dbscan(data_for_clustering, eps=0.5, min_samples=5)
df['DBSCAN_Cluster'] = dbscan_labels

print(f"DBSCAN clustering silhouette score: {dbscan_silhouette}")

# Visualizzazione dei risultati dei clustering

plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.scatter(df['Age'], df['Purchase Amount (USD)'], c=df['KMeans_Cluster'], cmap='viridis')
plt.title('K-Means Clustering')
plt.xlabel('Age')
plt.ylabel('Purchase Amount (USD)')

plt.subplot(132)
plt.scatter(df['Age'], df['Purchase Amount (USD)'], c=df['Hierarchical_Cluster'], cmap='viridis')
plt.title('Hierarchical Clustering')
plt.xlabel('Age')
plt.ylabel('Purchase Amount (USD)')

plt.subplot(133)
plt.scatter(df['Age'], df['Purchase Amount (USD)'], c=df['DBSCAN_Cluster'], cmap='viridis')
plt.title('DBSCAN Clustering')
plt.xlabel('Age')
plt.ylabel('Purchase Amount (USD)')

plt.show()

# Visualizzazione 3D dei clustering
fig = plt.figure(figsize=(15, 5))

ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(df['Age'], df['Purchase Amount (USD)'], df['Review Rating'], c=df['KMeans_Cluster'], cmap='viridis')
ax1.set_title('K-Means Clustering')
ax1.set_xlabel('Age')
ax1.set_ylabel('Purchase Amount (USD)')
ax1.set_zlabel('Review Rating')

ax2 = fig.add_subplot(132, projection='3d')
ax2.scatter(df['Age'], df['Purchase Amount (USD)'], df['Review Rating'], c=df['Hierarchical_Cluster'], cmap='viridis')
ax2.set_title('Hierarchical Clustering')
ax2.set_xlabel('Age')
ax2.set_ylabel('Purchase Amount (USD)')
ax2.set_zlabel('Review Rating')

ax3 = fig.add_subplot(133, projection='3d')
ax3.scatter(df['Age'], df['Purchase Amount (USD)'], df['Review Rating'], c=df['DBSCAN_Cluster'], cmap='viridis')
ax3.set_title('DBSCAN Clustering')
ax3.set_xlabel('Age')
ax3.set_ylabel('Purchase Amount (USD)')
ax3.set_zlabel('Review Rating')

plt.show()

# Visualizzazione 3D dei clustering con etichetta del cluster come asse Z
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
