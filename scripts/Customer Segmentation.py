import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer

#region Importazionie
file_path = r"C:\Users\JoaquimFrancalanci\OneDrive - ITS Angelo Rizzoli\Desktop\MachineLearning\shopping_behavior_updated.csv"
df = pd.read_csv(file_path)

print(df.head())

#Group customers based on their behaviors for targeted strategies
columns_for_clustering = ['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases', 'Frequency of Purchases']

numerical_features = [col for col in columns_for_clustering if df[col].dtype in ["int64", "float64"]]
categorical_features = [col for col in columns_for_clustering if col not in numerical_features]

encoder = OneHotEncoder(sparse_output=False)
encoded_categorical_features = encoder.fit_transform(df[categorical_features])

data_for_clustering = pd.concat([df[numerical_features], pd.DataFrame(encoded_categorical_features, columns=encoder.get_feature_names_out(categorical_features))], axis=1)
data_for_clustering.head()

scaler = StandardScaler()
scaler.fit(data_for_clustering)
data_for_clustering = scaler.transform(data_for_clustering)





