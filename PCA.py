#region Importazioni
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

#region Preparazione
# Caricamento dati
file_path = r"C:\Users\JoaquimFrancalanci\OneDrive - ITS Angelo Rizzoli\Desktop\MachineLearning\shopping_behavior_updated.csv"
df = pd.read_csv(file_path)

# Rimuovere duplicati
df = df.drop_duplicates()

# Rimuovere il campo 'Customer ID'
df = df.drop(columns=['Customer ID'])

# Separare le caratteristiche (X) e la variabile target (y)
X = df.drop(columns=['Purchase Amount (USD)'])
y = df['Purchase Amount (USD)']

# Codifica delle variabili categoriche
categorical_cols = X.select_dtypes(include=['object']).columns
for col in categorical_cols:
    X[col] = X[col].astype('category').cat.codes

print(f"Dataset after encoding: {X.shape}")

#region PCA 
# Standardizzazione delle feature
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Applicazione della PCA
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"Number of components selected by PCA: {pca.n_components_}")
print(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_)}")

# Divisione dei dati in training e test set
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

print(f"Shape of training data after PCA: {X_train.shape}")

#region Modello
# Inizializzazione del modello Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Addestramento del modello
rf_model.fit(X_train, y_train)

# Previsioni sui dati di test
y_pred = rf_model.predict(X_test)

# Valutazione del modello
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Model Performance after PCA:\nMAE: {mae}\nR²: {r2}")

# Importanza delle feature
feature_importances = variable_importance(rf_model)
print_var_importance(feature_importances['importance'], feature_importances['index'], X.columns)
variable_importance_plot(feature_importances['importance'], feature_importances['index'], X.columns)
