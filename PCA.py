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

#region Preparazione Dati
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
