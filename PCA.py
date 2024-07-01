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

#region GridSearchCV
# Ottimizzazione degli iperparametri con GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)

# Migliori parametri trovati
best_params = grid_search.best_params_
print("Best Parameters from GridSearchCV:")
print(best_params)

# Addestramento del modello con i migliori parametri
best_rf_model = grid_search.best_estimator_
y_pred_optimized = best_rf_model.predict(X_test)

# Valutazione del modello ottimizzato
mae_optimized = mean_absolute_error(y_test, y_pred_optimized)
r2_optimized = r2_score(y_test, y_pred_optimized)

print("Optimized Model Performance:")
print(f"MAE: {mae_optimized}")
print(f"R²: {r2_optimized}")

#region Risultati
# Visualizzazione dei risultati
df['Predicted'] = rf_model.predict(X_scaled)
grouped_df = df.groupby(['Gender', 'Location']).agg({'Purchase Amount (USD)': 'mean', 'Predicted': 'mean'}).reset_index()

grouped_df['Predicted'] = grouped_df['Predicted'].round(3)
grouped_df = grouped_df.sort_values(by='Location')

print("Table of Predicted Values by Gender and Location:")
print(grouped_df)

# Ciclo per generare grafici per maschi e femmine
for gender in ['Male', 'Female']:
    print(f"Table of Predicted Values for {gender}s by Location:")
    grouped_df_gender = grouped_df[grouped_df['Gender'] == gender]
    print(grouped_df_gender)
    plot_combined(grouped_df_gender, f'Predicted vs Actual Purchase Amounts for {gender}s by Location', 0, grouped_df['Purchase Amount (USD)'].max() + 1)

    dif_pred = grouped_df_gender['Predicted'] - grouped_df_gender['Purchase Amount (USD)']
    print(dif_pred)

    plt.figure(figsize=(12, 8))
    plt.bar(grouped_df_gender['Location'], dif_pred)
    plt.suptitle(f'Prediction differences for {gender}s')
    plt.show()

    try:
        median = np.median(dif_pred)
        q1 = np.percentile(dif_pred, 25)
        q3 = np.percentile(dif_pred, 75)
        iqr = q3 - q1
        lower_whisker = q1 - 1.5 * iqr
        upper_whisker = q3 + 1.5 * iqr

        non_outlier_mask = (dif_pred >= lower_whisker) & (dif_pred <= upper_whisker)
        non_outliers = dif_pred[non_outlier_mask]
        min_val = np.min(non_outliers)
        max_val = np.max(non_outliers)
        outliers = dif_pred[~non_outlier_mask]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        box = ax.boxplot(dif_pred, patch_artist=True, showfliers=True)
        colors = ['#FF6F61']
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
        for flier in box['fliers']:
            flier.set(marker='o', color='red', alpha=0.5)

        ax.text(1.1, median, f'Median: {median:.2f}', horizontalalignment='center', verticalalignment='center',
                fontsize=12, bbox=dict(facecolor='white', edgecolor='black'))
        ax.text(1.1, q1, f'Q1: {q1:.2f}', horizontalalignment='center', verticalalignment='center',
                fontsize=12, bbox=dict(facecolor='white', edgecolor='black'))
        ax.text(1.1, q3, f'Q3: {q3:.2f}', horizontalalignment='center', verticalalignment='center',
                fontsize=12, bbox=dict(facecolor='white', edgecolor='black'))
        ax.text(1.1, lower_whisker, f'Lower Whisker: {lower_whisker:.2f}', horizontalalignment='center',
                verticalalignment='center', fontsize=12, bbox=dict(facecolor='white', edgecolor='black'))
        ax.text(1.1, upper_whisker, f'Upper Whisker: {upper_whisker:.2f}', horizontalalignment='center',
                verticalalignment='center', fontsize=12, bbox=dict(facecolor='white', edgecolor='black'))
        ax.text(1.1, min_val, f'Min: {min_val:.2f}', horizontalalignment='center', verticalalignment='center',
                fontsize=12, bbox=dict(facecolor='white', edgecolor='black'))
        ax.text(1.1, max_val, f'Max: {max_val:.2f}', horizontalalignment='center', verticalalignment='center',
                fontsize=12, bbox=dict(facecolor='white', edgecolor='black'))

        ax.axhline(y=median, color='blue', linestyle='--', linewidth=1.5)
        ax.set_ylabel('Values')
        ax.set_title(f'Difference in {gender} Predictions')
        ax.grid(True, linestyle='--', alpha=0.7)

        plt.show()
    except Exception as e:
        print(f"Box Plot Error: {e}")


