import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Funzioni di utilità
def print_dx_perc(data_frame, col):
    if col in data_frame:
        uni_target = data_frame[col].value_counts()
        df_len = len(data_frame)
        for idx, count in uni_target.iteritems():
            perc = round((count / df_len) * 100, ndigits=2)
            print(f'{idx} accounts for {perc}% of the {col} column, equal to {count} occurrences on {df_len}')
    else:
        print(f'Column {col} is not in the dataframe')

def variable_importance(fit):
    try:
        if not hasattr(fit, 'fit'):
            return print("'{0}' is not an instantiated model from scikit-learn".format(fit))

        if not hasattr(fit, 'feature_importances_'):
            return print("Model does not have feature_importances_ attribute.")
    except KeyError:
        print("Model entered does not contain 'estimators_' attribute.")

    importances = fit.feature_importances_
    indices = np.argsort(importances)[::-1]
    return {'importance': importances, 'index': indices}

def print_var_importance(importance, indices, names_index):
    print("Feature ranking:")
    for f in range(len(indices)):
        print("{0}. The feature '{1}' has a Mean Decrease in Impurity of {2:.5f}"
              .format(f + 1, names_index[indices[f]], importance[indices[f]]))

def variable_importance_plot(importance, indices, names_index):
    importance_desc = [importance[i] for i in indices]
    feature_space = [names_index[i] for i in indices]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('#fafafa')
    plt.title('Feature Importances for Random Forest Model')
    plt.barh(range(len(importance_desc)), importance_desc, align="center", color='#875FDB')
    plt.yticks(range(len(importance_desc)), feature_space)
    plt.xlabel('Mean Decrease in Impurity')
    plt.ylabel('Feature')
    plt.gca().invert_yaxis()
    plt.show()
    plt.close()

# Caricare il dataset
file_path = r"C:\Users\JoaquimFrancalanci\OneDrive - ITS Angelo Rizzoli\Desktop\MachineLearning\shopping_behavior_updated.csv"
df = pd.read_csv(file_path)

# Visualizzare le prime righe del dataset
print("Dataset Head:")
print(df.head())

# Rimuovere il campo 'Customer ID' e 'Previous Purchases'
df = df.drop(columns=['Customer ID', 'Previous Purchases'])
print(df.dtypes)

# Separare le caratteristiche (X) e la variabile target (y)
X = df.drop(columns=['Purchase Amount (USD)'])
y = df['Purchase Amount (USD)']

# Codifica variabili categoriche
categorical_cols = X.select_dtypes(include=['object']).columns
print("Categorical columns:", categorical_cols)

for col in categorical_cols:
    X[col] = X[col].astype('category').cat.codes

print("Dataset after encoding:")
print(X.head())

# Divisione dei dati in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Shape of training data:")
print(X_train.shape)

# Inizializziamo il modello
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Addestriamo il modello
rf_model.fit(X_train, y_train)

# Previsioni sui dati di test
y_pred = rf_model.predict(X_test)


# Valutazione del modello
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Initial Model Performance:")
print(f"MAE: {mae}")
print(f"R²: {r2}") 

# Importanza delle feature
feature_importances = variable_importance(rf_model)
print_var_importance(feature_importances['importance'], feature_importances['index'], X.columns)
variable_importance_plot(feature_importances['importance'], feature_importances['index'], X.columns)

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

# Addestriamo il modello con i migliori parametri
best_rf_model = grid_search.best_estimator_
y_pred_optimized = best_rf_model.predict(X_test)

# Valutazione del modello ottimizzato
mae_optimized = mean_absolute_error(y_test, y_pred_optimized)
r2_optimized = r2_score(y_test, y_pred_optimized)

print("Optimized Model Performance:")
print(f"MAE: {mae_optimized}")
print(f"R²: {r2_optimized}")

# Aggiungere le predizioni suddivise per sesso e località
df['Predicted'] = rf_model.predict(X)
grouped_df = df.groupby(['Gender', 'Location']).agg({'Purchase Amount (USD)': 'mean', 'Predicted': 'mean'}).reset_index()

print(grouped_df)

# Arrotondare le predizioni a 3 cifre decimali
grouped_df['Predicted'] = grouped_df['Predicted'].round(3)

# Ordinare le location in ordine alfabetico2
grouped_df = grouped_df.sort_values(by='Location')

# Visualizzare la tabella delle predizioni per sesso e località
print("Table of Predicted Values by Gender and Location:")

# Suddividere la tabella delle predizioni in due grafici distinti
grouped_df_male = grouped_df[grouped_df['Gender'] == 'Male']
grouped_df_female = grouped_df[grouped_df['Gender'] == 'Female']

# Determinare i limiti comuni dell'asse y
y_min = min(grouped_df['Purchase Amount (USD)'].min(), grouped_df['Predicted'].min())
y_max = max(grouped_df['Purchase Amount (USD)'].max(), grouped_df['Predicted'].max())

# Funzione per creare il grafico combinato
def plot_combined(data, title, y_min, y_max):
    fig, ax1 = plt.subplots(figsize=(12, 8))

    sns.barplot(data=data, x='Location', y='Purchase Amount (USD)', color='green', ax=ax1)
    ax1.set_ylabel('Actual Purchase Amount (USD)')
    ax1.set_xlabel('Location')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    ax1.set_ylim(y_min, y_max)

    ax2 = ax1.twinx()
    sns.lineplot(data=data, x='Location', y='Predicted', color='red', marker='o', ax=ax2)
    ax2.set_ylabel('Predicted Purchase Amount (USD)')
    ax2.set_ylim(y_min, y_max)

    plt.title(title)
    fig.tight_layout()
    plt.show()

# Visualizzare le predizioni per i maschi
print("Table of Predicted Values for Males by Location:")
print(grouped_df_male)
plot_combined(grouped_df_male, 'Predicted vs Actual Purchase Amounts for Males by Location', 0, y_max + 1)

# Visualizzare le predizioni per le femmine
print("Table of Predicted Values for Females by Location:")
print(grouped_df_female)
plot_combined(grouped_df_female, 'Predicted vs Actual Purchase Amounts for Females by Location', 0, y_max + 1)


#Visualizzare differenza predizione attuale
Male_dif_pred = grouped_df_male['Predicted'] - grouped_df_male['Purchase Amount (USD)']
print(Male_dif_pred)

plt.figure(figsize=(9, 3))

plt.subplot(131)
plt.bar(grouped_df_male['Location'], Male_dif_pred)
plt.subplot(132)
plt.suptitle('Prediction differents')
plt.show()















