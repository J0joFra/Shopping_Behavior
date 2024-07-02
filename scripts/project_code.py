#region Importazioni
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

#region Funzioni
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

#region Analisi
file_path = r"C:\Users\JoaquimFrancalanci\OneDrive - ITS Angelo Rizzoli\Desktop\MachineLearning\shopping_behavior_updated.csv"
df = pd.read_csv(file_path)

df.describe()
df.info()
df.drop_duplicates()

df = df.drop(columns=['Customer ID']) #Rimuovere il campo 'Customer ID'
X = df.drop(columns=['Purchase Amount (USD)'])
y = df['Purchase Amount (USD)']

# Codifica variabili categoriche
categorical_cols = X.select_dtypes(include=['object']).columns
print("Categorical columns:", categorical_cols)

for col in categorical_cols:
    X[col] = X[col].astype('category').cat.codes

# Divisione dei dati in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Shape of training data: {X_train.shape}")

# Inizializziamo il modello
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train) # Addestriamo il modello
y_pred = rf_model.predict(X_test)

# Valutazione del modello
mae = mean_absolute_error(y_test, y_pred) 
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

#Gender Male or Female
for gender in ['Male', 'Female']:
    print(f"Table of Predicted Values for {gender}s by Location:")
    grouped_df_gender = grouped_df[grouped_df['Gender'] == gender]
    print(grouped_df_gender)
    plot_combined(grouped_df_gender, f'Predicted vs Actual Purchase Amounts for {gender}s by Location', 0, grouped_df['Purchase Amount (USD)'].max() + 1)

    # Calcolare la differenza di predizione attuale
    dif_pred = grouped_df_gender['Predicted'] - grouped_df_gender['Purchase Amount (USD)']
    print(dif_pred)

    plt.figure(figsize=(12, 8))
    plt.bar(grouped_df_gender['Location'], dif_pred)
    plt.suptitle(f'Prediction differences for {gender}s')
    plt.show()

    # Box plot
    try:
        # Calcolare i valori di mediana, Q1, Q3 e gli estremi
        median = np.median(dif_pred)
        q1 = np.percentile(dif_pred, 25)
        q3 = np.percentile(dif_pred, 75)
        iqr = q3 - q1
        lower_whisker = q1 - 1.5 * iqr
        upper_whisker = q3 + 1.5 * iqr

        # Identificare minimi e massimi che non siano outlier
        non_outlier_mask = (dif_pred >= lower_whisker) & (dif_pred <= upper_whisker)
        non_outliers = dif_pred[non_outlier_mask]
        min_val = np.min(non_outliers)
        max_val = np.max(non_outliers)
        outliers = dif_pred[~non_outlier_mask]
        print(outliers)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        box = ax.boxplot(dif_pred, patch_artist=True, showfliers=True)
        colors = ['#FF6F61']
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
        for flier in box['fliers']:
            flier.set(marker='o', color='red', alpha=0.5) #outliers

        # Mediana, Q1, Q3, whisker inferiori e superiori, massimo e minimo
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

        #Linea tratteggiata per la mediana
        ax.axhline(y=median, color='blue', linestyle='--', linewidth=1.5)
        ax.set_ylabel('Values')
        ax.set_title(f'Difference in {gender} Predictions')
        ax.grid(True, linestyle='--', alpha=0.7)

        plt.show()
    except Exception as e:
        print(f"Box Plot Error: {e}")

#region Linear Regression
# Suddivisione del dataset in base al genere
df_male = df[df['Gender'] == 'Male']
df_female = df[df['Gender'] == 'Female']

def regression_plot(df, title):
    np.random.seed(0)
    X = df.drop(columns=['Purchase Amount (USD)', 'Gender'])
    y = df['Purchase Amount (USD)']
    
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        X[col] = X[col].astype('category').cat.codes

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Scatter plot di valori effettivi vs previsti
    plt.scatter(y_test, y_pred, color="black")
    
    max_val = max(max(y_test), max(y_pred))
    min_val = min(min(y_test), min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
    
    plt.xlabel('Actual Purchase Amount (USD)')
    plt.ylabel('Predicted Purchase Amount (USD)')
    plt.title(title)
    plt.show()

regression_plot(df_male, 'Actual vs Predicted Purchase Amounts (Male)')
regression_plot(df_female, 'Actual vs Predicted Purchase Amounts (Female)')
