# Dataset Analysis

## Dataset
Il dataset principale utilizzato in questo progetto è `shopping_behavior_updated.csv`. Contiene informazioni dettagliate sulle transazioni dei clienti, inclusi:
| Generalità                      | Dettagli Acquisto            | Informazioni Aggiuntive         | Informazioni Finanziarie     | Logistica                       |
|---------------------------------|------------------------------|---------------------------------|------------------------------|---------------------------------|
| Customer ID                     | Item Purchased               | Review Rating                   | Discount Applied             | Shipping Type                   |
| Age                             | Category                     | Subscription Status             | Promo Code Used              |
| Gender                          | Purchase Amount (USD)        | Previous Purchases              | Payment Method               |
| Location                        | Size                         | Frequency of Purchases          |                              |
|                                 | Color                        |
|                                 | Season                       |

## Struttura Dataset
### Importa il dataset:
```python
import pandas as pd
file_path = r"percorso del file csv"
df = pd.read_csv(file_path)
```

### Visualizza le prime righe del dataset:
```python
print(df.head())
```
Output:
```
   Customer ID Item Purchased  ... Promo Code Used Payment Method
0         1001        Shirt A  ...            YES        Credit
1         1002        Shirt B  ...             NO         Debit
2         1003        Shirt A  ...            YES        Credit
3         1004        Shirt C  ...             NO         Debit
4         1005        Shirt A  ...            YES        Credit

[5 rows x 10 columns]
```

### Ottieni una descrizione statistica del dataset:
```python
print(df.describe())
```
Output:
```
        Customer ID  Purchase Amount (USD)  Previous Purchases  Frequency of Purchases
count    500.000000             500.000000          500.000000              500.000000
mean    1237.500000             105.500000            2.670000                3.742000
std      144.481833              56.562805            1.717098                2.253511
min     1001.000000              10.000000            0.000000                1.000000
25%     1119.750000              57.750000            1.000000                2.000000
50%     1237.500000             105.500000            3.000000                3.000000
75%     1355.250000             153.250000            4.000000                5.000000
max     1473.000000             200.000000            5.000000                8.000000
```

### Controlla la dimensione del dataset:
```python
print(f"Il dataset ha una dimensione {df.shape}\n")
print(df.info())
```
Output:
```
Il dataset ha una dimensione (500, 10)

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 500 entries, 0 to 499
Data columns (total 10 columns):
 #   Column                 Non-Null Count  Dtype  
---  ------                 --------------  -----  
 0   Customer ID            500 non-null    int64  
 1   Item Purchased         500 non-null    object 
 2   Review Rating          500 non-null    int64  
 3   Discount Applied       500 non-null    object 
 4   Purchase Amount (USD)  500 non-null    float64
 5   Previous Purchases     500 non-null    int64  
 6   Subscription Status    500 non-null    object 
 7   Promo Code Used        500 non-null    object 
 8   Payment Method         500 non-null    object 
 9   Shipping Type          500 non-null    object 
dtypes: float64(1), int64(3), object(6)
memory usage: 39.2+ KB
None
```

### Identifica le righe duplicate:
```python
duplicated_rows = df[df.duplicated(keep=False)]
print(f"Le righe duplicate sono {duplicated_rows.shape[0]}")
```
Output:
```
Le righe duplicate sono 25
```


## Analisi Colonne
### Visualizza boxplot di età e importo dell'acquisto:
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
df[['Age']].boxplot()
plt.title('Distribuzione dell\'età')
plt.ylabel('Età')
plt.show()

plt.figure(figsize=(6, 4))
df[['Purchase Amount (USD)']].boxplot()
plt.title('Distribuzione dell\'importo dell\'acquisto')
plt.ylabel('Importo (USD)')
plt.show()
```

## Contributi
Contributi sono benvenuti! Per favore, crea una pull request o apri un issue per discutere i cambiamenti che vuoi apportare.

## Licenza
Questo progetto è rilasciato sotto la [MIT License](LICENSE).

## Contatti
Per qualsiasi domanda o suggerimento, puoi contattarmi a [email@example.com](mailto:email@example.com).

Questo markdown include i comandi Python per caricare il dataset, eseguire operazioni di analisi descrittiva e visualizzazione, e mostra gli output simulati che potresti ottenere eseguendo questi comandi nel tuo ambiente Python. Assicurati di adattare i percorsi dei file e le variabili Python secondo il tuo contesto specifico.