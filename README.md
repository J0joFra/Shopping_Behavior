# Shopping Behavior Analysis

## Descrizione
Questo progetto analizza i dati relativi ai comportamenti di acquisto dei clienti per estrarre informazioni utili e fornire insights dettagliati. L'obiettivo è comprendere meglio i pattern di acquisto e utilizzare queste informazioni per migliorare le strategie di marketing e le operazioni di vendita.

## Struttura del Progetto
- `data/`: Contiene i dataset utilizzati per l'analisi.
- `notebooks/`: Jupyter notebooks con il codice di analisi ed esplorazione dei dati.
- `scripts/`: Script Python per la pulizia dei dati, l'analisi e la visualizzazione.
- `results/`: Output dell'analisi, inclusi grafici e report.
- `README.md`: Descrizione del progetto e istruzioni per l'uso.

## Contesto
Il [**Consumer Behavior and Shopping Habits Dataset**](https://www.kaggle.com/datasets/zeesolver/consumer-behavior-and-shopping-habits-datasets) fornisce approfondimenti completi sulle *preferenze*, le *tendenze* e i *modelli dei consumatori* durante le loro esperienze di acquisto. 

## Contenuto
Il dataset fornisce una panoramica dettagliata delle preferenze dei consumatori e dei comportamenti di acquisto. Include informazioni demografiche, cronologia degli acquisti, preferenze di prodotto e canali di acquisto preferiti (online o offline). 
> Questo dataset è essenziale per le aziende che mirano a personalizzare le proprie strategie per soddisfare le esigenze dei clienti e migliorare la loro esperienza di acquisto, guidando in ultima analisi le vendite e la fedeltà.

- ***ID cliente***: identificatore univoco assegnato a ciascun cliente.

- ***Età***: età del cliente.

- ***Genere***: l'identificazione di genere del cliente,.

- ***Articolo acquistato***: il prodotto o l'articolo specifico selezionato dal cliente durante la transazione.

- ***Categoria***: classificazione generale o gruppo a cui appartiene l'articolo acquistato (ad esempio, abbigliamento, elettronica, generi alimentari).

- ***Importo dell'acquisto (USD)***: il valore monetario della transazione.

- ***Posizione***: posizione geografica in cui è stato effettuato l'acquisto.

- ***Taglia***: la specifica della taglia dell'articolo acquistato.

- ***Colore***: la variante o la scelta del colore associata all'articolo acquistato.

- ***Stagione***: la pertinenza stagionale dell'articolo acquistato.

- ***Valutazione della recensione***: valutazione numerica o qualitativa fornita dal cliente in merito alla sua soddisfazione per l'articolo acquistato.

- ***Stato dell'abbonamento***: indica se il cliente ha optato per un servizio in abbonamento.

- ***Tipo di spedizione***: specifica il metodo utilizzato per consegnare l'articolo acquistato.

- ***Sconto applicato***: indica se sono stati applicati sconti promozionali all'acquisto.

- ***Codice promozionale utilizzato***: indica se è stato utilizzato un codice promozionale o un coupon durante la transazione.

- ***Acquisti precedenti***: fornisce informazioni sul numero o sulla frequenza degli acquisti precedenti effettuati dal cliente.

- ***Metodo di pagamento***: specifica la modalità di pagamento utilizzata dal cliente.

- ***Frequenza degli acquisti***: indica la frequenza con cui il cliente effettua attività di acquisto.

# Requisiti
Per eseguire il progetto, è necessario avere installati i seguenti strumenti e librerie Python:

### Strumenti

- **Python 3.x**: Assicurati di avere una versione aggiornata di Python 3 installata sul tuo sistema. Puoi scaricare Python dal sito ufficiale: [Python Downloads](https://www.python.org/downloads/).

- **Jupyter Notebook**: Questo strumento ti permette di eseguire e documentare il codice in un formato interattivo. Puoi installarlo utilizzando pip:
  ```sh
  pip install notebook
   ```
> Jupyter Notebook non è obbligatorio, ma è molto comodo specialmente per la presenza di molte righe di codice.

### Lirerie Python
Assicurati di avere le seguenti librerie installate:
- **pandas**: Libreria per la manipolazione e l'analisi dei dati.
```sh
pip install pandas
```
- **numpy**: Libreria per il calcolo numerico.
```sh
pip install numpy
```
- **matplotlib**: Libreria per la visualizzazione dei dati.
```sh
pip install matplotlib
```
- **seaborn**: Libreria per la visualizzazione dei dati basata su matplotlib.
```sh
pip install seaborn
```
- **scikit-learn**: Libreria per il machine learning.
```sh
pip install scikit-learn
```
