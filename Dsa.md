# Importing the required Dependecies


```python
import pandas as pd
import numpy as np
import datetime as dt
from pandas_datareader import data as pdr
from scipy.stats import norm, t
import matplotlib.pyplot as plt
import yfinance as yfin

yfin.pdr_override()
```

# Getting Nasdaq stocks data using yahoo finance


```python
# Get stocks names from the Nasdaq stock lists, sorting by Market Cap and putting them into lists
df = pd.read_csv("nasdaq.csv.csv")
df_sorted = df.sort_values(by="Market Cap", ascending= False)
top_10 = df_sorted.head(10)["Symbol"].tolist()
top_100 = df_sorted.head(100)["Symbol"].tolist()
top_1000 = df_sorted.head(1000)["Symbol"].tolist()

#1 stock on the top_1000 is not present in yahoo finance so we delete it and take the next one and some Symbol name adjustemnt
titolo_1001 = df_sorted.head(1100)["Symbol"].tolist()[1001]
top_1000 = [symbol.replace("WMT", titolo_1001) if symbol == "WMT" else symbol for symbol in top_1000]

top_10 = [symbol.replace("/", "-") for symbol in top_10]
top_100 = [symbol.replace("/", "-") for symbol in top_100]
top_1000 = [symbol.replace("/", "-") for symbol in top_1000]

```

# Getting Data and setting initial capital



```python
Capitale = 1000000;

inizio = fine - dt.timedelta(days=365)
fine = dt.datetime.now()

dati_10 = pdr.get_data_yahoo(top_10, inizio, fine)
dati_100 = pdr.get_data_yahoo(top_100, inizio, fine)
dati_1000 = pdr.get_data_yahoo(top_1000, inizio, fine)


```

Devi modificare solo questa parte il resto lo fa tutto in automatico


```python
# Assign the data that you want to analyze
dati = dati_100['Close']
simboli = top_100
numero_stocks = len(top_100)

```

# Check the returns and get Variance and Covariance


```python
ritorni = dati.pct_change()
ritorni = ritorni.dropna()
print(ritorni)


#calcoliamo i rendimenti medi
ritorni_medi = ritorni.mean()
print(ritorni_medi)

#calcoliamo le varianze e le covarianze tra i rendimenti
var_cov_mat = ritorni.cov() 
var_cov_mat_np = ritorni.cov().to_numpy() 
```

# Function that calculates the Standard Deviation


```python
def standard_deviation(symbol,symbol_list,covariance_matrix):
   index = symbol_list.index(symbol)
   std_dev = np.sqrt(covariance_matrix[index, index])
   return std_dev

#otteniamo la dev standard di ogni singola azione nel portafoglio
for symbol in simboli:
   std_deviation = standard_deviation(symbol,simboli,var_cov_mat_np)
   print(f"Standard Deviation {symbol}: {std_deviation}") 
```

# Calculate Portfolio Standard Deviation and add Average portfolio return


```python
#Fattore di normalizzazione
Giorni=250

#Creiamo il peso equamente distribuito tra 10 azioni
pesi_equi = 1 / numero_stocks


#np.full ci permette di creare un array con i pesi equidistribuiti per ciascuna azione del portafoglio
pesi_portafoglio = np.full(numero_stocks, pesi_equi) 

#Calcoliamo il rendimento medio atteso del portafoglio equidistribuito
ritorni_medi_port = np.sum(ritorni_medi * pesi_portafoglio) * Giorni 
print('rendimento atteso portafoglio', ritorni_medi_port)

#Deviazione standard portafoglio / portfolio standard deviation
std_port = np.sqrt(np.dot(pesi_portafoglio.T, np.dot(var_cov_mat, pesi_portafoglio)))*np.sqrt(Giorni) 


#Ritorni medio di portafoglio / average portfolio returns.
ritorni['portfolio'] = np.dot(ritorni,pesi_portafoglio) 

```

# VAR Calculation at 95% Confidence Interval


```python
VaR = norm.ppf(0.95)*std_port
print(VaR)
print("Normal VaR 95th CI       :      ", round(Capitale*VaR,2))
```

# Beta Calculation


```python
# Scarica i dati del NASDAQ Composite Index
nasdaq_data = pdr.get_data_yahoo('^IXIC',inizio,fine)['Close']
```


```python
# Calcolo dei ritorni del NASDAQ Composite Index
nasdaq_returns = nasdaq_data.pct_change().dropna()
beta_ritorni = ritorni
#beta_ritorni.drop('portfolio', axis=1, inplace=True)


```


```python
# Create a DataFrame to store Beta values
beta_values = pd.DataFrame(index=simboli, columns=['Beta'])


# Calculate Beta for each stock
for stock in ritorni:
    stock_returns_single = ritorni[stock]
    covariance = np.cov(stock_returns_single, nasdaq_returns)[0, 1]
    variance_market = np.var(nasdaq_returns)
    
    # Calculate Beta and store in the DataFrame
    beta = covariance / variance_market
    beta_values.at[stock, 'Beta'] = beta

# Display the DataFrame with Beta values
beta_values.head(50)
```


```python
beta_ritorni
```
