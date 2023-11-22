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

    [*********************100%%**********************]  10 of 10 completed
    [*********************100%%**********************]  100 of 100 completed
    [*********************100%%**********************]  1000 of 1000 completed
    


```python
# Assign the data that you want to analyze

dati = dati_1000['Close']
simboli = top_1000
numero_stocks = len(top_1000)

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

                       A       AAL      AAPL      ABBV      ABEV      ABNB  \
    Date                                                                     
    2023-10-12 -0.039160 -0.033654  0.005061 -0.006964 -0.023077 -0.031000   
    2023-10-13  0.014412 -0.028192 -0.010293 -0.002293  0.003937 -0.015004   
    2023-10-16  0.008958  0.018771 -0.000727 -0.004934  0.003922  0.008946   
    2023-10-17  0.008072  0.000000 -0.008785  0.013245 -0.015625  0.004154   
    2023-10-18 -0.033719 -0.048576 -0.007395  0.000670 -0.027778 -0.028160   
    2023-10-19  0.008287  0.007923 -0.002161 -0.025188  0.000000 -0.018008   
    2023-10-20 -0.002557 -0.032314 -0.014704  0.004879  0.004082 -0.029007   
    2023-10-23  0.001007  0.018953  0.000694 -0.010258 -0.004065  0.033479   
    2023-10-24 -0.033839 -0.007086  0.002543  0.010917  0.020408  0.013456   
    2023-10-25 -0.021204 -0.015165 -0.013492 -0.007177 -0.016000 -0.028768   
    2023-10-26  0.008801  0.009964 -0.024606 -0.000413  0.016260 -0.025570   
    2023-10-27 -0.014764 -0.020628  0.007969 -0.043182 -0.024000 -0.011951   
    2023-10-30 -0.015569  0.023810  0.012305  0.021306 -0.004098  0.025068   
    2023-10-31  0.021746 -0.002683  0.002819 -0.005004  0.041152  0.011458   
    2023-11-01 -0.004934 -0.000897  0.018739  0.009137  0.019763  0.009975   
    2023-11-02  0.015652  0.023339  0.020693  0.005545  0.011628 -0.033230   
    2023-11-03  0.043553  0.050877 -0.005181 -0.012844  0.034483  0.061818   
    2023-11-06 -0.013667 -0.026711  0.014605 -0.001556  0.018519 -0.036448   
    2023-11-07  0.027992  0.007719  0.014451  0.006799  0.007273  0.024626   
    2023-11-08 -0.010403  0.022128  0.005885 -0.000914 -0.003610 -0.028246   
    2023-11-09 -0.015084 -0.021649 -0.002625 -0.028093 -0.018116 -0.018188   
    2023-11-10  0.006776  0.004255  0.021874  0.003984  0.003690  0.022767   
    2023-11-13 -0.009311 -0.001695 -0.008584  0.000361  0.007353  0.008464   
    2023-11-14  0.038619  0.039898  0.014286 -0.004184  0.021898  0.063198   
    2023-11-15  0.017830  0.013878  0.003041 -0.003332  0.000000  0.013183   
    2023-11-16  0.005194 -0.018519  0.009042  0.004942 -0.003571 -0.016128   
    2023-11-17 -0.009108  0.008203 -0.000105  0.000145 -0.007168  0.006889   
    2023-11-20  0.007335  0.008950  0.009278  0.000072  0.003610  0.020055   
    2023-11-21  0.087208 -0.021774 -0.004231  0.003037 -0.017986 -0.022205   
    2023-11-22  0.000565  0.014839  0.003514 -0.001874  0.010989  0.017505   
    
                     ABT      ACGL     ACGLN     ACGLO  ...       YUM      YUMC  \
    Date                                                ...                       
    2023-10-12 -0.026446  0.000361 -0.015455 -0.006417  ... -0.017993 -0.021636   
    2023-10-13  0.007540  0.013365 -0.013372 -0.016393  ...  0.005419 -0.001301   
    2023-10-16  0.014416  0.018298  0.009428 -0.003535  ...  0.013433 -0.010979   
    2023-10-17 -0.000434  0.010735 -0.021016 -0.005068  ...  0.008358 -0.004704   
    2023-10-18  0.037117 -0.013507  0.001193 -0.007641  ...  0.012977  0.004348   
    2023-10-19 -0.001256 -0.019544 -0.010125 -0.007187  ... -0.003554 -0.019010   
    2023-10-20  0.014040 -0.012175  0.019856  0.032575  ... -0.005060 -0.012279   
    2023-10-23 -0.010333 -0.010875 -0.001770  0.006510  ... -0.009921  0.001554   
    2023-10-24 -0.010127  0.019057  0.028369  0.021393  ...  0.009683  0.039565   
    2023-10-25 -0.013079  0.000839 -0.005747 -0.006819  ...  0.003336 -0.014925   
    2023-10-26  0.004382 -0.012457 -0.002890  0.002452  ... -0.012966 -0.013258   
    2023-10-27 -0.012024 -0.013220 -0.005797 -0.017123  ...  0.005811  0.007869   
    2023-10-30  0.001616  0.018682 -0.010496 -0.002987  ...  0.003600  0.011617   
    2023-10-31  0.016667  0.045849  0.019446  0.018972  ...  0.008259 -0.010542   
    2023-11-01  0.004865  0.038302  0.005780  0.004900  ...  0.003641 -0.152207   
    2023-11-02  0.009999 -0.045111  0.025287  0.020478  ...  0.024485  0.027379   
    2023-11-03 -0.001251 -0.004538  0.026906  0.022456  ...  0.016657 -0.008737   
    2023-11-06 -0.008765 -0.004559  0.001638  0.007477  ...  0.000396 -0.009035   
    2023-11-07 -0.002105  0.002936  0.001090  0.000000  ... -0.007200 -0.010451   
    2023-11-08 -0.000633 -0.015104  0.012520  0.006725  ...  0.001036 -0.018876   
    2023-11-09 -0.012561  0.006538 -0.018817 -0.023727  ... -0.007165 -0.011910   
    2023-11-10  0.004490  0.015118  0.013151  0.021236  ...  0.009943  0.028280   
    2023-11-13  0.019368  0.007214  0.014603  0.006470  ... -0.005081  0.003156   
    2023-11-14  0.013467 -0.012360  0.027186  0.032599  ...  0.017716  0.020449   
    2023-11-15  0.009477 -0.033216 -0.001557  0.004002  ... -0.007136  0.009469   
    2023-11-16  0.023061  0.012703 -0.004158 -0.004429  ...  0.009556  0.000000   
    2023-11-17 -0.007082 -0.001314 -0.013570  0.001779  ... -0.001330  0.001091   
    2023-11-20  0.016575  0.010287  0.012698  0.002220  ...  0.003603  0.008281   
    2023-11-21  0.007016  0.019773 -0.004180 -0.007975  ...  0.000937 -0.007780   
    2023-11-22  0.007752 -0.002090 -0.002099  0.001787  ...  0.005458 -0.015901   
    
                       Z       ZBH      ZBRA        ZG        ZM        ZS  \
    Date                                                                     
    2023-10-12 -0.033526 -0.024964 -0.028965 -0.033852 -0.020827 -0.013869   
    2023-10-13 -0.016770  0.015382 -0.018732 -0.015625 -0.011344 -0.012482   
    2023-10-16 -0.006308  0.007191  0.038904 -0.007937  0.006853  0.021897   
    2023-10-17 -0.001646  0.014660 -0.009211 -0.003636  0.010446  0.000464   
    2023-10-18 -0.035092 -0.003753 -0.018170 -0.035280 -0.013628 -0.010970   
    2023-10-19 -0.038565 -0.012996 -0.014967 -0.034805 -0.009687 -0.007805   
    2023-10-20 -0.010916 -0.002385 -0.006991 -0.012020 -0.010423 -0.040693   
    2023-10-23 -0.012320  0.005738  0.003618 -0.010050 -0.009399  0.001788   
    2023-10-24  0.023129 -0.003043  0.003020  0.025114  0.016522  0.014217   
    2023-10-25 -0.029972 -0.011446 -0.033995 -0.029450 -0.028806 -0.043571   
    2023-10-26 -0.003666 -0.005017  0.029762  0.000537 -0.014084 -0.019034   
    2023-10-27  0.014455  0.000679  0.011473  0.012614 -0.003529 -0.001552   
    2023-10-30  0.009586  0.002132  0.012501  0.011132  0.011975  0.015806   
    2023-10-31 -0.069797  0.009670 -0.001621 -0.068676 -0.000333  0.011989   
    2023-11-01  0.005793  0.006321 -0.054624  0.007318  0.000834 -0.012855   
    2023-11-02 -0.026330  0.048825  0.045507 -0.029338  0.020823  0.016215   
    2023-11-03  0.061690 -0.013158  0.035266  0.058722  0.027905  0.032540   
    2023-11-06 -0.015389  0.005057 -0.038637 -0.021207 -0.018574 -0.005475   
    2023-11-07  0.024252 -0.030833  0.009805  0.026389  0.014073  0.048449   
    2023-11-08 -0.006051 -0.010101 -0.007691 -0.007848 -0.013718  0.014995   
    2023-11-09 -0.042880 -0.007057 -0.018989 -0.040098 -0.019085 -0.018740   
    2023-11-10  0.014934  0.012486  0.018072  0.009378  0.023248  0.025425   
    2023-11-13 -0.026976  0.003794 -0.017315 -0.027872 -0.008379  0.017882   
    2023-11-14  0.122375  0.021263  0.064409  0.121923  0.019987  0.050065   
    2023-11-15  0.011477  0.022485  0.006028  0.009809  0.015453 -0.013523   
    2023-11-16  0.004193  0.009502 -0.006822  0.001278 -0.005334  0.008778   
    2023-11-17 -0.039794  0.001076  0.011788 -0.037273  0.011356  0.006929   
    2023-11-20 -0.010233  0.007433  0.011559 -0.008221  0.029320  0.023951   
    2023-11-21 -0.009046  0.003556  0.008479 -0.009626 -0.000909 -0.003126   
    2023-11-22  0.037037  0.008237  0.015512  0.032937 -0.031999 -0.003763   
    
                     ZTO       ZTS  
    Date                            
    2023-10-12 -0.016360 -0.014919  
    2023-10-13  0.004158  0.000402  
    2023-10-16 -0.005797  0.004588  
    2023-10-17  0.000000 -0.005252  
    2023-10-18 -0.014994 -0.027086  
    2023-10-19 -0.010994 -0.008847  
    2023-10-20  0.009406 -0.005653  
    2023-10-23 -0.010165 -0.003531  
    2023-10-24  0.017972  0.003724  
    2023-10-25 -0.001681 -0.020644  
    2023-10-26 -0.009684 -0.034215  
    2023-10-27  0.001275 -0.012906  
    2023-10-30  0.008068  0.006217  
    2023-10-31 -0.007161  0.000000  
    2023-11-01 -0.002970 -0.035414  
    2023-11-02  0.018298  0.062533  
    2023-11-03  0.026745  0.008203  
    2023-11-06 -0.006105  0.005794  
    2023-11-07 -0.013923  0.024453  
    2023-11-08 -0.001246  0.022194  
    2023-11-09 -0.014969 -0.001405  
    2023-11-10  0.018151 -0.007677  
    2023-11-13 -0.004146 -0.002894  
    2023-11-14  0.015404  0.022626  
    2023-11-15 -0.003280  0.011410  
    2023-11-16 -0.011106  0.010995  
    2023-11-17 -0.061980 -0.009856  
    2023-11-20 -0.021286  0.007208  
    2023-11-21  0.000453  0.015165  
    2023-11-22  0.006341  0.003469  
    
    [30 rows x 1000 columns]
    A       0.003290
    AAL    -0.000201
    AAPL    0.002130
    ABBV   -0.002439
    ABEV    0.002129
              ...   
    ZG     -0.003855
    ZM     -0.000375
    ZS      0.003464
    ZTO    -0.003053
    ZTS     0.000623
    Length: 1000, dtype: float64
    

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

    Standard Deviation AAPL: 0.025301053713720698
    Standard Deviation MSFT: 0.022998439429258843
    Standard Deviation GOOG: 0.011140745999927638
    Standard Deviation GOOGL: 0.012530121819058566
    Standard Deviation AMZN: 0.016835974165927817
    Standard Deviation NVDA: 0.027090727925340872
    Standard Deviation META: 0.013116866208128377
    Standard Deviation BRK-A: 0.019096155790084213
    Standard Deviation BRK-B: 0.014363034644049132
    Standard Deviation HSBC: 0.013953168069922606
    Standard Deviation TSLA: 0.01610409866614021
    Standard Deviation LLY: 0.008442878152048479
    Standard Deviation UNH: 0.020315840178825397
    Standard Deviation TSM: 0.012388727909349901
    Standard Deviation V: 0.018137782304551625
    Standard Deviation NVO: 0.017059044968685974
    Standard Deviation SOFI: 0.013040822541456627
    Standard Deviation SHEL: 0.021217935982278183
    Standard Deviation JPM: 0.02017722978586995
    Standard Deviation XOM: 0.012666002567380326
    Standard Deviation AVGO: 0.01598429868691753
    Standard Deviation MA: 0.018731903136307716
    Standard Deviation JNJ: 0.015015472547682096
    Standard Deviation PG: 0.016337529015765368
    Standard Deviation ORCL: 0.02824660056479799
    Standard Deviation BHP: 0.02935813515056449
    Standard Deviation HD: 0.013842632547484643
    Standard Deviation CVX: 0.01155703555037106
    Standard Deviation ADBE: 0.068499107306949
    Standard Deviation MRK: 0.018129421526269052
    Standard Deviation TM: 0.011210612759463804
    Standard Deviation ASML: 0.013964087853924561
    Standard Deviation COST: 0.008971764491587849
    Standard Deviation KO: 0.011214942008147955
    Standard Deviation ABBV: 0.01077189394052984
    Standard Deviation PEP: 0.022933593568172633
    Standard Deviation NGG: 0.014695666048852064
    Standard Deviation BAC: 0.02212627624838063
    Standard Deviation BABA: 0.024866824750779178
    Standard Deviation CSCO: 0.010273256554470912
    Standard Deviation ACN: 0.010519816505384173
    Standard Deviation CRM: 0.039409767536687915
    Standard Deviation NVS: 0.01902130832886582
    Standard Deviation MCD: 0.05156923471643311
    Standard Deviation NFLX: 0.01653477599839343
    Standard Deviation LIN: 0.015124900727961965
    Standard Deviation AMD: 0.02600542062681304
    Standard Deviation SAP: 0.019419006813275803
    Standard Deviation TMO: 0.021785297590113602
    Standard Deviation TTE: 0.02138925786925416
    Standard Deviation TMUS: 0.015458716707980907
    Standard Deviation PFE: 0.028444139000680803
    Standard Deviation DIS: 0.012432889943830535
    Standard Deviation CMCSA: 0.014604567326466596
    Standard Deviation TBC: 0.018730698702228148
    Standard Deviation NKE: 0.01602930182881078
    Standard Deviation ABT: 0.02362584778295239
    Standard Deviation INTC: 0.013737950817493924
    Standard Deviation TBB: 0.023102329146691428
    Standard Deviation VZ: 0.035421730542528715
    Standard Deviation WFC: 0.018137194590989305
    Standard Deviation DHR: 0.012985856653182477
    Standard Deviation INTU: 0.019506481507967282
    Standard Deviation PDD: 0.019607706586308168
    Standard Deviation AMGN: 0.02665486327647524
    Standard Deviation PM: 0.012240765342907129
    Standard Deviation COP: 0.0220494928860259
    Standard Deviation LYG: 0.033080339139981874
    Standard Deviation IBM: 0.026934320532951996
    Standard Deviation QCOM: 0.029302319850217824
    Standard Deviation TXN: 0.008068665690657716
    Standard Deviation UNP: 0.023525095508935463
    Standard Deviation NOW: 0.02368631736763287
    Standard Deviation GE: 0.009490869193593963
    Standard Deviation SPGI: 0.03159501455552027
    Standard Deviation UL: 0.022939532337180282
    Standard Deviation RYAAY: 0.015760624559367765
    Standard Deviation MS: 0.034308213430371645
    Standard Deviation HON: 0.02180512363755846
    Standard Deviation AMAT: 0.013123881186605192
    Standard Deviation UPS: 0.018552354560434168
    Standard Deviation CAT: 0.018554614414765067
    Standard Deviation RTX: 0.011502073845317639
    Standard Deviation RY: 0.010806145932618208
    Standard Deviation SBUX: 0.03605238334280547
    Standard Deviation BA: 0.01655566256501607
    Standard Deviation SNY: 0.020349050610137264
    Standard Deviation NEE: 0.024306056361878953
    Standard Deviation T: 0.015294038777828655
    Standard Deviation AXP: 0.02063206008007636
    Standard Deviation LOW: 0.021619963188094928
    Standard Deviation LMT: 0.01576320452166932
    Standard Deviation TD: 0.015744703393270033
    Standard Deviation DE: 0.01427050865922322
    Standard Deviation ELV: 0.02210647996198127
    Standard Deviation BKNG: 0.017352584319177924
    Standard Deviation RIO: 0.023360039714912677
    Standard Deviation GS: 0.020171334883414997
    Standard Deviation TJX: 0.013824506670964529
    Standard Deviation BUD: 0.024848197377067175
    Standard Deviation HDB: 0.02084734851979751
    Standard Deviation BCS: 0.021047008848741135
    Standard Deviation SONY: 0.01597755347569911
    Standard Deviation UBER: 0.018869414200701554
    Standard Deviation SYK: 0.05072221210946725
    Standard Deviation MUFG: 0.0134691068357782
    Standard Deviation BMY: 0.032666551727493484
    Standard Deviation EQNR: 0.018650653185716667
    Standard Deviation AZN: 0.013301222900397749
    Standard Deviation PBR: 0.01670735488627707
    Standard Deviation BP: 0.023961779510062016
    Standard Deviation SCHW: 0.018705420544675583
    Standard Deviation MMC: 0.024446996202776496
    Standard Deviation BLK: 0.021519242270668055
    Standard Deviation ISRG: 0.030328368562535603
    Standard Deviation PLD: 0.01949675131722924
    Standard Deviation VRTX: 0.012649269036688697
    Standard Deviation MDLZ: 0.033851361823068506
    Standard Deviation PGR: 0.01481017592345386
    Standard Deviation GILD: 0.0271129197227308
    Standard Deviation MDT: 0.01806087141518243
    Standard Deviation ADP: 0.026825026028330427
    Standard Deviation DEO: 0.03868228144738648
    Standard Deviation CB: 0.02867343286020546
    Standard Deviation ETN: 0.019457160306545512
    Standard Deviation REGN: 0.013903582961063147
    Standard Deviation CVS: 0.04198722591651469
    Standard Deviation LRCX: 0.017616453286605476
    Standard Deviation UBS: 0.01428974680982217
    Standard Deviation AMT: 0.029106694690021043
    Standard Deviation CI: 0.03369340080448984
    Standard Deviation ADI: 0.016871142135029223
    Standard Deviation MU: 0.015721111046416544
    Standard Deviation C: 0.029507980609420992
    Standard Deviation IBN: 0.019344042594805138
    Standard Deviation ZTS: 0.02493671157070711
    Standard Deviation CME: 0.015922209025671933
    Standard Deviation SHOP: 0.017604549737566276
    Standard Deviation SNPS: 0.01540344107160253
    Standard Deviation BSX: 0.025740064208562494
    Standard Deviation SLB: 0.015221129381320935
    Standard Deviation ABNB: 0.029868169080332713
    Standard Deviation PANW: 0.03344531479874832
    Standard Deviation SO: 0.027548416022696805
    Standard Deviation NTES: 0.025856466243546882
    Standard Deviation FI: 0.013324268610768426
    Standard Deviation CNI: 0.008834156425885208
    Standard Deviation EQIX: 0.008227079890581105
    Standard Deviation ENB: 0.026591755586899028
    Standard Deviation MO: 0.012921465849784559
    Standard Deviation EOG: 0.018851162418250522
    Standard Deviation GSK: 0.020965860083183803
    Standard Deviation CDNS: 0.015217730924584606
    Standard Deviation NOC: 0.022918204912716154
    Standard Deviation CNQ: 0.013454979235943581
    Standard Deviation KLAC: 0.013823728859094173
    Standard Deviation INFY: 0.04783151761415384
    Standard Deviation BX: 0.02888965465799975
    Standard Deviation BTI: 0.01263872976135608
    Standard Deviation ITW: 0.025901153928315105
    Standard Deviation WM: 0.032085975802407365
    Standard Deviation VALE: 0.016050686731130623
    Standard Deviation RTO: 0.010377557856978864
    Standard Deviation RELX: 0.017523208680313634
    Standard Deviation CP: 0.013399806881656259
    Standard Deviation MELI: 0.016119012262320428
    Standard Deviation DUK: 0.03642672816393568
    Standard Deviation BDX: 0.019740366409309873
    Standard Deviation AON: 0.03744438666289048
    Standard Deviation ANET: 0.010128284522386045
    Standard Deviation GD: 0.021200201556502696
    Standard Deviation RACE: 0.012741072101653757
    Standard Deviation SHW: 0.012285392273364809
    Standard Deviation VMW: 0.021412268833730323
    Standard Deviation SMFG: 0.02532591464705708
    Standard Deviation TRI: 0.009601648314146885
    Standard Deviation MCO: 0.023079421885091617
    Standard Deviation ICE: 0.023965473182735158
    Standard Deviation CL: 0.021061705783814767
    Standard Deviation MCK: 0.03223597495026983
    Standard Deviation SAN: 0.02882969814113743
    Standard Deviation HUM: 0.014512174986627687
    Standard Deviation HCA: 0.013197048307314047
    Standard Deviation FDX: 0.023011044818799696
    Standard Deviation CHTR: 0.02097394227100932
    Standard Deviation STLA: 0.03767621740948894
    Standard Deviation CSX: 0.010462996373355191
    Standard Deviation WDAY: 0.015558089271383675
    Standard Deviation APD: 0.029966415710247846
    Standard Deviation PYPL: 0.02181942443060969
    Standard Deviation ITUB: 0.015353551190607799
    Standard Deviation ORLY: 0.01803828328405931
    Standard Deviation CMG: 0.019815029020942725
    Standard Deviation MNST: 0.014931708591722092
    Standard Deviation MAR: 0.01775853479631769
    Standard Deviation E: 0.01631354713273676
    Standard Deviation EPD: 0.03232885760644388
    Standard Deviation HMC: 0.014086227647053202
    Standard Deviation BMO: 0.017410816073482223
    Standard Deviation ROP: 0.01603120903654605
    Standard Deviation PXD: 0.007365520964380612
    Standard Deviation SCCO: 0.024858119814036322
    Standard Deviation MPC: 0.03644641557612566
    Standard Deviation TDG: 0.01589339291636122
    Standard Deviation CTAS: 0.01393151783466099
    Standard Deviation AMX: 0.010641871517259776
    Standard Deviation KKR: 0.026136483605332255
    Standard Deviation OXY: 0.01998879656037174
    Standard Deviation AJG: 0.018305610391011844
    Standard Deviation PH: 0.016476909852057107
    Standard Deviation ARM: 0.020291967280905574
    Standard Deviation USB: 0.011683309925755902
    Standard Deviation BN: 0.01310716610548749
    Standard Deviation DELL: 0.01298295543619151
    Standard Deviation SNOW: 0.011429074709446523
    Standard Deviation BNS: 0.010793654762586887
    Standard Deviation PFH: 0.009308511682137522
    Standard Deviation MSI: 0.00998224835697076
    Standard Deviation APH: 0.012887603172203046
    Standard Deviation BBVA: 0.012077740270536236
    Standard Deviation MMM: 0.026482206328447857
    Standard Deviation TT: 0.011417043839469275
    Standard Deviation ECL: 0.018395248616574083
    Standard Deviation LULU: 0.012679271090826637
    Standard Deviation TGT: 0.01909808673828561
    Standard Deviation RSG: 0.023812343896423067
    Standard Deviation EMR: 0.03714929541109719
    Standard Deviation PSX: 0.02678930980319713
    Standard Deviation ING: 0.016261614264243843
    Standard Deviation APO: 0.01616090175901697
    Standard Deviation FCX: 0.010020286571349128
    Standard Deviation PNC: 0.012024862232046506
    Standard Deviation PCG: 0.02528994218735794
    Standard Deviation WPP: 0.012427174737132432
    Standard Deviation AZO: 0.013725974993127808
    Standard Deviation AFL: 0.029166293998690623
    Standard Deviation NXPI: 0.016084858766976226
    Standard Deviation CRWD: 0.021256080511022044
    Standard Deviation WELL: 0.015235695711102866
    Standard Deviation MRVL: 0.02045418567348867
    Standard Deviation PCAR: 0.02025216999198923
    Standard Deviation CPRT: 0.028180039099144336
    Standard Deviation AIG: 0.014437595313743284
    Standard Deviation TEAM: 0.018553604388096806
    Standard Deviation MET: 0.02367952998831086
    Standard Deviation STZ: 0.02028752135266543
    Standard Deviation NSC: 0.02196154507332125
    Standard Deviation SRE: 0.021612929812523454
    Standard Deviation ADSK: 0.011952992491009198
    Standard Deviation BSBR: 0.008507489942781454
    Standard Deviation PSA: 0.012340706267744911
    Standard Deviation KDP: 0.01509178427791239
    Standard Deviation HES: 0.022775659966053238
    Standard Deviation ABEV: 0.02107025962934016
    Standard Deviation AESC: 0.018864211185926766
    Standard Deviation WMB: 0.013543546876678282
    Standard Deviation CARR: 0.018771906028008038
    Standard Deviation SU: 0.010872369448883704
    Standard Deviation TAK: 0.02477862809687151
    Standard Deviation ODFL: 0.022755329745906577
    Standard Deviation NIMC: 0.03506210417376887
    Standard Deviation CCI: 0.020844186900016156
    Standard Deviation ROST: 0.020138492915300624
    Standard Deviation CRH: 0.028473282074882655
    Standard Deviation EL: 0.03889290389692025
    Standard Deviation PAYX: 0.019887502024280725
    Standard Deviation ET: 0.016128006407713842
    Standard Deviation VLO: 0.012495811306422962
    Standard Deviation DHI: 0.018684830256370705
    Standard Deviation LNG: 0.0569019745531263
    Standard Deviation AEP: 0.014706635127376456
    Standard Deviation KMB: 0.03746643799421743
    Standard Deviation HLT: 0.01454521383631188
    Standard Deviation MFG: 0.02568562280189813
    Standard Deviation JD: 0.02101243666513941
    Standard Deviation KHC: 0.02341419198681876
    Standard Deviation SGEN: 0.014175790034330466
    Standard Deviation MSCI: 0.022582066160596704
    Standard Deviation MCHP: 0.020210558206394683
    Standard Deviation COF: 0.01861777267069996
    Standard Deviation COR: 0.018715227257854124
    Standard Deviation PLTR: 0.04361727729669329
    Standard Deviation WDS: 0.021524019226962118
    Standard Deviation EW: 0.018080789770849467
    Standard Deviation EXC: 0.01876960888210902
    Standard Deviation TEL: 0.01815139765991534
    Standard Deviation TFC: 0.02056438234725345
    Standard Deviation LI: 0.01563729016387197
    Standard Deviation F: 0.014113384777332833
    Standard Deviation GWW: 0.015144359039165809
    Standard Deviation DLR: 0.013569119130963368
    Standard Deviation HSY: 0.01186885351316914
    Standard Deviation CEG: 0.021055440770579987
    Standard Deviation TRV: 0.012685254119801634
    Standard Deviation ADM: 0.01095052848844168
    Standard Deviation CNC: 0.012057692425149468
    Standard Deviation FTNT: 0.05392498336940977
    Standard Deviation GIS: 0.025136519487339252
    Standard Deviation TTD: 0.01681469484846824
    Standard Deviation D: 0.03285667141342749
    Standard Deviation HLN: 0.01071940536707663
    Standard Deviation OKE: 0.011476084906092936
    Standard Deviation NUE: 0.020123793488768082
    Standard Deviation SPG: 0.020343804378265
    Standard Deviation TRP: 0.0167783594363945
    Standard Deviation KVUE: 0.015374508585380808
    Standard Deviation BIDU: 0.009644601158128317
    Standard Deviation LVS: 0.05060479234224623
    Standard Deviation STM: 0.017226263561485603
    Standard Deviation GM: 0.02531421814843257
    Standard Deviation ALC: 0.01635326452006006
    Standard Deviation O: 0.015264964206181135
    Standard Deviation KMI: 0.016891561098486232
    Standard Deviation DXCM: 0.04688823576373719
    Standard Deviation MPLX: 0.01963419491654384
    Standard Deviation EA: 0.018685449571259095
    Standard Deviation BCE: 0.010268290527599895
    Standard Deviation IQV: 0.020493628058611085
    Standard Deviation YUM: 0.01664623625070492
    Standard Deviation IDXX: 0.017931959878345036
    Standard Deviation CM: 0.012086162326144997
    Standard Deviation BK: 0.05590064935163216
    Standard Deviation LHX: 0.028408888542689876
    Standard Deviation JCI: 0.01637592742837754
    Standard Deviation VRSK: 0.02653626448520541
    Standard Deviation BKR: 0.008569448026934594
    Standard Deviation AME: 0.025571107859464228
    Standard Deviation LEN: 0.018425887509832168
    Standard Deviation MFC: 0.016154421054491434
    Standard Deviation DOW: 0.01777062392161661
    Standard Deviation HAL: 0.020364784647319386
    Standard Deviation FAST: 0.017846804473558443
    Standard Deviation ALL: 0.016909739007568717
    Standard Deviation DASH: 0.02322663771889004
    Standard Deviation SYY: 0.019662988500168878
    Standard Deviation WCN: 0.017623727243495185
    Standard Deviation AMP: 0.027446912492110453
    Standard Deviation CVE: 0.013005832314326178
    Standard Deviation OTIS: 0.01955629778644964
    Standard Deviation BBD: 0.01409774225329768
    Standard Deviation DDOG: 0.03153865707606552
    Standard Deviation SPOT: 0.014066558297954819
    Standard Deviation PRU: 0.020775913421770195
    Standard Deviation BIIB: 0.023119892838836895
    Standard Deviation ARES: 0.0359838661316584
    Standard Deviation CTSH: 0.01141070970764025
    Standard Deviation XEL: 0.015587004876522766
    Standard Deviation IMO: 0.04274198866914697
    Standard Deviation FERG: 0.031066305005210845
    Standard Deviation CSGP: 0.030397119165373614
    Standard Deviation SATX: 0.013713538170445605
    Standard Deviation CTVA: 0.01686331470777706
    Standard Deviation ACGL: 0.026655309089440166
    Standard Deviation KR: 0.03384837747762983
    Standard Deviation A: 0.025298456421581184
    Standard Deviation SQ: 0.020787100977066778
    Standard Deviation IT: 0.010834972144130843
    Standard Deviation ORAN: 0.017218488157105995
    Standard Deviation GEHC: 0.013041431115794554
    Standard Deviation ED: 0.015575965750156332
    Standard Deviation FIS: 0.014431787955436953
    Standard Deviation CMI: 0.014411340637877332
    Standard Deviation PEG: 0.022216876727376052
    Standard Deviation PUK: 0.02030257312188845
    Standard Deviation PPG: 0.0273622053761931
    Standard Deviation DKNG: 0.0054353639174501015
    Standard Deviation LYB: 0.021538268351792924
    Standard Deviation NU: 0.011734361590395277
    Standard Deviation NDAQ: 0.01831442117120579
    Standard Deviation URI: 0.033410492746807574
    Standard Deviation ROK: 0.026648349388389943
    Standard Deviation DD: 0.016914925344931723
    Standard Deviation VICI: 0.02323090739277112
    Standard Deviation DVN: 0.02004504671559031
    Standard Deviation CHT: 0.027797943001652906
    Standard Deviation CDW: 0.017909859346079613
    Standard Deviation GPN: 0.022230563305614355
    Standard Deviation MLM: 0.014515505654304334
    Standard Deviation FANG: 0.014968163796298499
    Standard Deviation GFS: 0.017828304958174678
    Standard Deviation SLF: 0.03433414891302129
    Standard Deviation CCEP: 0.02478526400484502
    Standard Deviation VMC: 0.026901638620707116
    Standard Deviation BBDO: 0.011897308387213406
    Standard Deviation CQP: 0.01872313137984641
    Standard Deviation ON: 0.030986103250193638
    Standard Deviation HBANM: 0.018933598818530056
    Standard Deviation VOD: 0.018516832631708828
    Standard Deviation IR: 0.010891018139199697
    Standard Deviation CPNG: 0.027599126775867833
    Standard Deviation NEM: 0.01914027777465842
    Standard Deviation GOLD: 0.017566766979732903
    Standard Deviation NTR: 0.023251194324374575
    Standard Deviation VEEV: 0.028266369491141075
    Standard Deviation HPQ: 0.023428231151601
    Standard Deviation ARGX: 0.02843786259834925
    Standard Deviation MRNA: 0.02089916448461488
    Standard Deviation PKX: 0.019294203542787255
    Standard Deviation MDB: 0.012918355381646653
    Standard Deviation DG: 0.014976634531307426
    Standard Deviation FMX: 0.010079482458109378
    Standard Deviation CAH: 0.009390043370560465
    Standard Deviation ZS: 0.011436987937176762
    Standard Deviation WEC: 0.03405840501859314
    Standard Deviation TU: 0.012977698534549786
    Standard Deviation DLTR: 0.014073605406881929
    Standard Deviation EXR: 0.02309308424359366
    Standard Deviation WST: 0.01885403463617004
    Standard Deviation TTWO: 0.018845636943801723
    Standard Deviation WIT: 0.021583835450656546
    Standard Deviation ANSS: 0.02144607046261475
    Standard Deviation PWR: 0.02507307562410126
    Standard Deviation SE: 0.020126461730350664
    Standard Deviation WTW: 0.018764034839850074
    Standard Deviation SPLK: 0.021923731755346457
    Standard Deviation EIX: 0.016549682948884755
    Standard Deviation AWK: 0.013006068367795264
    Standard Deviation ELP: 0.034686804292616814
    Standard Deviation BNTX: 0.0192243451103565
    Standard Deviation EC: 0.014393671679469254
    Standard Deviation BNH: 0.016272572805341785
    Standard Deviation FICO: 0.016826938505017718
    Standard Deviation RBLX: 0.023400714235261386
    Standard Deviation RCL: 0.013924914283111677
    Standard Deviation HBANP: 0.019976763270550178
    Standard Deviation AVB: 0.017611995490709725
    Standard Deviation SBAC: 0.01574878613740026
    Standard Deviation WBD: 0.011660492130388177
    Standard Deviation AEM: 0.011106441413218834
    Standard Deviation GIB: 0.019449717402054475
    Standard Deviation XYL: 0.012100191868761132
    Standard Deviation MPWR: 0.015282984364877076
    Standard Deviation GLW: 0.01396540146274585
    Standard Deviation FNV: 0.01364475832895914
    Standard Deviation FTV: 0.012870168865584208
    Standard Deviation DB: 0.017475091280108938
    Standard Deviation EFX: 0.017426026577058284
    Standard Deviation BNJ: 0.010816437363810005
    Standard Deviation HEI: 0.03715253562525889
    Standard Deviation TLK: 0.020215282747334345
    Standard Deviation TEF: 0.012072163424558488
    Standard Deviation MTD: 0.022137250063653
    Standard Deviation COIN: 0.01280194404424624
    Standard Deviation CHD: 0.017930388927069045
    Standard Deviation GRMN: 0.01776328131395493
    Standard Deviation HIG: 0.012572476906788
    Standard Deviation CBRE: 0.024888968248686694
    Standard Deviation IX: 0.020148014418457753
    Standard Deviation RCI: 0.02534425102239622
    Standard Deviation ZBH: 0.019055560329901564
    Standard Deviation TCOM: 0.01847600153662924
    Standard Deviation NWG: 0.015050946149966611
    Standard Deviation WY: 0.012855216245892342
    Standard Deviation TW: 0.009460184052614761
    Standard Deviation KEYS: 0.01095485527581454
    Standard Deviation DAL: 0.02611760938529648
    Standard Deviation QSR: 0.022907798071762044
    Standard Deviation TSCO: 0.03822687509223549
    Standard Deviation RMD: 0.015620428455194086
    Standard Deviation PINS: 0.01908916680129633
    Standard Deviation TROW: 0.012947239171962003
    Standard Deviation HUBS: 0.03367976709598887
    Standard Deviation NET: 0.014423458435478875
    Standard Deviation VRSN: 0.015283798975710907
    Standard Deviation BR: 0.038697734461013884
    Standard Deviation EBAY: 0.017636513010957495
    Standard Deviation DFS: 0.01593798096459664
    Standard Deviation RJF: 0.01641472737933034
    Standard Deviation ALNY: 0.028832152328443796
    Standard Deviation APTV: 0.017921134522907688
    Standard Deviation MOH: 0.02021581704229824
    Standard Deviation STT: 0.027622891216697438
    Standard Deviation ICLR: 0.04042783251231102
    Standard Deviation ETR: 0.015686881498570737
    Standard Deviation BRO: 0.017476696893301773
    Standard Deviation EQR: 0.023366434239309036
    Standard Deviation DTE: 0.0182332261717132
    Standard Deviation FE: 0.016497253304478245
    Standard Deviation CTRA: 0.02119646222462502
    Standard Deviation FCNCA: 0.028807420969622775
    Standard Deviation HWM: 0.0186985440380732
    Standard Deviation HPE: 0.016678419120669197
    Standard Deviation AEE: 0.01242218338520928
    Standard Deviation TS: 0.012279946053316686
    Standard Deviation WAB: 0.020958407667172763
    Standard Deviation FTS: 0.027083874636863136
    Standard Deviation LYV: 0.021182375087745328
    Standard Deviation STE: 0.016175096056384373
    Standard Deviation CBOE: 0.01573855251179954
    Standard Deviation MTB: 0.029904083714083436
    Standard Deviation WPM: 0.01786167225436084
    Standard Deviation NOK: 0.03527206958340584
    Standard Deviation ZTO: 0.013991080645879318
    Standard Deviation ULTA: 0.009248253522344006
    Standard Deviation INVH: 0.0173540447711368
    Standard Deviation ES: 0.011437058708886404
    Standard Deviation GPC: 0.011099314100991287
    Standard Deviation TRGP: 0.016874921827531166
    Standard Deviation OWL: 0.027057012980457186
    Standard Deviation PPL: 0.012941426130238916
    Standard Deviation NVR: 0.02114888299784685
    Standard Deviation UMC: 0.03186667369824242
    Standard Deviation SNAP: 0.017201006221214777
    Standard Deviation BEKE: 0.021581387596400985
    Standard Deviation ROL: 0.015271006880223833
    Standard Deviation CCJ: 0.023697609621139374
    Standard Deviation DOV: 0.022809941182054767
    Standard Deviation IFF: 0.020003206035932277
    Standard Deviation ZM: 0.010070011302129022
    Standard Deviation STLD: 0.01122734415298373
    Standard Deviation SIRI: 0.025457171141607254
    Standard Deviation YUMC: 0.040960367197578645
    Standard Deviation TECK: 0.031172743739630054
    Standard Deviation DRI: 0.008760017466459193
    Standard Deviation K: 0.018284854687772866
    Standard Deviation MT: 0.012181410987669524
    Standard Deviation HRL: 0.015463773471389892
    Standard Deviation TDY: 0.009907240210062838
    Standard Deviation MKL: 0.027603199153121663
    Standard Deviation JBHT: 0.02875820894893422
    Standard Deviation DUKB: 0.023391232028571796
    Standard Deviation PTC: 0.02361369340334151
    Standard Deviation WBA: 0.034039255838126205
    Standard Deviation SYM: 0.022058371433058572
    Standard Deviation PBA: 0.042419453268072976
    Standard Deviation PHM: 0.017114201190917024
    Standard Deviation TYL: 0.017515685473843186
    Standard Deviation WRB: 0.01884491650511781
    Standard Deviation PHG: 0.025683925212766738
    Standard Deviation LH: 0.021002583540330435
    Standard Deviation MKC: 0.013760915492234611
    Standard Deviation KOF: 0.013925254592891545
    Standard Deviation EBR: 0.033572224414101785
    Standard Deviation IRM: 0.02067570322982156
    Standard Deviation LPLA: 0.009875146767261407
    Standard Deviation VLTO: 0.018089939611993375
    Standard Deviation ASX: 0.02224738659439599
    Standard Deviation FDS: 0.008460581789308246
    Standard Deviation EDR: 0.013652767243604643
    Standard Deviation FITBI: 0.012297666327333706
    Standard Deviation ILMN: 0.02878036707884224
    Standard Deviation VTR: 0.025692561275893998
    Standard Deviation RYAN: 0.016106997166248173
    Standard Deviation FLT: 0.01532074212504588
    Standard Deviation FITB: 0.021452732419552093
    Standard Deviation CNP: 0.04305763185806817
    Standard Deviation AQNB: 0.014158159425687268
    Standard Deviation TVE: 0.021883660435362558
    Standard Deviation WMG: 0.01429526451651789
    Standard Deviation VIV: 0.02312421334045811
    Standard Deviation CHKP: 0.02216478725303915
    Standard Deviation BAX: 0.016553045234546544
    Standard Deviation J: 0.019429760109253152
    Standard Deviation AGNCN: 0.012934036997986646
    Standard Deviation AKAM: 0.020212104911996594
    Standard Deviation EXPD: 0.015950181362537265
    Standard Deviation ATO: 0.013947683727601728
    Standard Deviation TSN: 0.018906549855728025
    Standard Deviation EG: 0.02067400144840225
    Standard Deviation HOLX: 0.013614752834678119
    Standard Deviation ARE: 0.022826909856427626
    Standard Deviation PFG: 0.02276281443005913
    Standard Deviation CLX: 0.009051052651788289
    Standard Deviation RPRX: 0.02594756192904037
    Standard Deviation BAH: 0.011600851142767675
    Standard Deviation EQT: 0.01484161768977618
    Standard Deviation DECK: 0.033815830137736924
    Standard Deviation AER: 0.010032727940144717
    Standard Deviation FITBP: 0.014115805139253963
    Standard Deviation AXON: 0.043000248546256864
    Standard Deviation COO: 0.02105516463388431
    Standard Deviation CCL: 0.0157470109201889
    Standard Deviation JBL: 0.018748069475525687
    Standard Deviation CMS: 0.015198847159339382
    Standard Deviation ASBA: 0.020693574281089
    Standard Deviation NTAP: 0.02578399007487363
    Standard Deviation EXPE: 0.024393488636050436
    Standard Deviation BMRN: 0.014827381651049765
    Standard Deviation KB: 0.0266861663743572
    Standard Deviation CINF: 0.023493713849921476
    Standard Deviation RKT: 0.020772255955278405
    Standard Deviation FWONK: 0.009554917814343089
    Standard Deviation ERIC: 0.016228313961785146
    Standard Deviation CF: 0.03698320082207887
    Standard Deviation BALL: 0.015333330832134591
    Standard Deviation HUBB: 0.017624170357442
    Standard Deviation AGNCM: 0.016100775154540673
    Standard Deviation VRT: 0.027454743440585894
    Standard Deviation WLK: 0.022476367474511382
    Standard Deviation RS: 0.016935184884643753
    Standard Deviation SLMBP: 0.0063492259519586595
    Standard Deviation BG: 0.03247517645495318
    Standard Deviation STX: 0.010592391195351067
    Standard Deviation AGNCO: 0.037516604307751354
    Standard Deviation VFS: 0.019718907795741567
    Standard Deviation WAT: 0.026223293821318484
    Standard Deviation BLDR: 0.021359174428796247
    Standard Deviation BSY: 0.014144599426141182
    Standard Deviation MGA: 0.014620963300692432
    Standard Deviation OMC: 0.014986323597669745
    Standard Deviation DGX: 0.03877866108724879
    Standard Deviation HBAN: 0.02381276354391532
    Standard Deviation SOJE: 0.024362692454079148
    Standard Deviation TKO: 0.03989443703481689
    Standard Deviation RIVN: 0.020872872988557955
    Standard Deviation TXT: 0.02372157578957167
    Standard Deviation SREA: 0.016158534868291295
    Standard Deviation WSO: 0.016833100232865952
    Standard Deviation L: 0.023181740431130346
    Standard Deviation DT: 0.010890476555321733
    Standard Deviation MRO: 0.01807580581670312
    Standard Deviation IEX: 0.015808910033279295
    Standard Deviation NTRS: 0.01329580150671977
    Standard Deviation ALGN: 0.023314317622302846
    Standard Deviation FOXA: 0.02379913986397947
    Standard Deviation WDC: 0.041118697823245495
    Standard Deviation AVY: 0.03816969300624105
    Standard Deviation SUI: 0.032107097859633546
    Standard Deviation FSLR: 0.012018196625942728
    Standard Deviation SMCI: 0.015008204020219343
    Standard Deviation LDOS: 0.027378297441532088
    Standard Deviation FITBO: 0.009300975801143177
    Standard Deviation AGNCL: 0.038689015785452335
    Standard Deviation SNA: 0.017729735112894583
    Standard Deviation MAA: 0.022064900785293905
    Standard Deviation RF: 0.019318753202947046
    Standard Deviation PKG: 0.01603751268000004
    Standard Deviation AQNU: 0.00908897601603862
    Standard Deviation LUV: 0.01925715008117278
    Standard Deviation SWKS: 0.02218937777110005
    Standard Deviation LII: 0.018994644936285544
    Standard Deviation AGNCP: 0.01694249455397543
    Standard Deviation FWONA: 0.016199185307457453
    Standard Deviation BBY: 0.013311371234026408
    Standard Deviation EPAM: 0.01749670153370789
    Standard Deviation LW: 0.01869813155081698
    Standard Deviation ENTG: 0.020694861766285023
    Standard Deviation SQM: 0.02172390574646356
    Standard Deviation CNHI: 0.025725494147204517
    Standard Deviation CELH: 0.017227258912133152
    Standard Deviation ALB: 0.024386713060683456
    Standard Deviation FOX: 0.018433395596352892
    Standard Deviation SHG: 0.019176592522881664
    Standard Deviation CAG: 0.009302498519228994
    Standard Deviation APP: 0.022050643528019203
    Standard Deviation ESS: 0.03335308998727028
    Standard Deviation AEG: 0.017851728124821248
    Standard Deviation JHX: 0.01854169266055782
    Standard Deviation MGM: 0.023988343474012488
    Standard Deviation AMCR: 0.023845911542435018
    Standard Deviation TER: 0.018295083013988613
    Standard Deviation GGG: 0.018815618560298532
    Standard Deviation SWK: 0.028085003071895057
    Standard Deviation DPZ: 0.018947796076654245
    Standard Deviation SSNC: 0.013500430140676697
    Standard Deviation ERIE: 0.016411770177351015
    Standard Deviation LOGI: 0.01266249553583897
    Standard Deviation NIO: 0.03435167634488062
    Standard Deviation TME: 0.024631268149875696
    Standard Deviation MANH: 0.013815228490366779
    Standard Deviation TPL: 0.04761606006759209
    Standard Deviation POOL: 0.019229697940956504
    Standard Deviation CSL: 0.013915524151859107
    Standard Deviation BGNE: 0.02947645744140507
    Standard Deviation CE: 0.00654603923316385
    Standard Deviation NWS: 0.01616307616936081
    Standard Deviation NDSN: 0.01270037882316432
    Standard Deviation IOT: 0.015655904040497833
    Standard Deviation CRBG: 0.017164718062618097
    Standard Deviation AMH: 0.01163981259800337
    Standard Deviation MAS: 0.016897215556483776
    Standard Deviation AVTR: 0.020769698734413745
    Standard Deviation LNT: 0.01442961650332545
    Standard Deviation RPM: 0.02801463613485013
    Standard Deviation VST: 0.010444824725711253
    Standard Deviation GRAB: 0.011539302017707715
    Standard Deviation IHG: 0.022286143024942265
    Standard Deviation GDDY: 0.02450052492405195
    Standard Deviation UAL: 0.046890614288818515
    Standard Deviation ACI: 0.04553019348219115
    Standard Deviation RBA: 0.039799315127094025
    Standard Deviation GEN: 0.034285957347549835
    Standard Deviation NWSA: 0.07511868049990088
    Standard Deviation ELS: 0.014060617755132078
    Standard Deviation CPB: 0.009107992628556306
    Standard Deviation BAM: 0.020233054180541672
    Standard Deviation XP: 0.014811895991595584
    Standard Deviation SYF: 0.015352179232453205
    Standard Deviation OVV: 0.04120431141519492
    Standard Deviation LBRDA: 0.04041538347349912
    Standard Deviation GLPI: 0.02003853587081864
    Standard Deviation NMR: 0.030038396020064684
    Standard Deviation LBRDK: 0.011448123964184926
    Standard Deviation BKDT: 0.03952077169780516
    Standard Deviation BPYPM: 0.00897239944372008
    Standard Deviation QRTEP: 0.01838776202404249
    Standard Deviation LKQ: 0.01620579070027257
    Standard Deviation FNF: 0.01623814928062482
    Standard Deviation BIP: 0.012543097512827926
    Standard Deviation WPC: 0.009098536854031615
    Standard Deviation TAP: 0.018585082189823198
    Standard Deviation INCY: 0.023362732584667364
    Standard Deviation AGR: 0.017169043204912627
    Standard Deviation CFG: 0.023407959917466407
    Standard Deviation BPYPO: 0.04223909599598692
    Standard Deviation ROKU: 0.01372512943322626
    Standard Deviation BPYPP: 0.04661477031769399
    Standard Deviation GFI: 0.020799314887907754
    Standard Deviation IP: 0.04967100940579839
    Standard Deviation BEN: 0.011956423328527163
    Standard Deviation HST: 0.02189069882862138
    Standard Deviation EVRG: 0.02034394050912971
    Standard Deviation EDU: 0.015165547962782132
    Standard Deviation FLEX: 0.03906516414632947
    Standard Deviation APA: 0.020266654695581466
    Standard Deviation SJM: 0.0140028858285824
    Standard Deviation MORN: 0.012994568708999364
    Standard Deviation ARCC: 0.022538546274762882
    Standard Deviation PSTG: 0.006647480792891598
    Standard Deviation NBIX: 0.015797871611377867
    Standard Deviation VLYPO: 0.009857815076139882
    Standard Deviation MOS: 0.01521938855976696
    Standard Deviation KIM: 0.017465367490295744
    Standard Deviation IPG: 0.011132688009802652
    Standard Deviation REG: 0.020270427356706252
    Standard Deviation H: 0.01408494567070381
    Standard Deviation PARAP: 0.016303346105872887
    Standard Deviation AZPN: 0.0193898258912209
    Standard Deviation JKHY: 0.02661004671103872
    Standard Deviation OKTA: 0.013773165306551854
    Standard Deviation UGIC: 0.02422939824419519
    Standard Deviation VTRS: 0.01865691011722027
    Standard Deviation BPYPN: 0.01709020264731259
    Standard Deviation ACM: 0.07050609563694155
    Standard Deviation OC: 0.018588912110773858
    Standard Deviation GL: 0.013050988674561325
    Standard Deviation CG: 0.015519765059372359
    Standard Deviation RDY: 0.023786126710105413
    Standard Deviation EXAS: 0.03652273379679084
    Standard Deviation HTHT: 0.017926977218241918
    Standard Deviation CNA: 0.02325660460787951
    Standard Deviation PODD: 0.011761959893233023
    Standard Deviation SNN: 0.01818030060044801
    Standard Deviation AOS: 0.012283607462125487
    Standard Deviation PAA: 0.008050312151492035
    Standard Deviation RNR: 0.022484552317011047
    Standard Deviation LEGN: 0.03206541318375487
    Standard Deviation NICE: 0.014955408383360562
    Standard Deviation CHK: 0.0506112103798697
    Standard Deviation GFL: 0.01841974778051259
    Standard Deviation LECO: 0.015691656826332653
    Standard Deviation UDR: 0.015260931527316094
    Standard Deviation AES: 0.041294604165802676
    Standard Deviation UTHR: 0.0194395427649941
    Standard Deviation ZBRA: 0.042927641031919414
    Standard Deviation RVTY: 0.018339588093426052
    Standard Deviation BCH: 0.02261603467934647
    Standard Deviation WYNN: 0.021984963539844877
    Standard Deviation BEP: 0.0314056235495437
    Standard Deviation WES: 0.01693553421348672
    Standard Deviation ENPH: 0.06628777292983394
    Standard Deviation CHKEW: 0.020561588142854803
    Standard Deviation TRU: 0.009188674096871671
    Standard Deviation UHAL: 0.018773257890717144
    Standard Deviation NI: 0.015560696664548797
    Standard Deviation TRMB: 0.015870346400537826
    Standard Deviation CASY: 0.018925699457293756
    Standard Deviation KEY: 0.038954923469798154
    Standard Deviation RGA: 0.012325121333947446
    Standard Deviation PARAA: 0.008994325219587604
    Standard Deviation USFD: 0.04520845698807251
    Standard Deviation NRG: 0.01678866995707969
    Standard Deviation SAIA: 0.034644930430261005
    Standard Deviation TWLO: 0.013983272064602356
    Standard Deviation CDAY: 0.020738891156865655
    Standard Deviation PAYC: 0.012875152882458305
    Standard Deviation PNR: 0.028895853606854937
    Standard Deviation CX: 0.018295345067238483
    Standard Deviation PR: 0.014706617182290658
    Standard Deviation FMS: 0.057448487791251536
    Standard Deviation PAG: 0.02275748404432337
    Standard Deviation CHKEZ: 0.018163322392832595
    Standard Deviation KMX: 0.020845267151277855
    Standard Deviation WRK: 0.01909382451494852
    Standard Deviation EME: 0.020106337166745702
    Standard Deviation TEVA: 0.029250025109986054
    Standard Deviation FIVE: 0.05109863955196373
    Standard Deviation OTEX: 0.011501953390871759
    Standard Deviation DOX: 0.004406880038839405
    Standard Deviation U: 0.015356615996607452
    Standard Deviation CCK: 0.015569816953037274
    Standard Deviation TPG: 0.04857696821873236
    Standard Deviation DKS: 0.017143447625642048
    Standard Deviation BAP: 0.030894733121320173
    Standard Deviation OZKAP: 0.011899096708027959
    Standard Deviation WSM: 0.015099819858367693
    Standard Deviation DINO: 0.020218263271746303
    Standard Deviation TFII: 0.016799189535553136
    Standard Deviation REXR: 0.014544872383861398
    Standard Deviation PATH: 0.008601151198299609
    Standard Deviation LAMR: 0.04457926641653896
    Standard Deviation XPO: 0.021651044298246696
    Standard Deviation PFGC: 0.01401099406219324
    Standard Deviation TFX: 0.032736338685351286
    Standard Deviation CHRW: 0.01601380605460584
    Standard Deviation CPT: 0.03177394351462154
    Standard Deviation WTRG: 0.013929941094620747
    Standard Deviation FFIV: 0.01346382513415412
    Standard Deviation FTI: 0.03751507955370661
    Standard Deviation CZR: 0.012127372613281385
    Standard Deviation DBX: 0.03974807140891368
    Standard Deviation AFG: 0.01294417419550771
    Standard Deviation HII: 0.02121005389643778
    Standard Deviation VLYPP: 0.016656080861034553
    Standard Deviation EMN: 0.016376200118153363
    Standard Deviation OZK: 0.005792455228337911
    Standard Deviation BJ: 0.027696998796327595
    Standard Deviation TECH: 0.03690661187107399
    Standard Deviation NTNX: 0.036011523126026686
    Standard Deviation BIO: 0.012623194861969252
    Standard Deviation PRH: 0.012183434394898494
    Standard Deviation ALLE: 0.0774895233212362
    Standard Deviation EQH: 0.01957208954024951
    Standard Deviation CET: 0.016202187695620887
    Standard Deviation IBKR: 0.015813203506878815
    Standard Deviation UHS: 0.02009949853111399
    Standard Deviation CHE: 0.01888821658490358
    Standard Deviation COTY: 0.02479524421302447
    Standard Deviation CRL: 0.022080054339363615
    Standard Deviation CHDN: 0.016431149982389235
    Standard Deviation CHKEL: 0.01745053097200536
    Standard Deviation QGEN: 0.01353430342983422
    Standard Deviation WMS: 0.019534428699765377
    Standard Deviation PEAK: 0.018663341998538217
    Standard Deviation MKTX: 0.014729672429614217
    Standard Deviation LCID: 0.023692519187529893
    Standard Deviation SNX: 0.017960307459525288
    Standard Deviation CHWY: 0.02559600030099273
    Standard Deviation AGCO: 0.021466427144408176
    Standard Deviation JNPR: 0.014416814567584348
    Standard Deviation TTC: 0.09250856279959566
    Standard Deviation QRVO: 0.08088100709617069
    Standard Deviation PSO: 0.01113500844075714
    Standard Deviation TIMB: 0.018662601677108828
    Standard Deviation BURL: 0.017119399097546308
    Standard Deviation AIZ: 0.01275974093339795
    Standard Deviation SCI: 0.010212334959908228
    Standard Deviation UNM: 0.009632803675392761
    Standard Deviation UWMC: 0.030376415977670528
    Standard Deviation Z: 0.013704539376816118
    Standard Deviation DUOL: 0.021039477095449573
    Standard Deviation NLY: 0.012376631932470412
    Standard Deviation ESLT: 0.027452080823760806
    Standard Deviation TOL: 0.02810985033544676
    Standard Deviation SBS: 0.026778963474019545
    Standard Deviation CLH: 0.013368555418366738
    Standard Deviation BRKR: 0.01704812859299053
    Standard Deviation BSAC: 0.020674322511129856
    Standard Deviation PCTY: 0.0219587375302828
    Standard Deviation BLD: 0.02516555640178336
    Standard Deviation SKM: 0.022621359168137494
    Standard Deviation KEP: 0.026989059882978644
    Standard Deviation CLF: 0.03569334198005764
    Standard Deviation VIPS: 0.018888459515052025
    Standard Deviation PRS: 0.010779106470037018
    Standard Deviation MTN: 0.02051934152620533
    Standard Deviation NVT: 0.015246706867091158
    Standard Deviation KNSL: 0.01857658885355673
    Standard Deviation MEDP: 0.029485851510339896
    Standard Deviation FND: 0.019092631861164354
    Standard Deviation LSXMB: 0.009906041910005945
    Standard Deviation ITT: 0.026411823129987355
    Standard Deviation CUBE: 0.03968266531662571
    Standard Deviation ZG: 0.02288264459136848
    Standard Deviation RGEN: 0.019432205864529196
    Standard Deviation LSXMK: 0.01876874158970407
    Standard Deviation BXP: 0.013382783069853325
    Standard Deviation LSXMA: 0.030675809461475852
    Standard Deviation TTEK: 0.018789397896640334
    Standard Deviation DLB: 0.011003244656018166
    Standard Deviation ATR: 0.05460159701587028
    Standard Deviation DOCU: 0.015009515758228224
    Standard Deviation MNSO: 0.021847583582289744
    Standard Deviation HSIC: 0.021119477476862034
    Standard Deviation SYT: 0.03465799661454445
    Standard Deviation RHI: 0.01968905600743159
    Standard Deviation PNW: 0.014938371559797914
    Standard Deviation KNX: 0.028580272199460837
    Standard Deviation FUTU: 0.017123261337455675
    Standard Deviation RRC: 0.04508585245901582
    Standard Deviation WWD: 0.013115779328137553
    Standard Deviation ALV: 0.01951792104188878
    Standard Deviation ACGLO: 0.018506019650814706
    Standard Deviation PARA: 0.015185410648594945
    Standard Deviation AR: 0.0060939764809504445
    Standard Deviation CW: 0.012638344115980757
    Standard Deviation MTCH: 0.02558673372834438
    Standard Deviation EWBC: 0.0160803524088522
    Standard Deviation SSL: 0.016149737358122706
    Standard Deviation IEP: 0.014611196553899442
    Standard Deviation MUSA: 0.01157174192737509
    Standard Deviation ORI: 0.02221794219039792
    Standard Deviation AIU: 0.03665951272443347
    Standard Deviation ALLY: 0.02700435903232572
    Standard Deviation EGP: 0.025120814051689023
    Standard Deviation GGB: 0.01594826252312302
    Standard Deviation DVA: 0.02095894303723062
    Standard Deviation FCN: 0.02355096378062972
    Standard Deviation AAL: 0.018707558418539133
    Standard Deviation SKX: 0.012224153907160295
    Standard Deviation ONON: 0.009780165717029584
    Standard Deviation FBIN: 0.015199578582969117
    Standard Deviation X: 0.018353968232352993
    Standard Deviation OHI: 0.008113032960378597
    Standard Deviation GLOB: 0.02644401466337184
    Standard Deviation RL: 0.012074575630172635
    Standard Deviation BWA: 0.018584883755280543
    Standard Deviation TOST: 0.025336677363723744
    Standard Deviation AFRM: 0.0295510933187219
    Standard Deviation LSCC: 0.015676067873266173
    Standard Deviation BIRK: 0.013032264899606202
    Standard Deviation FRT: 0.03956501768449622
    Standard Deviation CYBR: 0.010120271479203042
    Standard Deviation NOV: 0.022187797916238157
    Standard Deviation ETSY: 0.030243767107720514
    Standard Deviation SRPT: 0.07043977727208624
    Standard Deviation SUZ: 0.015716487254691394
    Standard Deviation SEIC: 0.028838058589240788
    Standard Deviation YMM: 0.017684784609426414
    Standard Deviation JAZZ: 0.016749509831565163
    Standard Deviation LEA: 0.021854522258676734
    Standard Deviation GWRE: 0.012018694343578277
    Standard Deviation ARMK: 0.014513865358529684
    Standard Deviation CACI: 0.018242251813223136
    Standard Deviation TX: 0.028611489779902292
    Standard Deviation ESTC: 0.01772371789180935
    Standard Deviation WEX: 0.01512086538639953
    Standard Deviation SWN: 0.011311454389534598
    Standard Deviation CNM: 0.016370138100452545
    Standard Deviation PCOR: 0.033614859554053064
    Standard Deviation MSTR: 0.01778192332368532
    Standard Deviation DCI: 0.014817774166097843
    Standard Deviation GJS: 0.017169947933963984
    Standard Deviation LNW: 0.018244659728389456
    Standard Deviation VOYA: 0.021186612812206306
    Standard Deviation COLD: 0.015597877750905497
    Standard Deviation APPF: 0.026076996225035662
    Standard Deviation PEN: 0.02954377626284152
    Standard Deviation KBR: 0.049501989678021205
    Standard Deviation NYT: 0.026197226473428224
    Standard Deviation RRX: 0.0282832612070438
    Standard Deviation OLED: 0.01718879590785351
    Standard Deviation JEF: 0.014093890603907932
    Standard Deviation PRI: 0.0290039622167528
    Standard Deviation HLI: 0.014953096801968812
    Standard Deviation WCC: 0.012056422672032689
    Standard Deviation LAD: 0.014226048598730268
    Standard Deviation BBWI: 0.014774061830171445
    Standard Deviation STN: 0.021315154044496045
    Standard Deviation CCCS: 0.01506591361225865
    Standard Deviation NNN: 0.015138747113964008
    Standard Deviation RGLD: 0.02224327537235112
    Standard Deviation AU: 0.012049840371987641
    Standard Deviation NFE: 0.02039798012255807
    Standard Deviation BWXT: 0.013625809386936152
    Standard Deviation TXRH: 0.011086167075094537
    Standard Deviation WF: 0.017292264402194838
    Standard Deviation XPEV: 0.021292005501999712
    Standard Deviation HOOD: 0.015327912369921317
    Standard Deviation CAR: 0.017937831139502345
    Standard Deviation WFRD: 0.01971240350943452
    Standard Deviation GNTX: 0.01594286166131635
    Standard Deviation DAR: 0.01872780605519445
    Standard Deviation CAE: 0.027569349080892026
    Standard Deviation ACGLN: 0.02375989104641854
    Standard Deviation OGE: 0.01889591801364931
    Standard Deviation CMSD: 0.02345831632237463
    Standard Deviation IONS: 0.017383159171839354
    Standard Deviation BERY: 0.02089393498252833
    Standard Deviation CART: 0.014943385190897955
    Standard Deviation WSC: 0.0181894196532756
    Standard Deviation ONBPP: 0.02185744289012607
    Standard Deviation CMSC: 0.01086395086946831
    Standard Deviation GTLB: 0.012461959496990048
    Standard Deviation LBTYK: 0.013620245720864179
    Standard Deviation ACHC: 0.024791867902699674
    Standard Deviation WBS: 0.05116686220703007
    Standard Deviation INFA: 0.034367774660861765
    Standard Deviation CMSA: 0.016282124144251592
    Standard Deviation LBTYB: 0.04077836238539458
    Standard Deviation KGC: 0.009587243725610945
    Standard Deviation NYCB: 0.03146693904968724
    Standard Deviation MAT: 0.03578174914903776
    Standard Deviation CHRD: 0.014942263821190644
    Standard Deviation FIX: 0.025874731425190266
    Standard Deviation INGR: 0.03516462047033153
    Standard Deviation CSAN: 0.016863736442645695
    Standard Deviation MTDR: 0.021960325126443966
    Standard Deviation ROIV: 0.016276476651154097
    Standard Deviation ONBPO: 0.01903843795923615
    

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

    rendimento atteso portafoglio 0.24848922040500573
    

# VAR Calculation at 95% Confidence Interval


```python
VaR = norm.ppf(0.95)*std_port
print(VaR)
print("Normal VaR 95th CI       :      ", round(Capitale*VaR,2))
```

    0.28535376141058955
    Normal VaR 95th CI       :       285353.76
    

# Beta Calculation


```python
# Scarica i dati del NASDAQ Composite Index
nasdaq_data = pdr.get_data_yahoo('^IXIC',inizio,fine)['Close']
```

    [*********************100%%**********************]  1 of 1 completed
    


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


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Input In [185], in <cell line: 6>()
          6 for stock in ritorni:
          7     stock_returns_single = ritorni[stock]
    ----> 8     covariance = np.cov(stock_returns_single, nasdaq_returns)[0, 1]
          9     variance_market = np.var(nasdaq_returns)
         11     # Calculate Beta and store in the DataFrame
    

    File <__array_function__ internals>:5, in cov(*args, **kwargs)
    

    File ~\anaconda3\lib\site-packages\numpy\lib\function_base.py:2477, in cov(m, y, rowvar, bias, ddof, fweights, aweights, dtype)
       2475     if not rowvar and y.shape[0] != 1:
       2476         y = y.T
    -> 2477     X = np.concatenate((X, y), axis=0)
       2479 if ddof is None:
       2480     if bias == 0:
    

    File <__array_function__ internals>:5, in concatenate(*args, **kwargs)
    

    ValueError: all the input array dimensions for the concatenation axis must match exactly, but along dimension 1, the array at index 0 has size 30 and the array at index 1 has size 250



```python
beta_ritorni
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>AAL</th>
      <th>AAPL</th>
      <th>ABBV</th>
      <th>ABEV</th>
      <th>ABNB</th>
      <th>ABT</th>
      <th>ACGL</th>
      <th>ACGLN</th>
      <th>ACGLO</th>
      <th>...</th>
      <th>YUM</th>
      <th>YUMC</th>
      <th>Z</th>
      <th>ZBH</th>
      <th>ZBRA</th>
      <th>ZG</th>
      <th>ZM</th>
      <th>ZS</th>
      <th>ZTO</th>
      <th>ZTS</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2023-10-12</th>
      <td>-0.039160</td>
      <td>-0.033654</td>
      <td>0.005061</td>
      <td>-0.006964</td>
      <td>-0.023077</td>
      <td>-0.031000</td>
      <td>-0.026446</td>
      <td>0.000361</td>
      <td>-0.015455</td>
      <td>-0.006417</td>
      <td>...</td>
      <td>-0.017993</td>
      <td>-0.021636</td>
      <td>-0.033526</td>
      <td>-0.024964</td>
      <td>-0.028965</td>
      <td>-0.033852</td>
      <td>-0.020827</td>
      <td>-0.013869</td>
      <td>-0.016360</td>
      <td>-0.014919</td>
    </tr>
    <tr>
      <th>2023-10-13</th>
      <td>0.014412</td>
      <td>-0.028192</td>
      <td>-0.010293</td>
      <td>-0.002293</td>
      <td>0.003937</td>
      <td>-0.015004</td>
      <td>0.007540</td>
      <td>0.013365</td>
      <td>-0.013372</td>
      <td>-0.016393</td>
      <td>...</td>
      <td>0.005419</td>
      <td>-0.001301</td>
      <td>-0.016770</td>
      <td>0.015382</td>
      <td>-0.018732</td>
      <td>-0.015625</td>
      <td>-0.011344</td>
      <td>-0.012482</td>
      <td>0.004158</td>
      <td>0.000402</td>
    </tr>
    <tr>
      <th>2023-10-16</th>
      <td>0.008958</td>
      <td>0.018771</td>
      <td>-0.000727</td>
      <td>-0.004934</td>
      <td>0.003922</td>
      <td>0.008946</td>
      <td>0.014416</td>
      <td>0.018298</td>
      <td>0.009428</td>
      <td>-0.003535</td>
      <td>...</td>
      <td>0.013433</td>
      <td>-0.010979</td>
      <td>-0.006308</td>
      <td>0.007191</td>
      <td>0.038904</td>
      <td>-0.007937</td>
      <td>0.006853</td>
      <td>0.021897</td>
      <td>-0.005797</td>
      <td>0.004588</td>
    </tr>
    <tr>
      <th>2023-10-17</th>
      <td>0.008072</td>
      <td>0.000000</td>
      <td>-0.008785</td>
      <td>0.013245</td>
      <td>-0.015625</td>
      <td>0.004154</td>
      <td>-0.000434</td>
      <td>0.010735</td>
      <td>-0.021016</td>
      <td>-0.005068</td>
      <td>...</td>
      <td>0.008358</td>
      <td>-0.004704</td>
      <td>-0.001646</td>
      <td>0.014660</td>
      <td>-0.009211</td>
      <td>-0.003636</td>
      <td>0.010446</td>
      <td>0.000464</td>
      <td>0.000000</td>
      <td>-0.005252</td>
    </tr>
    <tr>
      <th>2023-10-18</th>
      <td>-0.033719</td>
      <td>-0.048576</td>
      <td>-0.007395</td>
      <td>0.000670</td>
      <td>-0.027778</td>
      <td>-0.028160</td>
      <td>0.037117</td>
      <td>-0.013507</td>
      <td>0.001193</td>
      <td>-0.007641</td>
      <td>...</td>
      <td>0.012977</td>
      <td>0.004348</td>
      <td>-0.035092</td>
      <td>-0.003753</td>
      <td>-0.018170</td>
      <td>-0.035280</td>
      <td>-0.013628</td>
      <td>-0.010970</td>
      <td>-0.014994</td>
      <td>-0.027086</td>
    </tr>
    <tr>
      <th>2023-10-19</th>
      <td>0.008287</td>
      <td>0.007923</td>
      <td>-0.002161</td>
      <td>-0.025188</td>
      <td>0.000000</td>
      <td>-0.018008</td>
      <td>-0.001256</td>
      <td>-0.019544</td>
      <td>-0.010125</td>
      <td>-0.007187</td>
      <td>...</td>
      <td>-0.003554</td>
      <td>-0.019010</td>
      <td>-0.038565</td>
      <td>-0.012996</td>
      <td>-0.014967</td>
      <td>-0.034805</td>
      <td>-0.009687</td>
      <td>-0.007805</td>
      <td>-0.010994</td>
      <td>-0.008847</td>
    </tr>
    <tr>
      <th>2023-10-20</th>
      <td>-0.002557</td>
      <td>-0.032314</td>
      <td>-0.014704</td>
      <td>0.004879</td>
      <td>0.004082</td>
      <td>-0.029007</td>
      <td>0.014040</td>
      <td>-0.012175</td>
      <td>0.019856</td>
      <td>0.032575</td>
      <td>...</td>
      <td>-0.005060</td>
      <td>-0.012279</td>
      <td>-0.010916</td>
      <td>-0.002385</td>
      <td>-0.006991</td>
      <td>-0.012020</td>
      <td>-0.010423</td>
      <td>-0.040693</td>
      <td>0.009406</td>
      <td>-0.005653</td>
    </tr>
    <tr>
      <th>2023-10-23</th>
      <td>0.001007</td>
      <td>0.018953</td>
      <td>0.000694</td>
      <td>-0.010258</td>
      <td>-0.004065</td>
      <td>0.033479</td>
      <td>-0.010333</td>
      <td>-0.010875</td>
      <td>-0.001770</td>
      <td>0.006510</td>
      <td>...</td>
      <td>-0.009921</td>
      <td>0.001554</td>
      <td>-0.012320</td>
      <td>0.005738</td>
      <td>0.003618</td>
      <td>-0.010050</td>
      <td>-0.009399</td>
      <td>0.001788</td>
      <td>-0.010165</td>
      <td>-0.003531</td>
    </tr>
    <tr>
      <th>2023-10-24</th>
      <td>-0.033839</td>
      <td>-0.007086</td>
      <td>0.002543</td>
      <td>0.010917</td>
      <td>0.020408</td>
      <td>0.013456</td>
      <td>-0.010127</td>
      <td>0.019057</td>
      <td>0.028369</td>
      <td>0.021393</td>
      <td>...</td>
      <td>0.009683</td>
      <td>0.039565</td>
      <td>0.023129</td>
      <td>-0.003043</td>
      <td>0.003020</td>
      <td>0.025114</td>
      <td>0.016522</td>
      <td>0.014217</td>
      <td>0.017972</td>
      <td>0.003724</td>
    </tr>
    <tr>
      <th>2023-10-25</th>
      <td>-0.021204</td>
      <td>-0.015165</td>
      <td>-0.013492</td>
      <td>-0.007177</td>
      <td>-0.016000</td>
      <td>-0.028768</td>
      <td>-0.013079</td>
      <td>0.000839</td>
      <td>-0.005747</td>
      <td>-0.006819</td>
      <td>...</td>
      <td>0.003336</td>
      <td>-0.014925</td>
      <td>-0.029972</td>
      <td>-0.011446</td>
      <td>-0.033995</td>
      <td>-0.029450</td>
      <td>-0.028806</td>
      <td>-0.043571</td>
      <td>-0.001681</td>
      <td>-0.020644</td>
    </tr>
    <tr>
      <th>2023-10-26</th>
      <td>0.008801</td>
      <td>0.009964</td>
      <td>-0.024606</td>
      <td>-0.000413</td>
      <td>0.016260</td>
      <td>-0.025570</td>
      <td>0.004382</td>
      <td>-0.012457</td>
      <td>-0.002890</td>
      <td>0.002452</td>
      <td>...</td>
      <td>-0.012966</td>
      <td>-0.013258</td>
      <td>-0.003666</td>
      <td>-0.005017</td>
      <td>0.029762</td>
      <td>0.000537</td>
      <td>-0.014084</td>
      <td>-0.019034</td>
      <td>-0.009684</td>
      <td>-0.034215</td>
    </tr>
    <tr>
      <th>2023-10-27</th>
      <td>-0.014764</td>
      <td>-0.020628</td>
      <td>0.007969</td>
      <td>-0.043182</td>
      <td>-0.024000</td>
      <td>-0.011951</td>
      <td>-0.012024</td>
      <td>-0.013220</td>
      <td>-0.005797</td>
      <td>-0.017123</td>
      <td>...</td>
      <td>0.005811</td>
      <td>0.007869</td>
      <td>0.014455</td>
      <td>0.000679</td>
      <td>0.011473</td>
      <td>0.012614</td>
      <td>-0.003529</td>
      <td>-0.001552</td>
      <td>0.001275</td>
      <td>-0.012906</td>
    </tr>
    <tr>
      <th>2023-10-30</th>
      <td>-0.015569</td>
      <td>0.023810</td>
      <td>0.012305</td>
      <td>0.021306</td>
      <td>-0.004098</td>
      <td>0.025068</td>
      <td>0.001616</td>
      <td>0.018682</td>
      <td>-0.010496</td>
      <td>-0.002987</td>
      <td>...</td>
      <td>0.003600</td>
      <td>0.011617</td>
      <td>0.009586</td>
      <td>0.002132</td>
      <td>0.012501</td>
      <td>0.011132</td>
      <td>0.011975</td>
      <td>0.015806</td>
      <td>0.008068</td>
      <td>0.006217</td>
    </tr>
    <tr>
      <th>2023-10-31</th>
      <td>0.021746</td>
      <td>-0.002683</td>
      <td>0.002819</td>
      <td>-0.005004</td>
      <td>0.041152</td>
      <td>0.011458</td>
      <td>0.016667</td>
      <td>0.045849</td>
      <td>0.019446</td>
      <td>0.018972</td>
      <td>...</td>
      <td>0.008259</td>
      <td>-0.010542</td>
      <td>-0.069797</td>
      <td>0.009670</td>
      <td>-0.001621</td>
      <td>-0.068676</td>
      <td>-0.000333</td>
      <td>0.011989</td>
      <td>-0.007161</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2023-11-01</th>
      <td>-0.004934</td>
      <td>-0.000897</td>
      <td>0.018739</td>
      <td>0.009137</td>
      <td>0.019763</td>
      <td>0.009975</td>
      <td>0.004865</td>
      <td>0.038302</td>
      <td>0.005780</td>
      <td>0.004900</td>
      <td>...</td>
      <td>0.003641</td>
      <td>-0.152207</td>
      <td>0.005793</td>
      <td>0.006321</td>
      <td>-0.054624</td>
      <td>0.007318</td>
      <td>0.000834</td>
      <td>-0.012855</td>
      <td>-0.002970</td>
      <td>-0.035414</td>
    </tr>
    <tr>
      <th>2023-11-02</th>
      <td>0.015652</td>
      <td>0.023339</td>
      <td>0.020693</td>
      <td>0.005545</td>
      <td>0.011628</td>
      <td>-0.033230</td>
      <td>0.009999</td>
      <td>-0.045111</td>
      <td>0.025287</td>
      <td>0.020478</td>
      <td>...</td>
      <td>0.024485</td>
      <td>0.027379</td>
      <td>-0.026330</td>
      <td>0.048825</td>
      <td>0.045507</td>
      <td>-0.029338</td>
      <td>0.020823</td>
      <td>0.016215</td>
      <td>0.018298</td>
      <td>0.062533</td>
    </tr>
    <tr>
      <th>2023-11-03</th>
      <td>0.043553</td>
      <td>0.050877</td>
      <td>-0.005181</td>
      <td>-0.012844</td>
      <td>0.034483</td>
      <td>0.061818</td>
      <td>-0.001251</td>
      <td>-0.004538</td>
      <td>0.026906</td>
      <td>0.022456</td>
      <td>...</td>
      <td>0.016657</td>
      <td>-0.008737</td>
      <td>0.061690</td>
      <td>-0.013158</td>
      <td>0.035266</td>
      <td>0.058722</td>
      <td>0.027905</td>
      <td>0.032540</td>
      <td>0.026745</td>
      <td>0.008203</td>
    </tr>
    <tr>
      <th>2023-11-06</th>
      <td>-0.013667</td>
      <td>-0.026711</td>
      <td>0.014605</td>
      <td>-0.001556</td>
      <td>0.018519</td>
      <td>-0.036448</td>
      <td>-0.008765</td>
      <td>-0.004559</td>
      <td>0.001638</td>
      <td>0.007477</td>
      <td>...</td>
      <td>0.000396</td>
      <td>-0.009035</td>
      <td>-0.015389</td>
      <td>0.005057</td>
      <td>-0.038637</td>
      <td>-0.021207</td>
      <td>-0.018574</td>
      <td>-0.005475</td>
      <td>-0.006105</td>
      <td>0.005794</td>
    </tr>
    <tr>
      <th>2023-11-07</th>
      <td>0.027992</td>
      <td>0.007719</td>
      <td>0.014451</td>
      <td>0.006799</td>
      <td>0.007273</td>
      <td>0.024626</td>
      <td>-0.002105</td>
      <td>0.002936</td>
      <td>0.001090</td>
      <td>0.000000</td>
      <td>...</td>
      <td>-0.007200</td>
      <td>-0.010451</td>
      <td>0.024252</td>
      <td>-0.030833</td>
      <td>0.009805</td>
      <td>0.026389</td>
      <td>0.014073</td>
      <td>0.048449</td>
      <td>-0.013923</td>
      <td>0.024453</td>
    </tr>
    <tr>
      <th>2023-11-08</th>
      <td>-0.010403</td>
      <td>0.022128</td>
      <td>0.005885</td>
      <td>-0.000914</td>
      <td>-0.003610</td>
      <td>-0.028246</td>
      <td>-0.000633</td>
      <td>-0.015104</td>
      <td>0.012520</td>
      <td>0.006725</td>
      <td>...</td>
      <td>0.001036</td>
      <td>-0.018876</td>
      <td>-0.006051</td>
      <td>-0.010101</td>
      <td>-0.007691</td>
      <td>-0.007848</td>
      <td>-0.013718</td>
      <td>0.014995</td>
      <td>-0.001246</td>
      <td>0.022194</td>
    </tr>
    <tr>
      <th>2023-11-09</th>
      <td>-0.015084</td>
      <td>-0.021649</td>
      <td>-0.002625</td>
      <td>-0.028093</td>
      <td>-0.018116</td>
      <td>-0.018188</td>
      <td>-0.012561</td>
      <td>0.006538</td>
      <td>-0.018817</td>
      <td>-0.023727</td>
      <td>...</td>
      <td>-0.007165</td>
      <td>-0.011910</td>
      <td>-0.042880</td>
      <td>-0.007057</td>
      <td>-0.018989</td>
      <td>-0.040098</td>
      <td>-0.019085</td>
      <td>-0.018740</td>
      <td>-0.014969</td>
      <td>-0.001405</td>
    </tr>
    <tr>
      <th>2023-11-10</th>
      <td>0.006776</td>
      <td>0.004255</td>
      <td>0.021874</td>
      <td>0.003984</td>
      <td>0.003690</td>
      <td>0.022767</td>
      <td>0.004490</td>
      <td>0.015118</td>
      <td>0.013151</td>
      <td>0.021236</td>
      <td>...</td>
      <td>0.009943</td>
      <td>0.028280</td>
      <td>0.014934</td>
      <td>0.012486</td>
      <td>0.018072</td>
      <td>0.009378</td>
      <td>0.023248</td>
      <td>0.025425</td>
      <td>0.018151</td>
      <td>-0.007677</td>
    </tr>
    <tr>
      <th>2023-11-13</th>
      <td>-0.009311</td>
      <td>-0.001695</td>
      <td>-0.008584</td>
      <td>0.000361</td>
      <td>0.007353</td>
      <td>0.008464</td>
      <td>0.019368</td>
      <td>0.007214</td>
      <td>0.014603</td>
      <td>0.006470</td>
      <td>...</td>
      <td>-0.005081</td>
      <td>0.003156</td>
      <td>-0.026976</td>
      <td>0.003794</td>
      <td>-0.017315</td>
      <td>-0.027872</td>
      <td>-0.008379</td>
      <td>0.017882</td>
      <td>-0.004146</td>
      <td>-0.002894</td>
    </tr>
    <tr>
      <th>2023-11-14</th>
      <td>0.038619</td>
      <td>0.039898</td>
      <td>0.014286</td>
      <td>-0.004184</td>
      <td>0.021898</td>
      <td>0.063198</td>
      <td>0.013467</td>
      <td>-0.012360</td>
      <td>0.027186</td>
      <td>0.032599</td>
      <td>...</td>
      <td>0.017716</td>
      <td>0.020449</td>
      <td>0.122375</td>
      <td>0.021263</td>
      <td>0.064409</td>
      <td>0.121923</td>
      <td>0.019987</td>
      <td>0.050065</td>
      <td>0.015404</td>
      <td>0.022626</td>
    </tr>
    <tr>
      <th>2023-11-15</th>
      <td>0.017830</td>
      <td>0.013878</td>
      <td>0.003041</td>
      <td>-0.003332</td>
      <td>0.000000</td>
      <td>0.013183</td>
      <td>0.009477</td>
      <td>-0.033216</td>
      <td>-0.001557</td>
      <td>0.004002</td>
      <td>...</td>
      <td>-0.007136</td>
      <td>0.009469</td>
      <td>0.011477</td>
      <td>0.022485</td>
      <td>0.006028</td>
      <td>0.009809</td>
      <td>0.015453</td>
      <td>-0.013523</td>
      <td>-0.003280</td>
      <td>0.011410</td>
    </tr>
    <tr>
      <th>2023-11-16</th>
      <td>0.005194</td>
      <td>-0.018519</td>
      <td>0.009042</td>
      <td>0.004942</td>
      <td>-0.003571</td>
      <td>-0.016128</td>
      <td>0.023061</td>
      <td>0.012703</td>
      <td>-0.004158</td>
      <td>-0.004429</td>
      <td>...</td>
      <td>0.009556</td>
      <td>0.000000</td>
      <td>0.004193</td>
      <td>0.009502</td>
      <td>-0.006822</td>
      <td>0.001278</td>
      <td>-0.005334</td>
      <td>0.008778</td>
      <td>-0.011106</td>
      <td>0.010995</td>
    </tr>
    <tr>
      <th>2023-11-17</th>
      <td>-0.009108</td>
      <td>0.008203</td>
      <td>-0.000105</td>
      <td>0.000145</td>
      <td>-0.007168</td>
      <td>0.006889</td>
      <td>-0.007082</td>
      <td>-0.001314</td>
      <td>-0.013570</td>
      <td>0.001779</td>
      <td>...</td>
      <td>-0.001330</td>
      <td>0.001091</td>
      <td>-0.039794</td>
      <td>0.001076</td>
      <td>0.011788</td>
      <td>-0.037273</td>
      <td>0.011356</td>
      <td>0.006929</td>
      <td>-0.061980</td>
      <td>-0.009856</td>
    </tr>
    <tr>
      <th>2023-11-20</th>
      <td>0.007335</td>
      <td>0.008950</td>
      <td>0.009278</td>
      <td>0.000072</td>
      <td>0.003610</td>
      <td>0.020055</td>
      <td>0.016575</td>
      <td>0.010287</td>
      <td>0.012698</td>
      <td>0.002220</td>
      <td>...</td>
      <td>0.003603</td>
      <td>0.008281</td>
      <td>-0.010233</td>
      <td>0.007433</td>
      <td>0.011559</td>
      <td>-0.008221</td>
      <td>0.029320</td>
      <td>0.023951</td>
      <td>-0.021286</td>
      <td>0.007208</td>
    </tr>
    <tr>
      <th>2023-11-21</th>
      <td>0.087208</td>
      <td>-0.021774</td>
      <td>-0.004231</td>
      <td>0.003037</td>
      <td>-0.017986</td>
      <td>-0.022205</td>
      <td>0.007016</td>
      <td>0.019773</td>
      <td>-0.004180</td>
      <td>-0.007975</td>
      <td>...</td>
      <td>0.000937</td>
      <td>-0.007780</td>
      <td>-0.009046</td>
      <td>0.003556</td>
      <td>0.008479</td>
      <td>-0.009626</td>
      <td>-0.000909</td>
      <td>-0.003126</td>
      <td>0.000453</td>
      <td>0.015165</td>
    </tr>
    <tr>
      <th>2023-11-22</th>
      <td>0.000565</td>
      <td>0.014839</td>
      <td>0.003514</td>
      <td>-0.001874</td>
      <td>0.010989</td>
      <td>0.017505</td>
      <td>0.007752</td>
      <td>-0.002090</td>
      <td>-0.002099</td>
      <td>0.001787</td>
      <td>...</td>
      <td>0.005458</td>
      <td>-0.015901</td>
      <td>0.037037</td>
      <td>0.008237</td>
      <td>0.015512</td>
      <td>0.032937</td>
      <td>-0.031999</td>
      <td>-0.003763</td>
      <td>0.006341</td>
      <td>0.003469</td>
    </tr>
  </tbody>
</table>
<p>30 rows × 1000 columns</p>
</div>




```python

```
