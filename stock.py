import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

symbol_dict = {
    "TOT": "Total",
    "XOM": "Exxon",
    "CVX": "Chevron",
    "COP": "ConocoPhillips",
    "VLO": "Valero Energy",
    "MSFT": "Microsoft",
    "IBM": "IBM",
    "TWX": "Time Warner",
    "CMCSA": "Comcast",
    "CVC": "Cablevision",
    "YHOO": "Yahoo",
    "DELL": "Dell",
    "HPQ": "HP",
    "AMZN": "Amazon",
    "TM": "Toyota",
    "CAJ": "Canon",
    "SNE": "Sony",
    "F": "Ford",
    "HMC": "Honda",
    "NAV": "Navistar",
    "NOC": "Northrop Grumman",
    "BA": "Boeing",
    "KO": "Coca Cola",
    "MMM": "3M",
    "MCD": "McDonald's",
    "PEP": "Pepsi",
    "K": "Kellogg",
    "UN": "Unilever",
    "MAR": "Marriott",
    "PG": "Procter Gamble",
    "CL": "Colgate-Palmolive",
    "GE": "General Electrics",
    "WFC": "Wells Fargo",
    "JPM": "JPMorgan Chase",
    "AIG": "AIG",
    "AXP": "American express",
    "BAC": "Bank of America",
    "GS": "Goldman Sachs",
    "AAPL": "Apple",
    "SAP": "SAP",
    "CSCO": "Cisco",
    "TXN": "Texas Instruments",
    "XRX": "Xerox",
    "WMT": "Wal-Mart",
    "HD": "Home Depot",
    "GSK": "GlaxoSmithKline",
    "PFE": "Pfizer",
    "SNY": "Sanofi-Aventis",
    "NVS": "Novartis",
    "KMB": "Kimberly-Clark",
    "R": "Ryder",
    "GD": "General Dynamics",
    "RTN": "Raytheon",
    "CVS": "CVS",
    "CAT": "Caterpillar",
    "DD": "DuPont de Nemours",
} #56 stocks


symbols, names = np.array(sorted(symbol_dict.items())).T

quotes = []

for symbol in symbols:
    print("Fetching quote history for %r" % symbol, file=sys.stderr)
    url = (
        "https://raw.githubusercontent.com/scikit-learn/examples-data/"
        "master/financial-data/{}.csv"
    )
    quotes.append(pd.read_csv(url.format(symbol)))

close_prices = np.vstack([q["close"] for q in quotes])
open_prices = np.vstack([q["open"] for q in quotes])

# The daily variations of the quotes are what carry the most information
variation = close_prices - open_prices

id1 = list(symbols).index("PFE")
id2 = list(symbols).index("NVS")
print("Pfizer:", id1)
print("Novartis:", id2)

# indices = [20, 1120, 1904, 1924]
indices =[  20,   50,  112,  188,  399,  412,  610, 1120, 1170, 1402, 1420, 1588, 1618, 1624, 1812, 1904, 1924, 1938, 2036, 2204, 2596, 2818, 2820, 2988]
n = 56
idx = [(k // n, k % n) for k in indices]
print(idx)
for i,j in idx:
    print(symbol_dict[symbols[i]], symbol_dict[symbols[j]])

import pdb; pdb.set_trace()

import matplotlib.pyplot as plt
plt.figure(figsize=(15,4))
plt.plot(variation[id1],label="PFE")
plt.plot(variation[id2],label="NVS")
plt.legend()
plt.savefig("data/pharma.png")
plt.close()

import pandas as pd
df = pd.DataFrame(variation[[id1,id2],:].T,columns=["PFE","NVS"])
df.to_csv("data/pharma.csv", index=False)

df = pd.DataFrame(variation.T,columns=symbols)
df.to_csv("data/stock.csv", index=False)

import pdb; pdb.set_trace()

# Marginal (normal scale)
plt.figure(figsize=(8, 8))
df = pd.DataFrame(variation.T,columns=symbols)
sns.jointplot(data=df,x="PFE",y="NVS",kind="scatter")
plt.savefig("Pharma.png", dpi=150)
plt.close()


# Marginal (log scale)
plt.figure(figsize=(8, 8))
log_df = df.apply(np.log)
sns.jointplot(data=log_df, x="PFE",y="NVS",kind="scatter")
plt.savefig("Pharma(log).png", dpi=150)
plt.close()

