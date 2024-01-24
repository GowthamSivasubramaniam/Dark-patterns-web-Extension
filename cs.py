import pandas as pd

tsv_file = 'dataset.tsv'
csv_file = 'dp.csv'

df = pd.read_csv(tsv_file, sep='\t')
df.to_csv(csv_file, index=False)
