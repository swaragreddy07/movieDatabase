import pandas as pd

# reading the dataset using pandas
dataset = pd.read_csv('imdb_top_1000.csv')

# filling the missing values
   
dataset["Certificate"] = dataset["Certificate"].fillna("")
dataset["Meta_score"] = dataset["Meta_score"].fillna(0)
dataset["Gross"] = dataset["Gross"].fillna(0)

