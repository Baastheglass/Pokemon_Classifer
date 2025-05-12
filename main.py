import pandas as pd
from sklearn.preprocessing import LabelEncoder

#809 pokemons
#initialisation
csv_data = pd.read_csv('pokemon.csv')
encoder = LabelEncoder()
print(csv_data)

#sanitising dataset
csv_data = csv_data.dropna()
