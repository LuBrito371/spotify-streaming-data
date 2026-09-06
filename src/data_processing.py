import pandas as pd
from sklearn.model_selection import train_test_split

def carregar_e_limpar_dados(caminho_csv):
    df = pd.read_csv(caminho_csv) 
    df['Y'] = (df['genre'] == 'Pop').astype(int)
    
    df_estudo = df[['energy', 'danceability', 'explicit', 'Y']].copy()
    df_estudo = df_estudo.dropna()
    
    return df_estudo

def dividir_dados(df_estudo, semente=42):
    X = df_estudo[['energy', 'danceability', 'explicit']]
    y = df_estudo['Y']
    
    return train_test_split(X, y, test_size=0.2, random_state=semente, stratify=y)