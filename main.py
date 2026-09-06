import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from src.data_processing import carregar_e_limpar_dados, dividir_dados
from src.visualization import plot_distribuicao_continua, plotar_matriz_confusao
from src.naive_bayes import probabilidade_a_posteriori, classificar_naive_bayes

def main():
    caminho_arquivo = './data/spotify_artist_streaming_2020_2025.csv'
    df_estudo = carregar_e_limpar_dados(caminho_arquivo)
    
    total_observacoes = len(df_estudo)
    distribuicao = df_estudo['Y'].value_counts()
    percentual = df_estudo['Y'].value_counts(normalize=True) * 100
    
    print("********** INFORMAÇÕES DA BASE DE DADOS **********")
    print(f"Total de observações válidas: {total_observacoes}")
    print(f"Classe 0 (Não Pop): {distribuicao[0]} ({percentual[0]:.2f}%)")
    print(f"Classe 1 (Pop): {distribuicao[1]} ({percentual[1]:.2f}%)\n")
    
    semente = 42
    X_train, X_test, y_train, y_test = dividir_dados(df_estudo, semente)
    
    print("********** DIVISÃO TREINO/TESTE **********")
    print(f"Observações no Treinamento: {len(X_train)} | Teste: {len(X_test)}\n")
    
    df_train = X_train.copy()
    df_train['Y'] = y_train
    
    priors = df_train['Y'].value_counts(normalize=True)
    print("--- PROBABILIDADES A PRIORI ---")
    print(f"P(Y=0): {priors[0]:.4f} | P(Y=1): {priors[1]:.4f}\n")
    
    print("********** PARÂMETROS **********")
    for classe in [0, 1]:
        dados = df_train[df_train['Y'] == classe]
        print(f"Classe Y = {classe}:")
        print(f"  Energy -> Média: {dados['energy'].mean():.4f}, Std: {dados['energy'].std():.4f}")
        print(f"  Danceability -> Média: {dados['danceability'].mean():.4f}, Std: {dados['danceability'].std():.4f}")
        print(f"  P(X3=1 | Y={classe}): {dados['explicit'].mean():.4f}\n")
        
    print("********** GERANDO GRÁFICOS DE VEROSSIMILHANÇA **********")
    plot_distribuicao_continua(df_train, 'energy', 'Etapa 2: Verossimilhança de Energy')
    plot_distribuicao_continua(df_train, 'danceability', 'Etapa 2: Verossimilhança de Danceability')
    
    probabilidade_a_posteriori(0.65, df_train, 'energy', priors)

    print("\n********** CLASSIFICANDO CONJUNTO DE TESTE **********")
    y_pred = X_test.apply(lambda row: classificar_naive_bayes(row, priors, df_train), axis=1)
    
    matriz = confusion_matrix(y_test, y_pred)
    
    print("\n********** RESULTADOS DA AVALIAÇÃO NO CONJUNTO DE TESTE **********")
    print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precisão: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall:   {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred, zero_division=0):.4f}")
    
    plotar_matriz_confusao(matriz)

if __name__ == "__main__":
    main()