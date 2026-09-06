import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


df = pd.read_csv('spotify_artist_streaming_2020_2025.csv') 


df['Y'] = (df['genre'] == 'Pop').astype(int)

X1 = 'energy'
X2 = 'danceability'
X3 = 'explicit'

df_estudo = df[[X1, X2, X3, 'Y']].copy()

df_estudo = df_estudo.dropna()

total_observacoes = len(df_estudo)
distribuicao = df_estudo['Y'].value_counts()
percentual = df_estudo['Y'].value_counts(normalize=True) * 100

print("--- INFORMAÇÕES DA BASE DE DADOS ---")
print(f"Total de observações válidas: {total_observacoes}")
print("\nDistribuição das Classes (Quantidade):")
print(f"Classe 0 (Não Pop): {distribuicao[0]}")
print(f"Classe 1 (Pop): {distribuicao[1]}")
print("\nDistribuição das Classes (Percentual):")
print(f"Classe 0: {percentual[0]:.2f}%")
print(f"Classe 1: {percentual[1]:.2f}%")

semente = 42

X = df_estudo[['energy', 'danceability', 'explicit']]
y = df_estudo['Y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=semente, stratify=y)

print("--- INFORMAÇÕES DA DIVISÃO TREINO/TESTE ---")
print(f"Semente aleatória: {semente}")
print(f"Proporção: 80% Treinamento, 20% Teste")
print(f"Observações no Treinamento: {len(X_train)}")
print(f"Observações no Teste: {len(X_test)}")
print(f"Desbalanceamento Treino - Pop(1): {y_train.mean():.2%}, Não Pop(0): {1 - y_train.mean():.2%}")
print(f"Desbalanceamento Teste - Pop(1): {y_test.mean():.2%}, Não Pop(0): {1 - y_test.mean():.2%}\n")

df_train = X_train.copy()
df_train['Y'] = y_train

priors = df_train['Y'].value_counts(normalize=True)
print("--- PROBABILIDADES A PRIORI ---")
print(f"P(Y=0): {priors[0]:.4f}")
print(f"P(Y=1): {priors[1]:.4f}\n")

print("--- PARÂMETROS DAS CARACTERÍSTICAS CONTÍNUAS ---")
for classe in [0, 1]:
    dados_classe = df_train[df_train['Y'] == classe]
    print(f"\nClasse Y = {classe}:")
    print(f"  Energy (X1) -> Media: {dados_classe['energy'].mean():.4f}, Desvio Padrao: {dados_classe['energy'].std():.4f}")
    print(f"  Danceability (X2) -> Media: {dados_classe['danceability'].mean():.4f}, Desvio Padrao: {dados_classe['danceability'].std():.4f}")

print("\n--- PARÂMETROS DA CARACTERÍSTICA CATEGÓRICA (explicit) ---")
for classe in [0, 1]:
    dados_classe = df_train[df_train['Y'] == classe]
    prob_explicit = dados_classe['explicit'].mean() # Proporção de 1s
    print(f"  P(X3=1 | Y={classe}) [Música Explícita]: {prob_explicit:.4f}")
    print(f"  P(X3=0 | Y={classe}) [Música Não Explícita]: {1 - prob_explicit:.4f}")

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- ETAPA 2 e 5: VISUALIZAÇÃO DAS VEROSSIMILHANÇAS ---

def plot_distribuicao_continua(df, feature_name, title):
    plt.figure(figsize=(8, 5))
    
    # Extraindo parâmetros (Note o recuo nestas linhas!)
    mu_0 = df[df['Y'] == 0][feature_name].mean()
    std_0 = df[df['Y'] == 0][feature_name].std()
    
    mu_1 = df[df['Y'] == 1][feature_name].mean()
    std_1 = df[df['Y'] == 1][feature_name].std()
    
    # Criando o eixo X (de 0 a 1, limite das nossas variáveis)
    x = np.linspace(0, 1, 1000)
    
    # Calculando a Verossimilhança (densidade) para cada ponto de x
    pdf_0 = norm.pdf(x, mu_0, std_0) # p(x|Y=0)
    pdf_1 = norm.pdf(x, mu_1, std_1) # p(x|Y=1)
    
    # Plotando as curvas
    plt.plot(x, pdf_0, label=f'Não Pop (Y=0) $\mu$={mu_0:.2f}', color='blue', linewidth=2)
    plt.plot(x, pdf_1, label=f'Pop (Y=1) $\mu$={mu_1:.2f}', color='red', linewidth=2)
    
    # Marcando o cruzamento visualmente (Fronteira preliminar)
    plt.fill_between(x, pdf_0, alpha=0.1, color='blue')
    plt.fill_between(x, pdf_1, alpha=0.1, color='red')
    
    plt.title(title)
    plt.xlabel(feature_name)
    plt.ylabel('Densidade p(x|Y=c)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()



print("Gerando gráficos de verossimilhança...")
plot_distribuicao_continua(df_train, 'energy', 'Etapa 2: Verossimilhança de Energy')
plot_distribuicao_continua(df_train, 'danceability', 'Etapa 2: Verossimilhança de Danceability')

def probabilidade_a_posteriori(x_val, df, feature_name, priors):
    # Parâmetros Y=0
    mu_0 = df[df['Y'] == 0][feature_name].mean()
    std_0 = df[df['Y'] == 0][feature_name].std()
    
    # Parâmetros Y=1
    mu_1 = df[df['Y'] == 1][feature_name].mean()
    std_1 = df[df['Y'] == 1][feature_name].std()
    
    # Calculando as Verossimilhanças p(x|Y=c)
    like_0 = norm.pdf(x_val, mu_0, std_0)
    like_1 = norm.pdf(x_val, mu_1, std_1)
    
    # Aplicando o Teorema de Bayes
    evidencia = (like_0 * priors[0]) + (like_1 * priors[1])
    post_0 = (like_0 * priors[0]) / evidencia
    post_1 = (like_1 * priors[1]) / evidencia
    
    print(f"\n--- Análise Bayesiana para {feature_name} = {x_val} ---")
    print(f"Verossimilhança p(x|Y=0): {like_0:.4f}")
    print(f"Verossimilhança p(x|Y=1): {like_1:.4f}")
    print(f"Posteriori P(Y=0|x): {post_0:.4%}")
    print(f"Posteriori P(Y=1|x): {post_1:.4%}")
    
    if post_1 > post_0:
        print("-> Decisão Bayesiana Univariada: POP (Y=1)")
    else:
        print("-> Decisão Bayesiana Univariada: NÃO POP (Y=0)")

# Vamos testar com um valor que vimos no gráfico que tem mais chance de ser Pop (ex: 0.65)
probabilidade_a_posteriori(0.65, df_train, 'energy', priors)

def classificar_naive_bayes(row, priors, df_train):
    log_probs = {}
    
    for c in [0, 1]:
        # 1. Log da Probabilidade a priori: log(P(Y=c))
        log_prob_c = np.log(priors[c])
        
        dados_classe = df_train[df_train['Y'] == c]
        
        # 2. Log da Verossimilhança das Contínuas (X1 e X2)
        for feature in ['energy', 'danceability']:
            mu = dados_classe[feature].mean()
            std = dados_classe[feature].std()
            verossimilhanca = norm.pdf(row[feature], mu, std)
            # Adiciona um valor ínfimo para evitar log(0)
            log_prob_c += np.log(verossimilhanca + 1e-9) 
            
        # 3. Log da Probabilidade da Categórica (X3)
        prob_explicit_1 = dados_classe['explicit'].mean()
        # Suavização básica de Laplace (evita probabilidade zero)
        if row['explicit'] == 1:
            p_x3 = prob_explicit_1 if prob_explicit_1 > 0 else 1e-9
        else:
            p_x3 = (1 - prob_explicit_1) if (1 - prob_explicit_1) > 0 else 1e-9
            
        log_prob_c += np.log(p_x3)
        
        log_probs[c] = log_prob_c
        
    # Retorna a classe com maior probabilidade logarítmica
    return 1 if log_probs[1] > log_probs[0] else 0

# Aplicando o classificador no conjunto de teste (SEÇÃO 6)
print("Classificando conjunto de teste. Aguarde...")
X_test_copy = X_test.copy()
y_pred = X_test_copy.apply(lambda row: classificar_naive_bayes(row, priors, df_train), axis=1)

# --- SEÇÃO 7: MATRIZ DE CONFUSÃO E MÉTRICAS ---

matriz = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n--- RESULTADOS DA AVALIAÇÃO NO CONJUNTO DE TESTE ---")
print("Matriz de Confusão:")
print(f"[{matriz[0][0]} (VN)]  [{matriz[0][1]} (FP)]")
print(f"[{matriz[1][0]} (FN)]  [{matriz[1][1]} (VP)]")
print("\nMétricas:")
print(f"Acurácia: {acc:.4f}")
print(f"Precisão: {prec:.4f}")
print(f"Recall:   {rec:.4f}")
print(f"F1-Score: {f1:.4f}")

# Visualização bonita da Matriz
plt.figure(figsize=(6,4))
sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Não Pop Predito', 'Pop Predito'], 
            yticklabels=['Não Pop Real', 'Pop Real'])
plt.title("Matriz de Confusão - Naive Bayes")
plt.show()