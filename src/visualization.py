import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

def plot_distribuicao_continua(df, feature_name, title):
    plt.figure(figsize=(8, 5))
    
    mu_0 = df[df['Y'] == 0][feature_name].mean()
    std_0 = df[df['Y'] == 0][feature_name].std()
    
    mu_1 = df[df['Y'] == 1][feature_name].mean()
    std_1 = df[df['Y'] == 1][feature_name].std()
    
    x = np.linspace(0, 1, 1000)
    
    pdf_0 = norm.pdf(x, mu_0, std_0) 
    pdf_1 = norm.pdf(x, mu_1, std_1) 
    
    plt.plot(x, pdf_0, label=f'Não Pop (Y=0) $\mu$={mu_0:.2f}', color='blue', linewidth=2)
    plt.plot(x, pdf_1, label=f'Pop (Y=1) $\mu$={mu_1:.2f}', color='red', linewidth=2)
    
    plt.fill_between(x, pdf_0, alpha=0.1, color='blue')
    plt.fill_between(x, pdf_1, alpha=0.1, color='red')
    
    plt.title(title)
    plt.xlabel(feature_name)
    plt.ylabel('Densidade p(x|Y=c)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plotar_matriz_confusao(matriz):
    plt.figure(figsize=(6,4))
    sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Não Pop Predito', 'Pop Predito'], 
                yticklabels=['Não Pop Real', 'Pop Real'])
    plt.title("Matriz de Confusão - Naive Bayes")
    plt.show()