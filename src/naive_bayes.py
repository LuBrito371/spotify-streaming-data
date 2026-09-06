import numpy as np
from scipy.stats import norm

def probabilidade_a_posteriori(x_val, df, feature_name, priors):
    mu_0 = df[df['Y'] == 0][feature_name].mean()
    std_0 = df[df['Y'] == 0][feature_name].std()
    
    mu_1 = df[df['Y'] == 1][feature_name].mean()
    std_1 = df[df['Y'] == 1][feature_name].std()
    
    like_0 = norm.pdf(x_val, mu_0, std_0)
    like_1 = norm.pdf(x_val, mu_1, std_1)
    
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

def classificar_naive_bayes(row, priors, df_train):
    log_probs = {}
    
    for c in [0, 1]:
        log_prob_c = np.log(priors[c])
        
        dados_classe = df_train[df_train['Y'] == c]
        
        for feature in ['energy', 'danceability']:
            mu = dados_classe[feature].mean()
            std = dados_classe[feature].std()
            verossimilhanca = norm.pdf(row[feature], mu, std)
            log_prob_c += np.log(verossimilhanca + 1e-9) 
            
        prob_explicit_1 = dados_classe['explicit'].mean()
        
        if row['explicit'] == 1:
            p_x3 = prob_explicit_1 if prob_explicit_1 > 0 else 1e-9
        else:
            p_x3 = (1 - prob_explicit_1) if (1 - prob_explicit_1) > 0 else 1e-9
            
        log_prob_c += np.log(p_x3)
        log_probs[c] = log_prob_c
        
    return 1 if log_probs[1] > log_probs[0] else 0