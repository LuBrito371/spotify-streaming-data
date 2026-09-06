# Estudo Dirigido: Classificadores Bayesianos 

**Instituição:** Universidade Federal do Agreste de Pernambuco (UFAPE) - Bacharelado em Ciência da Computação  
**Disciplina:** Inteligência Artificial 2026.1  
**Professor:** Prof. Dr. Luis Filipe
**Autores:** Luana Vitória da Silva Brito e Yasmin da Silva Muniz
**Vídeo de Apresentação:** (https://drive.google.com/file/d/1gVc9l2qYRI-ufHsFOPphIPHtXmKGgcAN/view?usp=sharing)

---

## Sobre o Projeto
Este projeto investiga experimentalmente o funcionamento de classificadores Bayesianos a partir da modelagem probabilística de características de áudio de uma base de dados real. 

O objetivo do modelo é classificar se uma música pertence ao gênero **"Pop"** (Y=1) ou **"Não Pop"** (Y=0) com base em três atributos extraídos do Spotify:
1. `energy` (Contínua)
2. `danceability` (Contínua)
3. `explicit` (Categórica Binária)

O estudo implementa classificadores Bayesianos univariados e um classificador multivariado **Naive Bayes**, construído no domínio logarítmico para evitar *underflow* numérico, seguido pela avaliação de desempenho através de uma Matriz de Confusão.

---

## Base de Dados
* **Nome:** Spotify Streaming Data and Audio Feature (2020-25)
* **Origem:** Kaggle[cite: 2]
* **Link para download:** (https://www.kaggle.com/datasets/beamhonor0911/spotify-artist-streaming-analytics-20202025)

> **Nota:** O arquivo CSV original não está versionado neste repositório devido ao seu tamanho. Faça o download no link acima e coloque o arquivo `spotify_artist_streaming_2020_2025.csv` na raiz do projeto antes de executar as análises.

---

## Estrutura do Repositório
O código foi modularizado para garantir a reprodutibilidade e separar as responsabilidades do pipeline de dados:
* `main.py`: Orquestrador principal do projeto. Executa o fluxo de carregamento, treinamento, classificação e avaliação.
* `src/data_processing.py`: Funções para limpeza dos dados, definição da variável alvo e separação (80% treino / 20% teste).
* `src/naive_bayes.py`: Core matemático contendo a extração de parâmetros (Normal e Bernoulli), cálculo da probabilidade a posteriori (Teorema de Bayes) e o classificador Naive Bayes multivariado.
* `src/visualization.py`: Rotinas de plotagem com `matplotlib` e `seaborn` para as densidades de probabilidade e a Matriz de Confusão.
* `requirements.txt`: Lista de dependências e bibliotecas necessárias para a execução do projeto.

---

## Comandos de Instalação e Execução

Abra o terminal na pasta onde você deseja salvar o projeto e siga o passo a passo abaixo:

```bash
git clone https://github.com/LuBrito371/spotify-streaming-data.git
cd spotify-streaming-data
pip install -r requirements.txt
python main.py
