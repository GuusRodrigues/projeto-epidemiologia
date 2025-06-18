# 🧪 Projeto Epidemiologia - Previsão de Casos de Mpox no Brasil

Este projeto realiza coleta, processamento e análise de dados de Mpox no Brasil, com previsão de novos casos por estado, usando técnicas de Machine Learning. Também disponibiliza um **dashboard interativo** para visualização de tendências, ranking e mapa de calor por estado.

---

## 📁 Estrutura do Projeto

projeto_pyspark/
├── data/
│ ├── raw/ # Dados brutos (API)
│ ├── processed/ # Dados tratados
│ └── models/ # Modelos treinados
├── src/
│ ├── data_collection.py # Coleta via API Mpox
│ ├── data_processing.py # Limpeza e tratamento com Pandas
│ ├── ml_pipeline.py # Predição com Random Forest
│ └── dashboard.py # Dashboard interativo com Dash
├── notebooks/
│ ├── eda.ipynb # Análise exploratória
│ └── model_evaluation.ipynb # Avaliação do modelo
├── reports/
│ └── analise_preditiva.pdf
└── requirements.txt # Dependências do projeto

---

## 🚀 Como rodar localmente

### 1. Pré-requisitos

- Python 3.8+
- Pip

### 2. Instale as dependências

pip install -r requirements.txt

3. Execute os scripts principais
bash
Copiar
Editar
# Coletar dados
python src/data_collection.py

# Processar dados
python src/data_processing.py

# Treinar e salvar modelo
python src/ml_pipeline.py

# Executar dashboard
python src/dashboard.py
Acesse o dashboard em: http://localhost:8050

📊 Funcionalidades do Dashboard
Série histórica por estado com previsão de 3 meses

Ranking anual de estados com mais casos

Mapa de calor dos casos com filtro por estado e ano

🤖 Modelo de Machine Learning
Algoritmo: RandomForestRegressor

Entrada: Lags de casos por estado (últimos 3 meses)

Saída: Previsão de casos para o mês seguinte

Treinamento feito com dados públicos via API Dados Abertos do Ministério da Saúde

🧰 Tecnologias Utilizadas
Python | Pandas | Scikit-learn

Dash e Plotly

Joblib para salvar modelos

Requests para coleta de dados

🤝 Contribuições
Sinta-se à vontade para abrir issues ou enviar pull requests. Toda contribuição é bem-vinda!

📄 Licença
Distribuído sob a licença MIT. Veja LICENSE para mais informações.


Se quiser, posso gerar o `requirements.txt` automaticamente para você com base no que foi usado no projeto. É só pedir: [gerar requirements.txt](f).

