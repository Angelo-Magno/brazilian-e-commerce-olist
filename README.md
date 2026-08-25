# Análise e Predição de Vendas — Brazilian E-Commerce (Olist)

Projeto de ciência de dados usando o dataset público **Brazilian E-Commerce Public Dataset by Olist**, com dados de ~100 mil pedidos realizados em marketplaces brasileiros entre 2016 e 2018.

## 🎯 Objetivo

> Descreva aqui o objetivo do projeto: ex. prever atraso na entrega, prever satisfação do cliente (review score), segmentar clientes, prever valor de vendas, etc.

## 📊 Sobre os dados

O dataset é **relacional**, composto por 9 tabelas conectadas por chaves como `order_id`, `customer_id`, `product_id` e `seller_id`. Veja o dicionário completo em [`docs/data_dictionary.md`](docs/data_dictionary.md) e a fonte oficial em [`references/fonte_dados.md`](references/fonte_dados.md).

## 🗂️ Estrutura do projeto

```
meu-projeto-ds/
├── data/            # raw, interim e processed
├── notebooks/        # EDA, limpeza/features, treinamento
├── src/               # scripts de processamento, features, treino e avaliação
├── models/            # modelos treinados
├── reports/           # relatórios, figuras e dashboard Power BI
├── docs/              # documentação do projeto
└── references/        # materiais de apoio e fonte dos dados
```

## 🚀 Como rodar

Veja o passo a passo completo em [`docs/getting_started.md`](docs/getting_started.md).

```bash
pip install -r requirements.txt
```

## 🧠 Modelo

Detalhes do modelo treinado (algoritmo, métricas, limitações) estão documentados em [`docs/model_card.md`](docs/model_card.md).

## 📈 Resultados

> Resuma aqui os principais resultados/insights do projeto (métricas, gráficos-chave, conclusões de negócio).

## 🛠️ Tecnologias

- Python (pandas, scikit-learn, XGBoost)
- Jupyter Notebooks
- Power BI

## 📄 Licença

Dataset sob licença CC BY-NC-SA 4.0 (ver [`references/fonte_dados.md`](references/fonte_dados.md)). Código do projeto: defina a licença desejada.
