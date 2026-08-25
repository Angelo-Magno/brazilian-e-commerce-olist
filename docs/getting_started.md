# Getting Started

Guia para configurar e rodar o projeto localmente.

## Pré-requisitos

- Python 3.10+
- Git
- (Opcional) Power BI Desktop, para abrir `reports/powerbi/relatorio_vendas.pbix`

## 1. Clonar o repositório

```bash
git clone <url-do-repositorio>
cd meu-projeto-ds
```

## 2. Criar ambiente virtual

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

## 3. Instalar dependências

```bash
pip install -r requirements.txt
```

## 4. Baixar os dados

Os dados não são versionados no repositório (ver `.gitignore`). Baixe o dataset do Kaggle e coloque os 9 CSVs em `data/raw/`:

🔗 https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

```
data/raw/
├── olist_orders_dataset.csv
├── olist_order_items_dataset.csv
├── olist_customers_dataset.csv
├── olist_products_dataset.csv
├── olist_sellers_dataset.csv
├── olist_order_payments_dataset.csv
├── olist_order_reviews_dataset.csv
├── olist_geolocation_dataset.csv
└── product_category_name_translation.csv
```

## 5. Rodar o pipeline

### Opção A — via notebooks (exploração)

```bash
jupyter notebook notebooks/1.0-eda-analise-exploratoria.ipynb
```

Ordem sugerida:
1. `1.0_eda_and_data_cleaning.ipynb`
2. `2.0_data_preprocessing_and_feature_engineering.ipynb`
3. `3.0_model_training.ipynb`

### Opção B — via scripts (`src/`)

```bash
python -m src.data_processing   # merge e limpeza das 9 tabelas
python -m src.features          # feature engineering
python -m src.train             # treinamento do modelo
python -m src.evaluate          # métricas e figuras
```

## 6. Estrutura de saída esperada

- `data/interim/orders_merged.parquet` — tabelas unidas
- `data/processed/ecommerce_dataset_final.parquet` — dataset final
- `models/xgboost_v1.0.pkl` — modelo treinado
- `reports/figures/` — gráficos de avaliação

## Dúvidas comuns

> Adicionar aqui erros comuns e soluções conforme o projeto evoluir.
