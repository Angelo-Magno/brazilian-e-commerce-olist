# Directory Schema — Projeto Olist E-commerce

Este documento descreve a estrutura de pastas e arquivos do projeto `meu-projeto-ds/`.

```
meu-projeto-ds/
│
├── data/
│   ├── raw/                          # CSVs originais do Olist, nunca alterados
│   ├── interim/                      # merges/tratamentos intermediários das 9 tabelas
│   └── processed/                    # dataset final tratado, pronto p/ modelagem
│
├── notebooks/
│   └── 1.0_eda_and_data_cleaning.ipynb
│
├── src/
│   ├── __pycache__/
│   ├── __init__.py
│   ├── data_processing.py            # merge das 9 tabelas + limpeza
│   ├── eda_utils.py                  # funções de apoio à análise exploratória
│   ├── features.py                   # criação e transformação de features
│   ├── models.py                     # definição/treinamento de modelos
│   ├── pipeline.py                   # orquestração do pipeline (fim a fim)
│   └── plots.py                      # geração de gráficos/figuras
│
├── models/                           # artefatos de modelo treinado (.pkl)
│
├── reports/
│   ├── figures/                      # imagens geradas por código (EDA, avaliação)
│   └── powerbi/                      # relatórios/dashboards em Power BI
│
├── docs/
│   ├── assets/
│   │   └── data_schema.png           # diagrama ER das tabelas do Olist
│   ├── business_understanding.md     # problema de negócio e KPIs
│   ├── data_dictionary.md            # dicionário de dados (uma seção por tabela)
│   ├── directory_schema.md           # este arquivo
│   ├── getting_started.md            # guia de instalação e reprodução
│   ├── manual_estrutura_pastas.docx  # manual detalhado da estrutura de pastas
│   ├── model_card.md                 # detalhes técnicos e éticos do modelo
│   └── project_proposal_olist.docx   # proposta de projeto
│
├── olist_venv/                       # ambiente virtual Python
│
├── references/
│   └── fonte_dados.md                # origem, licença e período do dataset
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Princípios de organização

- **`data/raw/`** nunca é alterado manualmente; **`data/interim/`** guarda os merges parciais entre as 9 tabelas relacionais do Olist; **`data/processed/`** contém o dataset final pronto para modelagem.
- **`src/`** concentra código reutilizável e testável, importado pelos notebooks — não deve conter análises exploratórias.
- **`reports/figures/`** recebe apenas imagens **geradas por código** (resultados de análise/modelo).
- **`docs/assets/`** recebe apenas imagens de **referência/documentação** (schema ER, fluxogramas, mockups) — nunca resultados de análise.
- **`docs/model_card.md`** é o único local onde a documentação do modelo deve existir (não duplicar em `models/`).
- **`references/`** = materiais **recebidos** de fontes externas; **`docs/`** = conteúdo **produzido** pelo autor do projeto para explicar o trabalho.
- **`olist_venv/`** e `__pycache__/` não devem ser versionados no Git (ver `.gitignore`).