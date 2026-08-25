# Model Card

## Visão geral

| Campo | Descrição |
|---|---|
| Nome do modelo | `xgboost_v1.0.pkl` |
| Algoritmo | XGBoost |
| Tipo de problema | > classificação / regressão — definir |
| Versão | v1.0 |
| Data de treinamento | > preencher |
| Autor | > preencher |

## Objetivo do modelo

> O que o modelo prevê e para qual decisão de negócio ele serve (ver `docs/business_understanding.md`).

## Dados de treinamento

- **Fonte:** Brazilian E-Commerce Public Dataset by Olist (ver `references/fonte_dados.md`)
- **Período:** 2016–2018
- **Tamanho do dataset:** > preencher (linhas, período de treino/teste)
- **Features utilizadas:** > listar principais features (ver `src/features.py`)
- **Variável-alvo:** > preencher

## Divisão dos dados

| Conjunto | % | Observações |
|---|---|---|
| Treino | — | — |
| Validação | — | — |
| Teste | — | — |

## Métricas de performance

| Métrica | Valor |
|---|---|
| — | — |

Figuras de avaliação (matriz de confusão, importância de features, etc.) em `reports/figures/`.

## Principais features (importância)

> Ver `reports/figures/importancia_features.png`

## Limitações e riscos

- Dados restritos ao período 2016–2018, podendo não refletir o comportamento atual do e-commerce.
- Dataset anonimizado — sem informações demográficas detalhadas dos clientes.
- Possível viés geográfico (concentração de pedidos em determinadas regiões do Brasil).
- > adicionar outras limitações identificadas durante a EDA.

## Uso pretendido

- **Uso adequado:** fins educacionais/portfólio, análise exploratória, prova de conceito.
- **Uso não recomendado:** decisões de negócio em produção sem validação adicional e dados mais recentes.

## Manutenção

> Definir se/quando o modelo deve ser retreinado e com qual frequência.
