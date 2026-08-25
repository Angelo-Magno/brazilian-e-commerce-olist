# Business Understanding — Sistema de Alerta Preventivo Operacional

## 1. Contexto do Negócio

Este projeto simula uma demanda de um marketplace de e-commerce (baseado no dataset público **Brazilian E-Commerce Public Dataset by Olist**), no qual as áreas de **Operações/Logística** e **Customer Experience (CX)** enfrentam o mesmo problema estrutural: **agem de forma reativa**, só percebendo falhas depois que elas já impactaram o cliente.

- A operação logística só descobre um atraso quando o pedido já está atrasado.
- O time de CX só descobre que um cliente ficou insatisfeito quando a nota de avaliação (review) já foi registrada — momento em que não há mais nada a fazer para reverter aquela experiência específica.

## 2. Problema de Negócio

> **"Como identificar, no momento da compra, quais pedidos têm maior risco de atraso na entrega e/ou de gerar uma avaliação negativa do cliente — para que as áreas de Operações e CX possam agir preventivamente?"**

Isso motiva a construção de **dois modelos preditivos complementares**, compartilhando a mesma base de dados e pipeline de features:

| Modelo | Pergunta que responde | Área que consome |
|---|---|---|
| **Modelo 1 — Risco de Atraso** | Este pedido tem alta probabilidade de ser entregue depois do prazo estimado? | Operações/Logística |
| **Modelo 2 — Risco de Review Ruim** | Este pedido tem alta probabilidade de receber nota de avaliação ≤ 2? | Customer Experience |

> Nota de arquitetura: os dois modelos compartilham o mesmo pipeline de features de origem (Seção 6), mas são treinados e avaliados de forma **independente** — ver justificativa abaixo e na Seção 8.

Os dois modelos são **independentes** — o Modelo 2 não recebe a previsão do Modelo 1 como informação de entrada. Isso foi uma escolha deliberada: usar a previsão de um modelo como "dado de entrada" de outro é arriscado, porque o segundo modelo passa a confiar em um número que já carrega os próprios erros e vieses do primeiro — é como tomar uma decisão com base numa estimativa alheia sem saber o quanto ela pode estar errada. Esse erro se propaga e pode até se amplificar de um modelo para o outro.

A hipótese de negócio de que **atraso é um dos principais motivadores de review ruim** é investigada e quantificada diretamente na fase de **Análise Exploratória (EDA)**: compara-se a taxa de review ruim entre pedidos que realmente atrasaram e pedidos entregues no prazo, usando o histórico observado. Essa abordagem conta a mesma história de negócio, com um pipeline de modelagem mais simples e sem o risco de propagar erro de um modelo para o outro.

## 3. Hipóteses de Negócio a Validar na EDA

1. Pedidos com prazo de entrega estimado mais apertado têm maior taxa de atraso.
2. A distância geográfica entre vendedor e cliente está associada a maior risco de atraso.
3. Existe sazonalidade relevante (Black Friday, Natal, e o evento atípico da greve dos caminhoneiros em maio de 2018) que aumenta o risco de atraso.
4. Vendedores com histórico de atraso tendem a repetir o padrão em pedidos futuros.
5. Atraso na entrega está correlacionado com review score baixo.
6. Categorias de produto específicas (ex: frete de itens grandes/frágeis) têm taxa de insatisfação inerentemente maior, independente de atraso.

## 4. Objetivos do Projeto

### Objetivo Geral
Desenvolver dois modelos de classificação binária capazes de prever, **no momento do pagamento aprovado do pedido**, o risco de atraso na entrega e o risco de avaliação negativa, entregando os resultados em um painel de negócio (Power BI) acionável pelas áreas de Operações e CX.

### Objetivos Específicos
- Construir um pipeline único e reutilizável de merge e engenharia de features a partir das 9 tabelas relacionais do Olist.
- Garantir que nenhuma feature utilizada nos modelos contenha informação não disponível no momento da decisão (ver Seção 6 — Restrição de Corte Temporal).
- Treinar, avaliar e comparar modelos de classificação priorizando recall da classe de risco (minoritária).
- Traduzir os resultados dos modelos em um painel executivo segmentável por vendedor, categoria e região.

## 5. Critérios de Sucesso

| Critério | Tipo | Descrição |
|---|---|---|
| Recall da classe de risco ≥ referência definida na fase de modelagem | Técnico | Prioridade sobre acurácia geral, dado o desbalanceamento das classes |
| PR-AUC reportado além de ROC-AUC | Técnico | Métrica mais robusta em cenário de classes desbalanceadas |
| Zero vazamento de dados temporal identificado na auditoria do pipeline | Técnico | Validação explícita do corte temporal de features |
| Painel Power BI permite priorização acionável por vendedor/região/categoria | Negócio | O modelo precisa gerar insight operacional, não só uma métrica isolada |
| Documentação clara das decisões e trade-offs no relatório executivo | Comunicação | Fundamental para o público não técnico (portfólio) |

## 6. Restrição Crítica: Corte Temporal de Features

**Regra vigente para todo o projeto:** só podem ser utilizadas como feature informações existentes **até o momento de `order_approved_at`** (pagamento aprovado). Qualquer coluna cujo timestamp seja posterior a esse marco (ex: `order_delivered_customer_date`, `review_creation_date`, `review_comment_message`) representa vazamento de dados (*data leakage*) e não pode ser utilizada como variável preditora.

Essa restrição está documentada em detalhe em `docs/data_dictionary.md`, tabela a tabela.

## 7. Escopo do Projeto

### Dentro do escopo
- Modelagem dos dois targets (atraso e review ruim) com features pré-entrega.
- Engenharia de features históricas por vendedor (ex: taxa de atraso acumulada), calculadas de forma temporalmente consistente (sem usar dados futuros ao ponto de corte de cada pedido).
- Dashboard Power BI com os dois scores de risco.
- Relatório executivo em linguagem de negócio.

### Fora do escopo
- Recomendação de ações corretivas automatizadas (ex: troca de transportadora) — o projeto entrega o **sinal de risco**, não a automação da resposta.
- Modelos de precificação de frete ou otimização logística de rotas.
- Deploy em produção / serving em tempo real (o projeto é de portfólio, com foco analítico e de modelagem batch).

## 8. Riscos e Restrições do Projeto

- **Dataset histórico (2016–2018):** contém um evento atípico (greve dos caminhoneiros, maio/2018) que pode distorcer padrões de atraso se não for tratado explicitamente na EDA.
- **Geolocalização opcional:** a tabela `geolocation` do Olist tem granularidade imprecisa (CEP truncado); distância vendedor-cliente será uma aproximação, não um valor exato.
- **Desbalanceamento de classes:** tanto atraso quanto review ruim tendem a ser eventos minoritários no dataset, exigindo tratamento específico na modelagem e escolha de métricas.
- **Decisão de arquitetura (modelos independentes):** avaliou-se usar a previsão do Modelo 1 como entrada do Modelo 2 (stacking), mas essa abordagem foi descartada por risco de propagação de erro e viés entre modelos. A relação entre atraso e review ruim é validada via EDA, não via arquitetura de modelagem.

## 9. Stakeholders (Simulados)

| Papel | Interesse |
|---|---|
| Gerente de Operações/Logística | Reduzir atrasos e custos de reentrega/reembolso |
| Gerente de CX | Reduzir volume de reclamações e melhorar NPS |
| Liderança de Dados | Validar viabilidade técnica e qualidade metodológica do pipeline |