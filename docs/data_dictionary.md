# Dicionário de Dados

Dataset relacional composto por 9 tabelas (CSVs) localizadas em `data/raw/`. Todas as tabelas podem ser conectadas via chaves como `order_id`, `customer_id`, `product_id` e `seller_id`. Ver diagrama ER em `docs/assets/data_schema.png`.

---

## olist_orders_dataset.csv

| Coluna | Tipo | Descrição |
|---|---|---|
| order_id | string | ID único do pedido |
| customer_id | string | ID do cliente associado ao pedido |
| order_status | string | Status do pedido (delivered, shipped, canceled, ...) |
| order_purchase_timestamp | datetime | Data/hora da compra |
| order_approved_at | datetime | Data/hora da aprovação do pagamento |
| order_delivered_carrier_date | datetime | Data em que o pedido foi entregue à transportadora |
| order_delivered_customer_date | datetime | Data em que o pedido foi entregue ao cliente |
| order_estimated_delivery_date | datetime | Data estimada de entrega |

## olist_order_items_dataset.csv

| Coluna | Tipo | Descrição |
|---|---|---|
| order_id | string | ID do pedido |
| order_item_id | int | Número sequencial do item dentro do pedido |
| product_id | string | ID do produto |
| seller_id | string | ID do vendedor |
| shipping_limit_date | datetime | Prazo limite para envio |
| price | float | Preço do item |
| freight_value | float | Valor do frete |

## olist_customers_dataset.csv

| Coluna | Tipo | Descrição |
|---|---|---|
| customer_id | string | ID do cliente (por pedido) |
| customer_unique_id | string | ID único do cliente (entre pedidos) |
| customer_zip_code_prefix | string | Prefixo do CEP |
| customer_city | string | Cidade |
| customer_state | string | Estado (UF) |

## olist_products_dataset.csv

| Coluna | Tipo | Descrição |
|---|---|---|
| product_id | string | ID do produto |
| product_category_name | string | Categoria do produto (em português) |
| product_weight_g | float | Peso em gramas |
| product_length_cm | float | Comprimento em cm |
| product_height_cm | float | Altura em cm |
| product_width_cm | float | Largura em cm |

## olist_sellers_dataset.csv

| Coluna | Tipo | Descrição |
|---|---|---|
| seller_id | string | ID do vendedor |
| seller_zip_code_prefix | string | Prefixo do CEP |
| seller_city | string | Cidade |
| seller_state | string | Estado (UF) |

## olist_order_payments_dataset.csv

| Coluna | Tipo | Descrição |
|---|---|---|
| order_id | string | ID do pedido |
| payment_sequential | int | Sequência do pagamento (caso haja mais de um) |
| payment_type | string | Tipo de pagamento (credit_card, boleto, ...) |
| payment_installments | int | Número de parcelas |
| payment_value | float | Valor pago |

## olist_order_reviews_dataset.csv

| Coluna | Tipo | Descrição |
|---|---|---|
| review_id | string | ID da avaliação |
| order_id | string | ID do pedido |
| review_score | int | Nota da avaliação (1 a 5) |
| review_comment_title | string | Título do comentário |
| review_comment_message | string | Mensagem do comentário |
| review_creation_date | datetime | Data de criação da avaliação |
| review_answer_timestamp | datetime | Data de resposta da avaliação |

## olist_geolocation_dataset.csv

| Coluna | Tipo | Descrição |
|---|---|---|
| geolocation_zip_code_prefix | string | Prefixo do CEP |
| geolocation_lat | float | Latitude |
| geolocation_lng | float | Longitude |
| geolocation_city | string | Cidade |
| geolocation_state | string | Estado (UF) |

## product_category_name_translation.csv

| Coluna | Tipo | Descrição |
|---|---|---|
| product_category_name | string | Nome da categoria em português |
| product_category_name_english | string | Nome da categoria em inglês |

---

## Dataset processado final

> Preencher após a etapa de limpeza/feature engineering: descrever as colunas finais presentes em `data/processed/ecommerce_dataset_final.parquet`, incluindo eventuais features derivadas (ex. tempo de entrega, atraso, ticket médio, etc.).
