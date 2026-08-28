"""
data_processing.py

Módulo responsável por carregar, unir (merge) e preparar as 9 tabelas
do dataset Olist Brazilian E-Commerce, gerando um dataset consolidado
pronto para EDA e feature engineering.

Fluxo geral:
    raw/ (9 CSVs) -> merge -> interim/ (dataset unificado) -> processed/ (tratado, Parquet)

Nomes de arquivo assumidos em data/raw/ (padrão do Kaggle):
    - olist_customers_dataset.csv
    - olist_orders_dataset.csv
    - olist_order_items_dataset.csv
    - olist_order_payments_dataset.csv
    - olist_order_reviews_dataset.csv
    - olist_products_dataset.csv
    - olist_sellers_dataset.csv
    - olist_geolocation_dataset.csv
    - product_category_name_translation.csv
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Configurações de caminho
# ---------------------------------------------------------------------
RAW_DATA_PATH = Path("../data/raw")
INTERIM_DATA_PATH = Path("../data/interim")
PROCESSED_DATA_PATH = Path("../data/processed")

RAW_FILENAMES = {
    "customers": "olist_customers_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

# Colunas de data que devem ser convertidas para datetime
DATE_COLS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "shipping_limit_date",
    "review_creation_date",
    "review_answer_timestamp"
]



# ---------------------------------------------------------------------
# 1. Carregamento das tabelas brutas
# ---------------------------------------------------------------------
def load_raw_tables(raw_path: Path = RAW_DATA_PATH) -> dict[str, pd.DataFrame]:
    """
    Carrega os 9 CSVs originais do Olist em um dicionário de DataFrames.

    Retorna:
        dict com chaves: 'customers', 'orders', 'order_items', 'payments',
        'reviews', 'products', 'sellers', 'geolocation', 'category_translation'
    """
    tables = {}
    for key, filename in RAW_FILENAMES.items():
        filepath = raw_path / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
        logger.info(f"Carregando {key} de {filepath}")
        tables[key] = pd.read_csv(filepath)
    return tables


# ---------------------------------------------------------------------
# 2. Funções de merge (uma por relação lógica)
# ---------------------------------------------------------------------
def merge_orders_customers(orders: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    """Une orders + customers via customer_id (relação 1:1)."""
    df = orders.merge(customers, on="customer_id", how="left", validate="m:1")
    logger.info(f"orders + customers -> {df.shape}")
    return df


def merge_order_items(df: pd.DataFrame, order_items: pd.DataFrame) -> pd.DataFrame:
    """
    Une o resultado anterior + order_items via order_id.
    Atenção: um order_id pode ter múltiplos order_item_id (pedidos com
    vários produtos), então esse merge expande o número de linhas
    (granularidade passa a ser "item do pedido", não "pedido").
    """
    df = df.merge(order_items, on="order_id", how="inner", validate="1:m")
    logger.info(f"+ order_items -> {df.shape}")
    return df


def merge_products(
    df: pd.DataFrame, products: pd.DataFrame, category_translation: pd.DataFrame
) -> pd.DataFrame:
    """Une + products via product_id, e traduz categoria (PT -> EN)."""
    products_translated = products.merge(
        category_translation, on="product_category_name", how="left"
    )
    df = df.merge(products_translated, on="product_id", how="left", validate="m:1")
    logger.info(f"+ products (traduzido) -> {df.shape}")
    return df


def merge_sellers(df: pd.DataFrame, sellers: pd.DataFrame) -> pd.DataFrame:
    """Une + sellers via seller_id."""
    df = df.merge(sellers, on="seller_id", how="left", validate="m:1")
    logger.info(f"+ sellers -> {df.shape}")
    return df


def merge_payments(df: pd.DataFrame, payments: pd.DataFrame) -> pd.DataFrame:
    """
    Une + payments via order_id.
    Atenção: um pedido pode ter múltiplas formas de pagamento
    (payment_sequential > 1). Para não distorcer a granularidade
    de "item do pedido", agregamos os pagamentos por order_id antes
    do merge (soma do valor pago, contagem de parcelas máxima, etc.).
    """
    payments_agg = (
        payments.groupby("order_id")
        .agg(
            payment_value_total=("payment_value", "sum"),
            payment_installments_max=("payment_installments", "max"),
            payment_types=("payment_type", lambda x: " | ".join(sorted(set(x))) if len(x) else "None"),
            n_payment_methods=("payment_type", "nunique"),
        )
        .reset_index()
    )
    df = df.merge(payments_agg, on="order_id", how="left", validate="m:1")
    logger.info(f"+ payments (agregado) -> {df.shape}")
    return df


def merge_reviews(df: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Une + reviews via order_id.
    Atenção: em raros casos um order_id tem mais de um review_id.
    Mantemos apenas o review mais recente (review_answer_timestamp) por
    pedido para preservar a granularidade "item do pedido" sem duplicar.
    """
    reviews_dedup = (
        reviews.sort_values("review_answer_timestamp")
        .drop_duplicates(subset="order_id", keep="last")
    )
    df = df.merge(reviews_dedup, on="order_id", how="left", validate="m:1")
    logger.info(f"+ reviews (deduplicado) -> {df.shape}")
    return df


def merge_geolocation(df: pd.DataFrame, geolocation: pd.DataFrame) -> pd.DataFrame:
    """
    Opcional: enriquece com dados geográficos via zip_code_prefix.
    A tabela geolocation tem múltiplas lat/long por CEP (imprecisão de
    coordenadas dentro do mesmo prefixo). Aqui agregamos por média para
    ter uma lat/long representativa por prefixo de CEP do cliente.

    Não é chamada por padrão em build_merged_dataset() -- ative
    explicitamente se for usar features geográficas.
    """
    geo_agg = (
        geolocation.groupby("geolocation_zip_code_prefix")
        .agg(
            customer_lat=("geolocation_lat", "mean"),
            customer_lng=("geolocation_lng", "mean"),
        )
        .reset_index()
        .rename(columns={"geolocation_zip_code_prefix": "customer_zip_code_prefix"})
    )
    df = df.merge(geo_agg, on="customer_zip_code_prefix", how="left")
    logger.info(f"+ geolocation (agregado) -> {df.shape}")
    return df


# ---------------------------------------------------------------------
# 3. Pipeline principal de merge
# ---------------------------------------------------------------------
def build_merged_dataset(tables: dict[str, pd.DataFrame], include_geolocation: bool = False) -> pd.DataFrame:
    """
    Orquestra todos os merges na ordem correta e retorna o dataset unificado.
    Este é o dataset que vai para data/interim/.

    Ordem: orders -> customers -> order_items -> products -> sellers
           -> payments -> reviews -> (geolocation, opcional)
    """
    df = merge_orders_customers(tables["orders"], tables["customers"])
    df = merge_order_items(df, tables["order_items"])
    df = merge_products(df, tables["products"], tables["category_translation"])
    df = merge_sellers(df, tables["sellers"])
    df = merge_payments(df, tables["payments"])
    df = merge_reviews(df, tables["reviews"])

    if include_geolocation:
        df = merge_geolocation(df, tables["geolocation"])

    logger.info(f"Dataset final unificado: {df.shape}")
    return df


# ---------------------------------------------------------------------
# 4. Limpeza / tratamentos pós-merge (duplicatas, tipos, nulos)
# ---------------------------------------------------------------------
def clean_merged_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trata duplicatas geradas pelos merges, converte tipos de data,
    remove duplicatas exatas e reporta nulos por coluna.
    """
    df = df.copy()

    # Conversão de datas
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Remove duplicatas exatas (linha inteira repetida)
    before = len(df)
    df = df.drop_duplicates()
    logger.info(f"Duplicatas exatas removidas: {before - len(df)}")

    # Relatório de nulos (apenas log, não remove nada automaticamente --
    # decisão de imputação/remoção fica para a etapa de feature engineering)
    null_report = df.isnull().sum()
    null_report = null_report[null_report > 0].sort_values(ascending=False)
    if not null_report.empty:
        logger.info(f"Colunas com valores nulos:\n{null_report}")

    return df


# ---------------------------------------------------------------------
# 5. Persistência
# ---------------------------------------------------------------------
def save_interim(df: pd.DataFrame, path: Path = INTERIM_DATA_PATH, filename: str = "olist_merged.parquet") -> None:
    """Salva o dataset unificado (pré-tratamento final) em data/interim/."""
    path.mkdir(parents=True, exist_ok=True)
    filepath = path / filename
    df.to_parquet(filepath, index=False)
    logger.info(f"Salvo em {filepath} ({df.shape[0]} linhas, {df.shape[1]} colunas)")


def save_processed(df: pd.DataFrame, path: Path = PROCESSED_DATA_PATH, filename: str = "olist_processed.parquet") -> None:
    """Salva o dataset final tratado em Parquet, pronto para modelagem."""
    path.mkdir(parents=True, exist_ok=True)
    filepath = path / filename
    df.to_parquet(filepath, index=False)
    logger.info(f"Salvo em {filepath} ({df.shape[0]} linhas, {df.shape[1]} colunas)")


# ---------------------------------------------------------------------
# 6. Execução como script
# ---------------------------------------------------------------------
def main():
    tables = load_raw_tables()
    df_merged = build_merged_dataset(tables, include_geolocation=False)
    save_interim(df_merged)

    df_clean = clean_merged_dataset(df_merged)
    save_processed(df_clean)


if __name__ == "__main__":
    main()