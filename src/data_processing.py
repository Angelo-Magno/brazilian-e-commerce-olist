"""
data_processing.py

Módulo responsável por carregar, unir (merge) e preparar as 9 tabelas
do dataset Olist Brazilian E-Commerce, gerando um dataset consolidado
na granularidade de pedido (uma linha por order_id), pronto para EDA
e feature engineering.

Fluxo geral:
    raw/ (9 CSVs) -> merge -> interim/ (item-level unificado)
                  -> agregação por pedido -> processed/ (order-level, Parquet)

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

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
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

# ---------------------------------------------------------------------
# Renomeação de colunas com grafia errada no CSV original
# ---------------------------------------------------------------------
# O Olist grava "lenght" (com erro de digitação) nos CSVs. Corrigimos o nome
# logo após a leitura, para que todo o restante do código use um único nome.
COLUMN_RENAMES = {
    "product_name_lenght": "product_name_length",
    "product_description_lenght": "product_description_length",
}

# ---------------------------------------------------------------------
# Esquema de tipos por tabela (aplicado direto no read_csv)
# ---------------------------------------------------------------------
# Regras adotadas:
#   - Chaves (ids) e CEPs: "string". CEP como string preserva zeros à esquerda.
#   - Colunas de baixa cardinalidade (status, tipo de pagamento, UF): "category".
#   - Inteiros com possíveis nulos: dtype anulável ("Int16", "Int32").
#   - Datas: listadas em "parse_dates" (nulos viram NaT, esperado em pedidos
#     cancelados ou não entregues).
# Atenção: as chaves de "dtype" devem ser exatamente os nomes do CSV original.
TABLE_SCHEMAS = {
    "customers": {
        "dtype": {
            "customer_id": "string",
            "customer_unique_id": "string",
            "customer_zip_code_prefix": "string",
            "customer_city": "string",
            "customer_state": "category",
        },
        "parse_dates": [],
    },
    "orders": {
        "dtype": {
            "order_id": "string",
            "customer_id": "string",
            "order_status": "category",
        },
        "parse_dates": [
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    },
    "order_items": {
        "dtype": {
            "order_id": "string",
            "order_item_id": "Int16",
            "product_id": "string",
            "seller_id": "string",
            "price": "float64",
            "freight_value": "float64",
        },
        "parse_dates": ["shipping_limit_date"],
    },
    "payments": {
        "dtype": {
            "order_id": "string",
            "payment_sequential": "Int8",
            "payment_type": "category",
            "payment_installments": "Int8",
            "payment_value": "float64",
        },
        "parse_dates": [],
    },
    "reviews": {
        "dtype": {
            "review_id": "string",
            "order_id": "string",
            "review_score": "Int8",
            "review_comment_title": "string",
            "review_comment_message": "string",
        },
        "parse_dates": ["review_creation_date", "review_answer_timestamp"],
    },
    "products": {
        "dtype": {
            "product_id": "string",
            "product_category_name": "string",
            # Grafia original do CSV ("lenght"); renomeada após a leitura
            "product_name_lenght": "Int16",
            "product_description_lenght": "Int32",
            "product_photos_qty": "Int16",
            "product_weight_g": "Int32",
            "product_length_cm": "Int32",
            "product_height_cm": "Int32",
            "product_width_cm": "Int32",
        },
        "parse_dates": [],
    },
    "sellers": {
        "dtype": {
            "seller_id": "string",
            "seller_zip_code_prefix": "string",
            "seller_city": "string",
            "seller_state": "category",
        },
        "parse_dates": [],
    },
    "geolocation": {
        "dtype": {
            "geolocation_zip_code_prefix": "string",
            "geolocation_lat": "float64",
            "geolocation_lng": "float64",
            "geolocation_city": "string",
            "geolocation_state": "category",
        },
        "parse_dates": [],
    },
    "category_translation": {
        "dtype": {
            "product_category_name": "string",
            "product_category_name_english": "string",
        },
        "parse_dates": [],
    },
}

# Limites aproximados do território brasileiro (usados para filtrar coordenadas inválidas)
BRAZIL_LAT_RANGE = (-33.75116944, 5.27438888)
BRAZIL_LNG_RANGE = (-73.98283055, -34.79314722)

# Raio médio da Terra em km (cálculo da distância aproximada por haversine)
EARTH_RADIUS_KM = 6371.0

# Ano a partir do qual shipping_limit_date é considerado inválido
SHIPPING_LIMIT_MAX_VALID_YEAR = 2018

# Traduções ausentes no CSV original de tradução de categorias
MISSING_CATEGORY_TRANSLATIONS = {
    "portateis_cozinha_e_preparadores_de_alimentos": "kitchen_portables_and_food_preparators",
    "pc_gamer": "pc_gamer",
}

# Colunas que são atributos do pedido (idênticas em todos os itens do mesmo
# order_id), por isso usamos "first" ao agregar para uma linha por pedido.
ORDER_LEVEL_COLUMNS = [
    "customer_id",
    "order_status",
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "customer_unique_id",
    "customer_zip_code_prefix",
    "customer_city",
    "customer_state",
    "payment_value_total",
    "payment_installments_max",
    "payment_sequential_max",
    "payment_types",
    "n_payment_methods",
    "review_id",
    "review_score",
    "review_comment_title",
    "review_comment_message",
    "review_creation_date",
    "review_answer_timestamp",
    "customer_lat",
    "customer_lng",
    "customer_geo_n_points"
]


# ---------------------------------------------------------------------
# 1. Carregamento das tabelas brutas
# ---------------------------------------------------------------------
def load_raw_tables(raw_path: Path = RAW_DATA_PATH) -> dict[str, pd.DataFrame]:
    """
    Carrega os 9 CSVs originais do Olist em um dicionário de DataFrames,
    já com os tipos corretos e com as colunas renomeadas (COLUMN_RENAMES).

    Se alguma coluna não puder ser convertida, o erro aparece aqui, na carga,
    e não depois dos merges.

    Retorna:
        dict com chaves: 'customers', 'orders', 'order_items', 'payments',
        'reviews', 'products', 'sellers', 'geolocation', 'category_translation'
    """
    tables = {}
    for key, filename in RAW_FILENAMES.items():
        filepath = raw_path / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
        schema = TABLE_SCHEMAS[key]
        logger.info(f"Carregando {key} de {filepath}")
        df = pd.read_csv(
            filepath,
            dtype=schema["dtype"],
            parse_dates=schema["parse_dates"],
        )
        # Padroniza nomes com erro de digitação do dataset original
        tables[key] = df.rename(columns=COLUMN_RENAMES)
    return tables


# ---------------------------------------------------------------------
# 2. Funções de merge (uma por relação lógica)
# ---------------------------------------------------------------------
def merge_orders_customers(
    orders: pd.DataFrame, customers: pd.DataFrame
) -> pd.DataFrame:
    """Une orders + customers via customer_id (relação m:1)."""
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
            payment_types=(
                "payment_type",
                lambda x: " | ".join(sorted(set(x.dropna()))) or pd.NA,
            ),
            n_payment_methods=("payment_type", "nunique"),
            payment_sequential_max=("payment_sequential", "max"),
        )
        .reset_index()
        .astype(
            {
                "payment_value_total": "float64",
                "payment_installments_max": "Int8",
                "payment_types": "string",
                "n_payment_methods": "Int8",
                "payment_sequential_max": "Int8",
            }
        )
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
    reviews_dedup = reviews.sort_values("review_answer_timestamp").drop_duplicates(
        subset="order_id", keep="last"
    )
    df = df.merge(reviews_dedup, on="order_id", how="left", validate="m:1")
    logger.info(f"+ reviews (deduplicado) -> {df.shape}")
    return df


def aggregate_geolocation(geolocation: pd.DataFrame) -> pd.DataFrame:
    """
    Reduz a tabela de geolocalização a uma linha por prefixo de CEP,
    usando a mediana das coordenadas como ponto representativo.
    """
    # Remove coordenadas fora do Brasil antes de agregar
    in_brazil = geolocation["geolocation_lat"].between(*BRAZIL_LAT_RANGE) & geolocation[
        "geolocation_lng"
    ].between(*BRAZIL_LNG_RANGE)
    n_invalid = (~in_brazil).sum()
    logger.info(f"Coordenadas fora do Brasil removidas: {n_invalid}")

    geo_agg = (
        geolocation[in_brazil]
        .groupby("geolocation_zip_code_prefix")
        .agg(
            lat=("geolocation_lat", "median"),
            lng=("geolocation_lng", "median"),
            n_points=("geolocation_lat", "size"),
        )
        .reset_index()
        .astype(
            {
                "lat": "float64",
                "lng": "float64",
                "n_points": "Int32",
            }
        )
        .rename(columns={"geolocation_zip_code_prefix": "zip_code_prefix"})
    )
    logger.info(f"geolocation agregada -> {geo_agg.shape}")
    return geo_agg


def merge_geolocation(df: pd.DataFrame, geolocation: pd.DataFrame) -> pd.DataFrame:
    """
    Une as coordenadas de cliente e vendedor via prefixo de CEP.
    Deve ser chamada depois de merge_sellers, pois depende de seller_zip_code_prefix.
    Prefixos sem correspondência ficam com lat/lng nulos (how="left").
    """
    geo_agg = aggregate_geolocation(geolocation)

    for role in ("customer", "seller"):
        # Renomeia as colunas para evitar colisão entre os dois merges
        geo_role = geo_agg.rename(
            columns={
                "zip_code_prefix": f"{role}_zip_code_prefix",
                "lat": f"{role}_lat",
                "lng": f"{role}_lng",
                "n_points": f"{role}_geo_n_points",
            }
        )
        df = df.merge(
            geo_role,
            on=f"{role}_zip_code_prefix",
            how="left",
            validate="m:1",
        )

        # Reporta prefixos sem cobertura para decidir o tratamento depois
        n_missing = df[f"{role}_lat"].isna().sum()
        logger.info(
            f"+ geolocation ({role}) -> {df.shape} | sem coordenadas: {n_missing}"
        )

    return df


def haversine_km(
    lat1: pd.Series, lng1: pd.Series, lat2: pd.Series, lng2: pd.Series
) -> pd.Series:
    """
    Calcula a distância aproximada em km entre dois pontos (fórmula de haversine).
    Funciona com Series do pandas e propaga NaN quando alguma coordenada falta.
    """
    lat1, lng1, lat2, lng2 = (
        np.radians(lat1.astype("float64")),
        np.radians(lng1.astype("float64")),
        np.radians(lat2.astype("float64")),
        np.radians(lng2.astype("float64")),
    )
    d_lat = lat2 - lat1
    d_lng = lng2 - lng1
    a = np.sin(d_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(d_lng / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def add_seller_customer_distance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona seller_customer_distance_km, a distância aproximada entre o
    vendedor e o cliente. Deve ser chamada depois de merge_geolocation.
    """
    df = df.assign(
        seller_customer_distance_km=haversine_km(
            df["seller_lat"], df["seller_lng"], df["customer_lat"], df["customer_lng"]
        )
    )
    
    # Flag para pedidos no mesmo prefixo: nesses casos a distância é 0 por construção
    if {"customer_zip_code_prefix", "seller_zip_code_prefix"}.issubset(df.columns):
        df["same_zip_prefix"] = (
            df["customer_zip_code_prefix"] == df["seller_zip_code_prefix"]
        ).astype("Int8")

    logger.info("Distância vendedor-cliente calculada.")
    return df


# ---------------------------------------------------------------------
# 3. Pipeline principal de merge
# ---------------------------------------------------------------------
def build_merged_dataset(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Orquestra todos os merges na ordem correta e retorna o dataset unificado
    na granularidade de item. Este é o dataset que vai para data/interim/.

    Ordem: orders -> customers -> order_items -> products -> sellers
           -> payments -> reviews -> geolocation -> distância
    """
    df = merge_orders_customers(tables["orders"], tables["customers"])
    df = merge_order_items(df, tables["order_items"])
    df = merge_products(df, tables["products"], tables["category_translation"])
    df = merge_sellers(df, tables["sellers"])
    df = merge_payments(df, tables["payments"])
    df = merge_reviews(df, tables["reviews"])
    df = merge_geolocation(df, tables["geolocation"])
    df = add_seller_customer_distance(df)

    logger.info(f"Dataset item-level unificado: {df.shape}")
    return df


def fix_missing_category_translations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preenche product_category_name_english para categorias que não existem
    no CSV de tradução original. Deve ser aplicada após merge_products.
    Só preenche onde a tradução está nula, sem sobrescrever traduções existentes.
    """
    df = df.copy()
 
    manual_translation = df["product_category_name"].map(MISSING_CATEGORY_TRANSLATIONS)
    mask = df["product_category_name_english"].isna() & manual_translation.notna()
    df.loc[mask, "product_category_name_english"] = manual_translation[mask]
 
    logger.info(f"Traduções de categoria corrigidas após o merge: {mask.sum()}")
    return df


def fix_invalid_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte valores impossíveis em nulos.
    Deve ser aplicada no dataset item-level, antes da agregação por pedido,
    para que somas e máximos não incorporem os valores inválidos.
    """
    df = df.copy()
 
    # Datas limite de envio fora do período do dataset
    invalid_date = df["shipping_limit_date"].dt.year > SHIPPING_LIMIT_MAX_VALID_YEAR
    df.loc[invalid_date, "shipping_limit_date"] = pd.NaT
 
    # Peso zero ou negativo é impossível
    invalid_weight = df["product_weight_g"] <= 0
    df.loc[invalid_weight, "product_weight_g"] = pd.NA
 
    # Número de parcelas zero ou negativo é inválido
    invalid_installments = df["payment_installments_max"] <= 0
    df.loc[invalid_installments, "payment_installments_max"] = pd.NA
 
    logger.info(
        "Valores inválidos tratados | "
        f"shipping_limit_date: {invalid_date.sum()} | "
        f"product_weight_g: {invalid_weight.sum()} | "
        f"payment_installments_max: {invalid_installments.sum()}"
    )
    return df


# ---------------------------------------------------------------------
# 4. Agregação para granularidade de pedido e limpeza
# ---------------------------------------------------------------------
def collapse_item_level_to_order_level(
    df: pd.DataFrame, order_level_cols: list[str] = ORDER_LEVEL_COLUMNS
) -> pd.DataFrame:
    """Agrega as colunas de item para uma linha por order_id."""

    # Falha explícita se alguma coluna de pedido não existir (evita erro silencioso)
    missing = [col for col in order_level_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Colunas de pedido ausentes no DataFrame: {missing}")

    # Colunas de pedido: pegar o primeiro valor (são idênticas dentro do pedido)
    order_aggs = {col: (col, "first") for col in order_level_cols}

    def mode_or_nan(s: pd.Series):
        mode = s.mode()
        return mode.iat[0] if not mode.empty else pd.NA

    # Soma com min_count=1 para manter nulo quando todos os valores forem nulos
    sum_or_nan = lambda s: s.sum(min_count=1)
    string_or_nan = lambda s: " | ".join(sorted(set(s.dropna()))) or pd.NA

    # Volume calculado por item ANTES da agregação (atributo físico do item)
    df = df.assign(
        product_volume_cm3=(
            df["product_length_cm"] * df["product_height_cm"] * df["product_width_cm"]
        )
    )

    item_aggs = {
        "n_items": ("order_item_id", "count"),
        "n_distinct_products": ("product_id", "nunique"),
        "n_sellers": ("seller_id", "nunique"),
        "n_seller_states": ("seller_state", "nunique"),
        "seller_states": ("seller_state", string_or_nan),
        "seller_cities": ("seller_city", string_or_nan),
        "max_shipping_limit_date": ("shipping_limit_date", "max"),
        "min_shipping_limit_date": ("shipping_limit_date", "min"),
        "total_price": ("price", sum_or_nan),
        "max_item_price": ("price", "max"),
        "mean_item_price": ("price", "mean"),
        "total_freight": ("freight_value", sum_or_nan),
        "main_category": ("product_category_name_english", mode_or_nan),
        "n_categories": ("product_category_name_english", "nunique"),
        "categories": ("product_category_name_english", string_or_nan),
        "mean_name_length": ("product_name_length", "mean"),
        "mean_description_length": ("product_description_length", "mean"),
        "min_photos_qty": ("product_photos_qty", "min"),
        "total_weight_g": ("product_weight_g", sum_or_nan),
        "max_weight_g": ("product_weight_g", "max"),
        "total_volume_cm3": ("product_volume_cm3", sum_or_nan),
        "max_item_volume_cm3": ("product_volume_cm3", "max"),
        "max_seller_distance_km": ("seller_customer_distance_km", "max"),
        "mean_seller_distance_km": ("seller_customer_distance_km", "mean"),
        # Quantos vendedores do pedido ficaram sem coordenadas
        "n_sellers_missing_distance": (
            "seller_customer_distance_km",
            lambda s: s.isna().sum(),
        ),
        "min_seller_geo_n_points": ("seller_geo_n_points", "min"),
        "max_seller_geo_n_points": ("seller_geo_n_points", "max"),
    }

    df_order = (
        df.groupby("order_id")
        .agg(**order_aggs, **item_aggs)
        .astype(
            {
                "n_categories": "Int16",
                "n_items": "Int16",
                "n_distinct_products": "Int16",
                "n_sellers": "Int16",
                "n_seller_states": "Int16",
                "seller_states": "string",
                "n_sellers_missing_distance": "Int16",
                "min_seller_geo_n_points": "Int32",
                "max_seller_geo_n_points": "Int32",
            }
        )
        .reset_index()
    )
    logger.info(f"Agregado para granularidade de pedido -> {df_order.shape}")

    return df_order


def clean_order_level_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicatas exatas e reporta nulos por coluna.
    Não remove nulos automaticamente: a decisão de imputação fica para
    a etapa de feature engineering.
    """
    df = df.copy()

    before = len(df)
    df = df.drop_duplicates()
    logger.info(f"Duplicatas exatas removidas: {before - len(df)}")

    null_report = df.isnull().sum()
    null_report = null_report[null_report > 0].sort_values(ascending=False)
    if not null_report.empty:
        logger.info(f"Colunas com valores nulos:\n{null_report}")

    return df


# ---------------------------------------------------------------------
# 5. Persistência
# ---------------------------------------------------------------------
def save_interim(
    df: pd.DataFrame,
    path: Path = INTERIM_DATA_PATH,
    filename: str = "olist_merged.parquet",
) -> None:
    """Salva o dataset unificado item-level em data/interim/."""
    path.mkdir(parents=True, exist_ok=True)
    filepath = path / filename
    df.to_parquet(filepath, index=False)
    logger.info(f"Salvo em {filepath} ({df.shape[0]} linhas, {df.shape[1]} colunas)")


def save_processed(
    df: pd.DataFrame,
    path: Path = PROCESSED_DATA_PATH,
    filename: str = "olist_processed.parquet",
) -> None:
    """Salva o dataset final (order-level) em Parquet, pronto para modelagem."""
    path.mkdir(parents=True, exist_ok=True)
    filepath = path / filename
    df.to_parquet(filepath, index=False)
    logger.info(f"Salvo em {filepath} ({df.shape[0]} linhas, {df.shape[1]} colunas)")


# ---------------------------------------------------------------------
# 6. Execução como script
# ---------------------------------------------------------------------
def main():
    tables = load_raw_tables()

    df_items = build_merged_dataset(tables)
    df_items = fix_invalid_values(df_items)
    df_items = fix_missing_category_translations(df_items)
    save_interim(df_items)

    df_orders = collapse_item_level_to_order_level(df_items)
    df_clean = clean_order_level_dataset(df_orders)
    save_processed(df_clean)


if __name__ == "__main__":
    main()
