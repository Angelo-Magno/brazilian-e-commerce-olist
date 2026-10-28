import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


EARTH_RADIUS_KM = 6371.0088  # raio médio da Terra


def dataframe_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gera um DataFrame de resumo/diagnóstico para cada feature de um DataFrame.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada a ser analisado.

    Retorna
    -------
    pd.DataFrame
        DataFrame com uma linha por feature, contendo:
        feature, dtype, total_null, percent_null, unique, top, freq,
        mean, std, 25%, 50%, 75%, min, max, skew, kurtosis.
        Para colunas datetime, apenas min/max são preenchidos (demais
        estatísticas numéricas ficam como NaN).
    """
    n_rows = len(df)
    results = []

    for column in df.columns:
        series = df[column]
        dtype = series.dtype
        is_numeric = pd.api.types.is_numeric_dtype(series)
        is_datetime = pd.api.types.is_datetime64_any_dtype(series)

        total_null = series.isnull().sum()
        percent_null = (total_null / n_rows) * 100 if n_rows > 0 else np.nan

        unique = series.nunique(dropna=True)
        count = series.count()

        # top e freq (valor mais frequente)
        value_counts = series.value_counts(dropna=True, ascending=False)
        if not value_counts.empty:
            top = value_counts.index[0]
            freq = value_counts.iloc[0]
        else:
            top = np.nan
            freq = np.nan

        # estatísticas numéricas (inclui skew/kurtosis)
        if is_numeric:
            mean = series.mean()
            std = series.std()
            p25 = series.quantile(0.25)
            p50 = series.quantile(0.50)
            p75 = series.quantile(0.75)
            minimum = series.min()
            maximum = series.max()
            skew = series.skew()
            kurtosis = series.kurt()
        # estatísticas específicas para colunas datetime (min/max de data)
        elif is_datetime:
            mean = std = p25 = p50 = p75 = skew = kurtosis = np.nan
            minimum = series.min()
            maximum = series.max()
        else:
            mean = std = p25 = p50 = p75 = minimum = maximum = skew = kurtosis = np.nan

        results.append({
            "feature": column,
            "dtype": dtype,
            "total_null": total_null,
            "percent_null": round(percent_null, 6) if pd.notnull(percent_null) else np.nan,
            "unique": unique,
            "count": count,
            "top": top,
            "freq": freq,
            "mean": mean,
            "std": std,
            "25%": p25,
            "50%": p50,
            "75%": p75,
            "min": minimum,
            "max": maximum,
            "skew": skew,
            "kurtosis": kurtosis
        })

    summary_df = pd.DataFrame(results).sort_values(by=["feature"]).set_index("feature")
    return summary_df


def show_multi_item_orders(
    df: pd.DataFrame,
    order_id_col: str = "order_id",
    required_cols: list[str] = ["order_id", "price", "freight_value", "payment_value_total"]
) -> pd.DataFrame:
    """
    Exibe pedidos com mais de um item, agrupando linhas do mesmo order_id
    lado a lado para facilitar a inspeção visual.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame no grão de item do pedido, contendo as colunas
        'order_id', 'price', 'freight_value', 'payment_value'
        e 'payment_value_total'.
    order_id_col : str, default 'order_id'
        Nome da coluna identificadora do pedido.

    Returns
    -------
    pd.DataFrame
        Subconjunto do df original, filtrado para pedidos com mais
        de um item e ordenado por order_id (linhas do mesmo pedido
        ficam contíguas).
    """

    # valida se todas as colunas necessárias existem no df
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colunas ausentes no DataFrame: {missing_cols}")

    # conta quantos itens cada pedido possui
    item_counts = df.groupby(order_id_col).size()

    # filtra apenas os order_ids com mais de um item
    multi_item_order_ids = item_counts[item_counts > 1].index

    # filtra o df original mantendo apenas esses pedidos
    result = df[df[order_id_col].isin(multi_item_order_ids)]

    # ordena por order_id para manter itens do mesmo pedido próximos
    result = result.sort_values(by=order_id_col).reset_index(drop=True)

    return result[required_cols]


def check_payment_consistency(
    df: pd.DataFrame,
    order_id_col: str = "order_id",
    tolerance: float = 0.02,
) -> pd.DataFrame:
    """
    Verifica se a soma de price + freight_value por pedido bate com
    payment_value_total, considerando uma tolerância para arredondamentos.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame no grão de item do pedido, contendo as colunas
        'order_id', 'price', 'freight_value' e 'payment_value_total'.
    order_id_col : str, default 'order_id'
        Nome da coluna identificadora do pedido.
    tolerance : float, default 0.01
        Diferença máxima aceitável (em valor absoluto) entre os dois
        valores para considerar como "batendo".

    Returns
    -------
    pd.DataFrame
        DataFrame agregado por order_id com as colunas:
        'price_freight_sum', 'payment_value_total', 'difference'
        e 'is_consistent' (bool).
    """
    required_cols = [order_id_col, "price", "freight_value", "payment_value_total", "payment_types"]

    # valida se todas as colunas necessárias existem no df
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colunas ausentes no DataFrame: {missing_cols}")

    # agrega por order_id: soma de price e freight_value, "first" para payment_value_total
    grouped = df.groupby(order_id_col).agg(
        price_sum=("price", "sum"),
        freight_sum=("freight_value", "sum"),
        payment_value_total=("payment_value_total", "first"),
        payment_types=("payment_types", "first")
    )

    # calcula a soma price + freight_value
    grouped["price_freight_sum"] = grouped["price_sum"] + grouped["freight_sum"]

    # calcula a diferença absoluta entre os dois valores
    grouped["difference"] = np.abs(
        grouped["price_freight_sum"] - grouped["payment_value_total"]
    )

    # flag de consistência com base na tolerância definida
    grouped["is_consistent"] = grouped["difference"] <= tolerance

    result = grouped[
        ["payment_types", "price_freight_sum", "payment_value_total", "difference", "is_consistent"]
    ].reset_index()

    return result


def add_seller_customer_distance(
    df: pd.DataFrame,
    customer_lat_col: str = "customer_lat",
    customer_lng_col: str = "customer_lng",
    seller_lat_col: str = "seller_lat",
    seller_lng_col: str = "seller_lng",
) -> pd.DataFrame:
    """
    Adiciona a distância (km) entre vendedor e cliente usando a fórmula de haversine.
    Linhas sem coordenadas em algum dos lados ficam com distância nula (NaN).
    A distância é uma aproximação: usa o ponto central de cada prefixo de CEP,
    não o endereço real.
    """
    required = [customer_lat_col, customer_lng_col, seller_lat_col, seller_lng_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Colunas de coordenadas ausentes no DataFrame: {missing}")

    # Converte para float64 do NumPy (colunas anuláveis do pandas viram NaN)
    lat1 = np.radians(df[customer_lat_col].to_numpy(dtype="float64", na_value=np.nan))
    lng1 = np.radians(df[customer_lng_col].to_numpy(dtype="float64", na_value=np.nan))
    lat2 = np.radians(df[seller_lat_col].to_numpy(dtype="float64", na_value=np.nan))
    lng2 = np.radians(df[seller_lng_col].to_numpy(dtype="float64", na_value=np.nan))

    # Fórmula de haversine
    delta_lat = lat2 - lat1
    delta_lng = lng2 - lng1
    a = np.sin(delta_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lng / 2) ** 2
    # clip evita erro numérico quando 'a' passa levemente de 1 por arredondamento
    distance_km = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

    df = df.copy()
    df["seller_customer_distance_km"] = distance_km

    # Flag para pedidos no mesmo prefixo: nesses casos a distância é 0 por construção
    if {"customer_zip_code_prefix", "seller_zip_code_prefix"}.issubset(df.columns):
        df["same_zip_prefix"] = (
            df["customer_zip_code_prefix"] == df["seller_zip_code_prefix"]
        ).astype("Int8")

    return df


def save_table_as_image(
    df,
    filepath,
    title=None,
    header_color="#2C3E50",
    header_text_color="white",
    row_colors=("#FFFFFF", "#F2F5F8"),
    edge_color="#D5DBE1",
    font_size=11,
    row_height=0.5,
    min_col_width=1.0,
    max_col_width=4.0,
    show_index=True,
    float_fmt="{:,.2f}",
    int_fmt="{:,}",
    dpi=200,
):
    """Salva um DataFrame como imagem usando matplotlib, com visual limpo e legível."""

    def fmt(v):
        if pd.isna(v):
            return "–"
        if isinstance(v, (int, np.integer)):
            return int_fmt.format(v)
        if isinstance(v, (float, np.floating)):
            return float_fmt.format(v)
        return str(v)

    # Prepara dados (opcionalmente inclui o índice como primeira coluna)
    data = df.copy()
    if show_index:
        data.insert(0, data.index.name or "", data.index)
    cell_text = [[fmt(v) for v in row] for row in data.values]
    col_labels = [str(c) for c in data.columns]

    # Largura de cada coluna proporcional ao maior texto (em caracteres)
    char_w = font_size * 0.0095  # polegadas por caractere (aprox.)
    max_len = [
        max([len(col_labels[i])] + [len(r[i]) for r in cell_text])
        for i in range(len(col_labels))
    ]
    col_widths = [
        min(max(l * char_w + 0.5, min_col_width), max_col_width) for l in max_len
    ]

    fig_width = sum(col_widths)
    fig_height = row_height * (len(df) + 1) + (0.7 if title else 0.2)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        colWidths=[w / fig_width for w in col_widths],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.0)

    # Estilo das células
    for (row, col), cell in table.get_celld().items():
        cell.set_height(row_height / fig_height)
        cell.set_edgecolor(edge_color)
        cell.set_linewidth(0.8)

        if row == 0:  # cabeçalho
            cell.set_facecolor(header_color)
            cell.set_text_props(color=header_text_color, weight="bold")
        else:
            cell.set_facecolor(row_colors[(row - 1) % 2])
            cell.set_text_props(color="#2B2B2B")
            if show_index and col == 0:  # primeira coluna (índice) em negrito
                cell.set_text_props(weight="bold", color="#2B2B2B")
                cell.set_text_props(ha="left")
                cell.PAD = 0.05

    if title:
        ax.set_title(title, fontsize=font_size + 4, weight="bold",
                     color=header_color, pad=12)

    fig.savefig(filepath, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)



if __name__ == "__main__":

    sample_path = "../data/raw/olist_orders_dataset.csv"

    try:
        sample_df = pd.read_csv(sample_path)
        summary = dataframe_summary(sample_df)
        print(f"Resumo gerado com sucesso para: {sample_path}\n")
        print(summary.to_string(index=False))
    except FileNotFoundError:
        print(
            f"[Aviso] Arquivo não encontrado em '{sample_path}'.\n"
            "Ajuste o caminho para apontar para um CSV real dentro de data/raw/ "
            "do seu projeto Olist antes de rodar este teste."
        )