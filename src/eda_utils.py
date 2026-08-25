import pandas as pd
import numpy as np


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

        # top e freq (valor mais frequente)
        value_counts = series.value_counts(dropna=True)
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
            "percent_null": round(percent_null, 2) if pd.notnull(percent_null) else np.nan,
            "unique": unique,
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


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Teste rápido usando um dos CSVs do Olist (ajuste o caminho conforme
    # a estrutura real do seu projeto: data/raw/<nome_do_arquivo>.csv)
    # ------------------------------------------------------------------
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