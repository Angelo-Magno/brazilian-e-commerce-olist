"""
plots.py

Reusable plotting functions for the Olist EDA and modeling project.
Covers: target distribution, categorical/numerical distributions,
correlation analysis, time series trends, geographic breakdowns,
and delay-vs-review relationship validation.

Design principles:
- Every function returns the matplotlib Figure/Axes object (never calls plt.show())
  so plots can be reused in notebooks, saved to reports/figures/, or embedded elsewhere.
- Functions accept an optional `ax` so they can be composed into subplots.
- Functions accept `save_path` to persist figures directly into reports/figures/.
- A single default color is used across all plots for visual consistency
  (see DEFAULT_COLOR / RISK_COLORS below).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Paleta padrão do projeto (consistência visual entre todos os notebooks)
DEFAULT_COLOR = "#4C72B0"
RISK_COLORS = {"low_risk": "#4C72B0", "high_risk": "#C44E52"}
CATEGORICAL_PALETTE = sns.color_palette("viridis", as_cmap=False)
FIGSIZE_DEFAULT = (8, 5)

sns.set_theme(style="dark", palette="viridis")


def _get_categorical_colors(n: int) -> list:
    """Retorna `n` cores da paleta categórica padrão do projeto, ciclando se necessário."""
    base = CATEGORICAL_PALETTE
    if n <= len(base):
        return sns.color_palette("viridis", n_colors=n)
    reps = int(np.ceil(n / len(base)))
    return (list(base) * reps)[:n]


def _save_if_needed(fig: plt.Figure, save_path: Optional[str | Path]) -> None:
    """Salva a figura em disco caso um caminho seja fornecido."""
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")


def plot_target_balance(
    df: pd.DataFrame,
    target_col: str,
    labels: Optional[Sequence[str]] = None,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str | Path] = None,
) -> plt.Axes:
    """
    Plota o balanceamento de classes de uma variável-alvo binária
    (ex: risco de atraso, risco de review ruim).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE_DEFAULT)
    else:
        fig = ax.get_figure()

    counts = df[target_col].value_counts().sort_index()
    pct = counts / counts.sum() * 100

    bars = ax.bar(counts.index.astype(str), counts.values, color=[
        RISK_COLORS["low_risk"], RISK_COLORS["high_risk"]
    ][: len(counts)])

    if labels is not None:
        ax.set_xticklabels(labels)

    for bar, p in zip(bars, pct):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{p:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_title(f"Class balance — {target_col}")
    ax.set_ylabel("Count")
    ax.set_xlabel(target_col)

    _save_if_needed(fig, save_path)
    return ax


def plot_numeric_distribution(
    df: pd.DataFrame,
    column: str,
    bins: int = 50,
    log_scale: bool = False,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str | Path] = None,
) -> plt.Axes:
    """
    Histograma + KDE para uma coluna numérica.
    Útil para inspecionar skewness antes de decidir transformações.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE_DEFAULT)
    else:
        fig = ax.get_figure()

    sns.histplot(df[column].dropna(), bins=bins, kde=True, ax=ax, color=DEFAULT_COLOR)

    if log_scale:
        ax.set_xscale("log")

    skew = df[column].skew()
    ax.set_title(f"Distribution — {column} (skew={skew:.2f})")
    ax.set_xlabel(column)

    _save_if_needed(fig, save_path)
    return ax


def plot_categorical_counts(
    df: pd.DataFrame,
    column: str,
    top_n: Optional[int] = None,
    multicolor: bool = False,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str | Path] = None,
) -> plt.Axes:
    """
    Barras verticais com as categorias mais frequentes de uma coluna.
    Exibe, no topo de cada barra, a quantidade absoluta e o percentual em
    relação ao total (incluindo categorias fora do recorte, se houver).

    top_n: número de categorias a exibir. Se None, exibe todas as categorias.
    multicolor: se True, usa uma cor da paleta categórica do projeto por barra;
                se False (padrão), usa a cor única DEFAULT_COLOR.
    """
    value_counts = df[column].value_counts()
    total = value_counts.sum()  # total geral, sempre, mesmo com recorte
    counts = value_counts if top_n is None else value_counts.head(top_n)
    n_shown = len(counts)

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(12, n_shown * 0.6), 6))
    else:
        fig = ax.get_figure()

    bar_colors = _get_categorical_colors(n_shown) if multicolor else DEFAULT_COLOR
    bars = ax.bar(counts.index.astype(str), counts.values, color=bar_colors)

    # Anotação de quantidade + percentual no topo de cada barra
    max_height = counts.values.max()
    for bar, count in zip(bars, counts.values):
        pct = count / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_height * 0.01,
            f"{count:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylim(0, max_height * 1.15)  # espaço extra para as anotações
    title_scope = f"Top {n_shown}" if top_n is not None else "All"
    ax.set_title(f"{title_scope} categories — {column}")
    ax.set_ylabel("Count")
    ax.set_xlabel(column)
    ax.tick_params(axis="x", rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")

    _save_if_needed(fig, save_path)
    return ax


def plot_risk_rate_by_category(
    df: pd.DataFrame,
    category_col: str,
    target_col: str,
    top_n: int = 15,
    min_count: int = 30,
    multicolor: bool = False,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str | Path] = None,
) -> plt.Axes:
    """
    Plota a TAXA de risco (média do target binário 0/1) por categoria,
    filtrando categorias com poucas observações (min_count) para evitar
    ruído estatístico. Essencial para segmentar risco por vendedor,
    região ou categoria de produto.

    multicolor: se True, usa uma cor da paleta categórica do projeto por barra;
                se False (padrão), usa a cor de risco RISK_COLORS["high_risk"].
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    else:
        fig = ax.get_figure()

    grouped = (
        df.groupby(category_col)[target_col]
        .agg(["mean", "count"])
        .query("count >= @min_count")
        .sort_values("mean", ascending=False)
        .head(top_n)
        .sort_values("mean")
    )

    bar_colors = _get_categorical_colors(len(grouped)) if multicolor else RISK_COLORS["high_risk"]
    ax.barh(grouped.index.astype(str), grouped["mean"], color=bar_colors)
    ax.set_title(f"Risk rate by {category_col} (min {min_count} obs)")
    ax.set_xlabel(f"Mean of {target_col}")
    ax.set_ylabel(category_col)
    ax.axvline(df[target_col].mean(), color="black", linestyle="--", linewidth=1, label="Overall mean")
    ax.legend()

    _save_if_needed(fig, save_path)
    return ax


def plot_correlation_heatmap(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    method: str = "pearson",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str | Path] = None,
) -> plt.Axes:
    """
    Heatmap de correlação entre variáveis numéricas.
    Se `columns` não for informado, usa todas as colunas numéricas do df.
    """
    cols = columns if columns is not None else df.select_dtypes(include=np.number).columns.tolist()
    corr = df[cols].corr(method=method)

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, len(cols) * 0.6), max(5, len(cols) * 0.5)))
    else:
        fig = ax.get_figure()

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(f"Correlation heatmap ({method})")

    _save_if_needed(fig, save_path)
    return ax


def plot_delay_vs_review_relationship(
    df: pd.DataFrame,
    delay_col: str,
    review_score_col: str,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str | Path] = None,
) -> plt.Axes:
    """
    Valida quantitativamente, via EDA, a relação entre atraso e review score
    (Option C: relação validada nos dados históricos, não como feature derivada
    de modelo). Plota a distribuição de review_score condicionada a atraso vs.
    não atraso.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE_DEFAULT)
    else:
        fig = ax.get_figure()

    plot_df = df[[delay_col, review_score_col]].dropna().copy()
    plot_df[delay_col] = plot_df[delay_col].map(
        {0: "On time", 1: "Delayed"} if set(plot_df[delay_col].unique()) <= {0, 1} else {}
    ).fillna(plot_df[delay_col])

    sns.countplot(
        data=plot_df,
        x=review_score_col,
        hue=delay_col,
        ax=ax,
        palette=[RISK_COLORS["low_risk"], RISK_COLORS["high_risk"]],
    )

    ax.set_title(f"{review_score_col} distribution by delivery status")
    ax.set_xlabel("Review score")
    ax.set_ylabel("Count")
    ax.legend(title="Delivery status")

    _save_if_needed(fig, save_path)
    return ax


def plot_monthly_trend(
    df: pd.DataFrame,
    date_col: str,
    value_col: Optional[str] = None,
    agg: str = "count",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str | Path] = None,
) -> plt.Axes:
    """
    Plota a tendência mensal de uma métrica (contagem de pedidos, taxa de
    risco, etc). Se value_col for None, agrega contagem de linhas por mês.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.get_figure()

    temp = df.copy()
    temp["_month"] = pd.to_datetime(temp[date_col]).dt.to_period("M").dt.to_timestamp()

    if value_col is None:
        series = temp.groupby("_month").size()
        ylabel = "Order count"
    else:
        series = temp.groupby("_month")[value_col].agg(agg)
        ylabel = f"{agg}({value_col})"

    ax.plot(series.index, series.values, marker="o", color=DEFAULT_COLOR)
    ax.set_title(f"Monthly trend — {ylabel}")
    ax.set_xlabel("Month")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)

    _save_if_needed(fig, save_path)
    return ax


def plot_geographic_risk(
    df: pd.DataFrame,
    state_col: str,
    target_col: str,
    min_count: int = 30,
    multicolor: bool = False,
    save_path: Optional[str | Path] = None,
) -> plt.Axes:
    """
    Wrapper de plot_risk_rate_by_category especializado para estado/região
    (ex: customer_state, seller_state), usado no dashboard de risco por região.
    """
    return plot_risk_rate_by_category(
        df=df,
        category_col=state_col,
        target_col=target_col,
        top_n=27,  # número de estados brasileiros
        min_count=min_count,
        multicolor=multicolor,
        save_path=save_path,
    )


def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[str] = ("Low risk", "High risk"),
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str | Path] = None,
) -> plt.Axes:
    """
    Matriz de confusão para avaliação dos modelos de classificação binária
    (Risco de Atraso / Risco de Review Ruim).
    """
    from sklearn.metrics import confusion_matrix

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.get_figure()

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar=False,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix")

    _save_if_needed(fig, save_path)
    return ax


def plot_precision_recall_curve(
    y_true: Sequence[int],
    y_proba: Sequence[float],
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str | Path] = None,
) -> plt.Axes:
    """
    Curva Precision-Recall com PR-AUC anotado — métrica preferida do projeto
    para lidar com o desbalanceamento da classe de risco (minoritária).
    """
    from sklearn.metrics import average_precision_score, precision_recall_curve

    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE_DEFAULT)
    else:
        fig = ax.get_figure()

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)

    ax.plot(recall, precision, color=RISK_COLORS["high_risk"], label=f"PR-AUC = {pr_auc:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve")
    ax.legend()

    _save_if_needed(fig, save_path)
    return ax


def plot_roc_curve(
    y_true: Sequence[int],
    y_proba: Sequence[float],
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str | Path] = None,
) -> plt.Axes:
    """
    Curva ROC com ROC-AUC anotado.
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE_DEFAULT)
    else:
        fig = ax.get_figure()

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)

    ax.plot(fpr, tpr, color=DEFAULT_COLOR, label=f"ROC-AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve")
    ax.legend()

    _save_if_needed(fig, save_path)
    return ax


def plot_feature_importance(
    feature_names: Sequence[str],
    importances: Sequence[float],
    top_n: int = 20,
    multicolor: bool = False,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str | Path] = None,
) -> plt.Axes:
    """
    Plota as top_n features mais importantes de um modelo treinado
    (ex: model.feature_importances_ do XGBoost).

    multicolor: se True, usa uma cor da paleta categórica do projeto por barra;
                se False (padrão), usa a cor única DEFAULT_COLOR.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    else:
        fig = ax.get_figure()

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .sort_values("importance")
    )

    bar_colors = _get_categorical_colors(len(importance_df)) if multicolor else DEFAULT_COLOR
    ax.barh(importance_df["feature"], importance_df["importance"], color=bar_colors)
    ax.set_title(f"Top {top_n} feature importances")
    ax.set_xlabel("Importance")

    _save_if_needed(fig, save_path)
    return ax