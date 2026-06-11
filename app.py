"""
Dashboard d'exploration et visualisation automatique de données
Application Streamlit avec Pandas, Plotly Express et NumPy
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import StringIO

# ─────────────────────────────────────────────
# Configuration de la page
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DataViz Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS personnalisé
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Palette : bleu ardoise profond + accent cyan */
    :root {
        --primary:   #1E3A5F;
        --accent:    #00C2CB;
        --surface:   #F5F7FA;
        --card:      #FFFFFF;
        --text:      #1A1A2E;
        --muted:     #6B7280;
    }

    .main { background-color: var(--surface); }

    /* Titre principal */
    .hero-title {
        font-family: 'Georgia', serif;
        font-size: 2.4rem;
        font-weight: 700;
        color: var(--primary);
        letter-spacing: -0.5px;
        margin-bottom: 0.1rem;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: var(--muted);
        margin-bottom: 1.5rem;
    }
    .accent-line {
        height: 4px;
        width: 60px;
        background: var(--accent);
        border-radius: 2px;
        margin-bottom: 1.5rem;
    }

    /* Cartes de métriques */
    .metric-card {
        background: var(--card);
        border-radius: 12px;
        padding: 1rem 1.4rem;
        border-left: 4px solid var(--accent);
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .metric-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--muted);
        margin-bottom: 2px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--primary);
    }

    /* Badges de type */
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }
    .badge-quant  { background:#DBEAFE; color:#1D4ED8; }
    .badge-qual   { background:#D1FAE5; color:#065F46; }
    .badge-time   { background:#FEF3C7; color:#92400E; }
    .badge-other  { background:#F3F4F6; color:#374151; }

    /* Section headers */
    .section-header {
        font-family: 'Georgia', serif;
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--primary);
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--primary) !important;
    }
    [data-testid="stSidebar"] * {
        color: #E0E7EF !important;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label,
    [data-testid="stSidebar"] .stSlider label {
        color: #B0C4DE !important;
        font-size: 0.82rem !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# FONCTIONS UTILITAIRES
# ═══════════════════════════════════════════════

@st.cache_data
def load_data(file) -> pd.DataFrame:
    """Charge un fichier CSV uploadé et retourne un DataFrame pandas."""
    try:
        content = file.read().decode("utf-8")
        df = pd.read_csv(StringIO(content))
        if df.empty:
            st.error("⚠️ Le fichier CSV est vide.")
            return None
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier : {e}")
        return None


def detect_variable_types(df: pd.DataFrame) -> dict:
    """
    Détecte automatiquement les types de chaque colonne :
    - 'temporal'      : colonnes de type datetime ou détectées comme telles
    - 'quantitative'  : int / float
    - 'qualitative'   : object / category / bool
    - 'other'         : autres cas
    Retourne un dict {col_name: type_label}.
    """
    types = {}
    for col in df.columns:
        # 1. Déjà datetime
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            types[col] = "temporal"
            continue

        # 2. Numérique
        if pd.api.types.is_numeric_dtype(df[col]):
            types[col] = "quantitative"
            continue

        # 3. Tentative de conversion datetime (object)
        if df[col].dtype == object:
            try:
                converted = pd.to_datetime(df[col], infer_datetime_format=True, errors="raise")
                # Réussit uniquement si la majorité des valeurs sont valides
                if converted.notna().mean() > 0.8:
                    df[col] = converted
                    types[col] = "temporal"
                    continue
            except Exception:
                pass

        # 4. Category / object / bool → qualitatif
        if df[col].dtype in ["object", "category", "bool"]:
            types[col] = "qualitative"
            continue

        types[col] = "other"
    return types


def apply_filters(df: pd.DataFrame, col_types: dict) -> pd.DataFrame:
    """
    Affiche dans la sidebar des filtres dynamiques selon le type de chaque colonne.
    Retourne le DataFrame filtré.
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔽 Filtres")

    filtered = df.copy()

    # On n'affiche des filtres que pour quelques colonnes (max 4) pour ne pas surcharger
    shown = 0
    for col, ctype in col_types.items():
        if shown >= 4:
            break
        if ctype == "qualitative":
            unique_vals = df[col].dropna().unique().tolist()
            if 1 < len(unique_vals) <= 30:
                selected = st.sidebar.multiselect(
                    f"{col}",
                    options=unique_vals,
                    default=unique_vals,
                    key=f"filter_{col}",
                )
                if selected:
                    filtered = filtered[filtered[col].isin(selected)]
                shown += 1
        elif ctype == "quantitative":
            col_min = float(df[col].min())
            col_max = float(df[col].max())
            if col_min < col_max:
                chosen = st.sidebar.slider(
                    f"{col}",
                    min_value=col_min,
                    max_value=col_max,
                    value=(col_min, col_max),
                    key=f"slider_{col}",
                )
                filtered = filtered[filtered[col].between(chosen[0], chosen[1])]
                shown += 1

    return filtered


# ─── Fonctions de création de graphiques ───────

def create_histogram(df, col, title=""):
    """Histogramme pour variable quantitative."""
    fig = px.histogram(
        df, x=col, nbins=30,
        title=title or f"Distribution de {col}",
        color_discrete_sequence=["#00C2CB"],
        template="plotly_white",
        marginal="box",
    )
    fig.update_layout(bargap=0.05, title_font_size=16)
    return fig


def create_scatter(df, x_col, y_col, color_col=None, title=""):
    """Scatter plot pour deux variables quantitatives."""
    fig = px.scatter(
        df, x=x_col, y=y_col, color=color_col,
        title=title or f"Relation entre {x_col} et {y_col}",
        template="plotly_white",
        opacity=0.75,
        trendline="ols",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig.update_layout(title_font_size=16)
    return fig


def create_boxplot(df, x_col, y_col, title=""):
    """Boxplot pour qualitative × quantitative."""
    fig = px.box(
        df, x=x_col, y=y_col,
        title=title or f"Distribution de {y_col} par {x_col}",
        color=x_col,
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Bold,
        notched=True,
    )
    fig.update_layout(showlegend=False, title_font_size=16)
    return fig


def create_bar_chart(df, col, title=""):
    """Bar chart pour variable qualitative (compte des occurrences)."""
    counts = df[col].value_counts().reset_index()
    counts.columns = [col, "count"]
    fig = px.bar(
        counts, x=col, y="count",
        title=title or f"Occurrences de {col}",
        color="count",
        color_continuous_scale=["#DBEAFE", "#1E3A5F"],
        template="plotly_white",
        text_auto=True,
    )
    fig.update_layout(coloraxis_showscale=False, title_font_size=16)
    return fig


def create_line_chart(df, x_col, y_col, title=""):
    """Line chart pour variable temporelle."""
    dff = df[[x_col, y_col]].dropna().sort_values(x_col)
    fig = px.line(
        dff, x=x_col, y=y_col,
        title=title or f"Évolution de {y_col} dans le temps",
        template="plotly_white",
        color_discrete_sequence=["#00C2CB"],
        markers=True,
    )
    fig.update_layout(title_font_size=16)
    return fig


def auto_chart(df, x_col, y_col, col_types):
    """
    Choisit automatiquement le bon graphique selon les types de x_col / y_col.
    """
    tx = col_types.get(x_col, "other")
    ty = col_types.get(y_col, "other") if y_col else None

    if tx == "temporal":
        if ty == "quantitative":
            return create_line_chart(df, x_col, y_col), f"Évolution de {y_col} dans le temps"
        return create_bar_chart(df, x_col)

    if tx == "quantitative" and ty is None:
        return create_histogram(df, x_col), f"Distribution de {x_col}"

    if tx == "quantitative" and ty == "quantitative":
        return create_scatter(df, x_col, y_col), f"Relation entre {x_col} et {y_col}"

    if tx == "qualitative" and ty == "quantitative":
        return create_boxplot(df, x_col, y_col), f"Distribution de {y_col} par {x_col}"

    if tx == "quantitative" and ty == "qualitative":
        return create_boxplot(df, y_col, x_col), f"Distribution de {x_col} par {y_col}"

    if tx == "qualitative" and ty is None:
        return create_bar_chart(df, x_col), f"Occurrences de {x_col}"

    # Fallback
    return create_bar_chart(df, x_col), f"Occurrences de {x_col}"


# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════

def main():
    # ── Sidebar ──────────────────────────────────
    st.sidebar.markdown(
        "<div style='font-size:1.4rem;font-weight:700;margin-bottom:0.3rem;'>📊 DataViz Explorer</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("<div style='font-size:0.8rem;opacity:0.6;margin-bottom:1rem;'>Exploration automatique de données</div>", unsafe_allow_html=True)

    page = st.sidebar.radio(
        "Navigation",
        ["📂 Chargement", "🔍 Aperçu & Infos", "📈 Dashboard"],
        label_visibility="collapsed",
    )

    uploaded_file = st.sidebar.file_uploader("Importer un fichier CSV", type=["csv"])

    # ── Héro ─────────────────────────────────────
    st.markdown('<div class="hero-title">DataViz Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Exploration et visualisation automatique de vos données CSV</div>', unsafe_allow_html=True)
    st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)

    # ── Chargement ───────────────────────────────
    if uploaded_file is None:
        st.info("👈 Commencez par importer un fichier CSV depuis la barre latérale.")

        # Exemple de dataset démo
        if st.button("🎲 Charger un dataset de démonstration"):
            np.random.seed(42)
            n = 120
            demo = pd.DataFrame({
                "Date":       pd.date_range("2023-01-01", periods=n, freq="3D"),
                "Région":     np.random.choice(["Nord", "Sud", "Est", "Ouest"], n),
                "Catégorie":  np.random.choice(["A", "B", "C"], n),
                "Ventes":     np.random.normal(5000, 1200, n).round(2),
                "Quantité":   np.random.randint(10, 200, n),
                "Prix":       np.random.uniform(20, 150, n).round(2),
                "Marge":      np.random.normal(0.25, 0.08, n).round(3),
                "Satisfaction": np.random.randint(1, 6, n),
                "Age_client": np.random.randint(18, 70, n),
                "Sexe":       np.random.choice(["H", "F"], n),
                "NPS":        np.random.randint(-100, 101, n),
            })
            # Injecter quelques valeurs manquantes
            for col in ["Marge", "NPS", "Age_client"]:
                demo.loc[np.random.choice(demo.index, 5), col] = np.nan
            st.session_state["df"] = demo
            st.session_state["source"] = "Démonstration"
            st.rerun()
        return

    # Charger depuis le fichier uploadé
    if "df" not in st.session_state or st.session_state.get("source") != uploaded_file.name:
        df = load_data(uploaded_file)
        if df is None:
            return
        st.session_state["df"] = df
        st.session_state["source"] = uploaded_file.name
        st.success(f"✅ Fichier « {uploaded_file.name} » chargé avec succès.")

    df = st.session_state["df"]
    col_types = detect_variable_types(df)

    # ── Filtres (sidebar) ────────────────────────
    df_filtered = apply_filters(df, col_types)

    # ════════════════════════════════════════════
    # PAGE : Chargement
    # ════════════════════════════════════════════
    if "Chargement" in page:
        st.markdown('<div class="section-header">Aperçu du fichier</div>', unsafe_allow_html=True)

        rows_shown = st.slider("Nombre de lignes à afficher", 5, min(50, len(df)), 10)
        st.dataframe(df.head(rows_shown), use_container_width=True)

        # Métriques
        col1, col2, col3, col4 = st.columns(4)
        missing = int(df.isnull().sum().sum())
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Lignes</div><div class="metric-value">{len(df):,}</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Colonnes</div><div class="metric-value">{df.shape[1]}</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Valeurs manquantes</div><div class="metric-value">{missing}</div></div>', unsafe_allow_html=True)
        with col4:
            mem = df.memory_usage(deep=True).sum()
            st.markdown(f'<div class="metric-card"><div class="metric-label">Mémoire</div><div class="metric-value">{mem/1024:.1f} Ko</div></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════
    # PAGE : Aperçu & Infos
    # ════════════════════════════════════════════
    elif "Aperçu" in page:
        st.markdown('<div class="section-header">Types de variables détectés</div>', unsafe_allow_html=True)

        badge_map = {
            "quantitative": ("badge-quant", "Quantitative"),
            "qualitative":  ("badge-qual",  "Qualitative"),
            "temporal":     ("badge-time",  "Temporelle"),
            "other":        ("badge-other", "Autre"),
        }

        cols = st.columns(4)
        for i, (col_name, ctype) in enumerate(col_types.items()):
            badge_cls, badge_lbl = badge_map.get(ctype, ("badge-other", "Autre"))
            with cols[i % 4]:
                st.markdown(
                    f"<div style='margin-bottom:0.6rem;'>"
                    f"<span style='font-size:0.85rem;font-weight:600;'>{col_name}</span><br>"
                    f"<span class='badge {badge_cls}'>{badge_lbl}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # Statistiques descriptives
        st.markdown('<div class="section-header">Statistiques descriptives</div>', unsafe_allow_html=True)
        quant_cols = [c for c, t in col_types.items() if t == "quantitative"]
        if quant_cols:
            st.dataframe(df[quant_cols].describe().T.round(2), use_container_width=True)
        else:
            st.info("Aucune colonne quantitative détectée.")

        # Valeurs manquantes
        st.markdown('<div class="section-header">Valeurs manquantes par colonne</div>', unsafe_allow_html=True)
        missing_df = df.isnull().sum().reset_index()
        missing_df.columns = ["Colonne", "Manquantes"]
        missing_df["% manquant"] = (missing_df["Manquantes"] / len(df) * 100).round(1)
        missing_df = missing_df[missing_df["Manquantes"] > 0]
        if missing_df.empty:
            st.success("✅ Aucune valeur manquante !")
        else:
            fig_miss = px.bar(
                missing_df, x="Colonne", y="% manquant",
                color="% manquant",
                color_continuous_scale=["#FEF3C7", "#DC2626"],
                template="plotly_white",
                title="Pourcentage de valeurs manquantes par colonne",
                text_auto=True,
            )
            fig_miss.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_miss, use_container_width=True)

        # Corrélations
        if len(quant_cols) >= 2:
            st.markdown('<div class="section-header">Matrice de corrélation</div>', unsafe_allow_html=True)
            corr = df[quant_cols].corr().round(2)
            fig_corr = px.imshow(
                corr,
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
                text_auto=True,
                template="plotly_white",
                title="Corrélations (Pearson)",
            )
            st.plotly_chart(fig_corr, use_container_width=True)

    # ════════════════════════════════════════════
    # PAGE : Dashboard
    # ════════════════════════════════════════════
    elif "Dashboard" in page:
        st.markdown('<div class="section-header">Dashboard interactif</div>', unsafe_allow_html=True)

        all_cols = list(df_filtered.columns)
        quant_cols = [c for c, t in col_types.items() if t == "quantitative"]
        qual_cols  = [c for c, t in col_types.items() if t == "qualitative"]

        # ── Sélecteurs ────────────────────────────
        ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 2])
        with ctrl1:
            x_col = st.selectbox("Variable X", options=all_cols, key="x_col")
        with ctrl2:
            y_options = ["(aucune)"] + [c for c in all_cols if c != x_col]
            y_raw = st.selectbox("Variable Y", options=y_options, key="y_col")
            y_col = None if y_raw == "(aucune)" else y_raw
        with ctrl3:
            chart_type = st.selectbox(
                "Type de graphique",
                ["Auto", "Histogramme", "Scatter Plot", "Bar Chart", "Boxplot", "Line Chart"],
                key="chart_type",
            )

        # Couleur optionnelle
        color_col = None
        if chart_type in ["Scatter Plot", "Auto"] and qual_cols:
            color_col = st.selectbox("Couleur par (optionnel)", ["(aucune)"] + qual_cols, key="color_col")
            if color_col == "(aucune)":
                color_col = None

        # ── Génération du graphique ───────────────
        fig = None
        title = ""

        try:
            if df_filtered.empty:
                st.warning("⚠️ Aucune donnée après application des filtres.")
            elif chart_type == "Auto":
                fig, title = auto_chart(df_filtered, x_col, y_col, col_types)
            elif chart_type == "Histogramme":
                if col_types.get(x_col) != "quantitative":
                    st.warning(f"L'histogramme nécessite une colonne quantitative. « {x_col} » est de type {col_types.get(x_col)}.")
                else:
                    title = f"Distribution de {x_col}"
                    fig = create_histogram(df_filtered, x_col, title)
            elif chart_type == "Scatter Plot":
                if y_col is None:
                    st.warning("Sélectionnez une variable Y pour le Scatter Plot.")
                elif col_types.get(x_col) != "quantitative" or col_types.get(y_col) != "quantitative":
                    st.warning("Le Scatter Plot nécessite deux colonnes quantitatives.")
                else:
                    title = f"Relation entre {x_col} et {y_col}"
                    fig = create_scatter(df_filtered, x_col, y_col, color_col, title)
            elif chart_type == "Bar Chart":
                title = f"Occurrences de {x_col}"
                fig = create_bar_chart(df_filtered, x_col, title)
            elif chart_type == "Boxplot":
                if y_col is None:
                    st.warning("Sélectionnez une variable Y pour le Boxplot.")
                elif col_types.get(y_col) != "quantitative":
                    st.warning("La variable Y doit être quantitative pour un Boxplot.")
                else:
                    title = f"Distribution de {y_col} par {x_col}"
                    fig = create_boxplot(df_filtered, x_col, y_col, title)
            elif chart_type == "Line Chart":
                if y_col is None:
                    st.warning("Sélectionnez une variable Y pour le Line Chart.")
                elif col_types.get(x_col) not in ("temporal", "quantitative"):
                    st.warning("La variable X doit être temporelle ou quantitative pour un Line Chart.")
                else:
                    title = f"Évolution de {y_col}"
                    fig = create_line_chart(df_filtered, x_col, y_col, title)

            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Erreur lors de la création du graphique : {e}")

        # ── Suggestions automatiques ──────────────
        st.markdown('<div class="section-header">Visualisations automatiques suggérées</div>', unsafe_allow_html=True)

        tab_quant, tab_qual, tab_time = st.tabs(["📐 Quantitatives", "🏷️ Qualitatives", "🕐 Temporelles"])

        with tab_quant:
            cols_q = [c for c, t in col_types.items() if t == "quantitative"]
            if not cols_q:
                st.info("Aucune colonne quantitative.")
            else:
                chosen = st.selectbox("Choisir une colonne", cols_q, key="sq")
                st.plotly_chart(create_histogram(df_filtered, chosen), use_container_width=True)

        with tab_qual:
            cols_ql = [c for c, t in col_types.items() if t == "qualitative"]
            if not cols_ql:
                st.info("Aucune colonne qualitative.")
            else:
                chosen = st.selectbox("Choisir une colonne", cols_ql, key="sql")
                st.plotly_chart(create_bar_chart(df_filtered, chosen), use_container_width=True)

        with tab_time:
            cols_t = [c for c, t in col_types.items() if t == "temporal"]
            if not cols_t:
                st.info("Aucune colonne temporelle détectée.")
            elif not quant_cols:
                st.info("Aucune colonne quantitative disponible pour l'axe Y.")
            else:
                tc = st.selectbox("Colonne date", cols_t, key="tc")
                yc = st.selectbox("Colonne valeur", quant_cols, key="yc")
                st.plotly_chart(create_line_chart(df_filtered, tc, yc), use_container_width=True)


# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()
