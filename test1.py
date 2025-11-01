import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import math
import io
import base64
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Configuration de la page
st.set_page_config(
    page_title="Plateforme de Réassurance - Théorie & Data Science",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé amélioré
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1f4e79 0%, #2e75b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2e75b6;
        border-bottom: 3px solid #2e75b6;
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
    }
    .concept-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #2e75b6;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .theory-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #ffc107;
    }
    .data-box {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #17a2b8;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
    }
    .formula-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    .success-box {
        background: linear-gradient(135deg, #d1f2eb 0%, #a3e4d7 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .case-study-box {
        background: linear-gradient(135deg, #e8f4fd 0%, #d1e7f7 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #0d6efd;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# FONCTIONS DATA SCIENCE
# =============================================================================

# Schéma de mapping des colonnes
SCHEMA = {
    "date": ["date", "period", "periode", "month", "quarter", "year"],
    "lob": ["lob", "branche", "line_of_business"],
    "region": ["region", "zone", "pays", "geography"],
    "cedant": ["cedant", "cedente", "ceding_company"],
    "gross_premium": ["gross_premium", "primes_brutes", "gwp"],
    "ceded_premium": ["ceded_premium", "primes_cedees", "ceded"],
    "earned_premium": ["earned_premium", "primes_acquises", "ep"],
    "incurred_claims": ["incurred_claims", "sinistres_encourus", "icl"],
    "paid_claims": ["paid_claims", "sinistres_payes", "pcl"],
    "ibnr": ["ibnr", "reserves_ibnr"],
    "rbns": ["rbns", "reserves_rbns"],
    "acq_expense": ["acq_expense", "frais_acquisition"],
    "adm_expense": ["adm_expense", "frais_admin", "g&a"],
    "investment_income": ["investment_income", "produits_financiers"],
    "claims_count": ["claims_count", "nombre_sinistres"],
    "exposure": ["exposure", "exposition", "policies", "risks"],
    "scr": ["scr", "exigence_capital"],
    "own_funds": ["own_funds", "fonds_propres"],
}
REQUIRED_BASE = ["date", "earned_premium", "incurred_claims"]

def _infer_date_col(s: pd.Series) -> pd.Series:
    """Tente de parser une colonne date."""
    try:
        parsed = pd.to_datetime(s, errors="coerce", dayfirst=True)
        if parsed.notna().mean() > 0.6:
            return parsed
    except Exception:
        pass
    if s.dtype.kind in "if":
        return pd.to_datetime(s.astype(int).astype(str) + "-01-01", errors="coerce")
    return pd.to_datetime(s, errors="coerce")

def make_demo_data(periods=16, seed=42, freq="Q"):
    """Jeu de données de démonstration."""
    rng = np.random.default_rng(seed)
    idx = pd.period_range("2022Q1", periods=periods, freq=freq).to_timestamp()
    lobs = ["Property Cat", "Casualty", "Vie", "Santé"]
    regions = ["EU", "NA", "Asia"]
    rows = []
    for dt in idx:
        for lob in lobs:
            for region in regions[:2]:  # Réduire la combinatoire
                gwp = rng.normal(50, 8) * 100000  # Échelle plus réaliste
                ced = gwp * rng.uniform(0.15, 0.45)
                ep = gwp * rng.uniform(0.75, 0.95)
                cnt = rng.poisson(110 if lob == "Property Cat" else 85)
                expo = rng.integers(900, 1600)
                sev = rng.lognormal(mean=9.35 if lob == "Property Cat" else 9.1, sigma=0.35)
                inc = float(cnt) * float(sev)
                paid = inc * rng.uniform(0.6, 0.9)
                ibnr = inc * rng.uniform(0.06, 0.18)
                rbns = inc * rng.uniform(0.05, 0.15)
                acq = ep * rng.uniform(0.08, 0.14)
                adm = ep * rng.uniform(0.05, 0.09)
                inv = gwp * rng.uniform(0.01, 0.03)
                scr = ep * rng.uniform(0.28, 0.42)
                own = scr * rng.uniform(1.25, 1.9)
                rows.append([
                    dt, "CedantA", lob, region, gwp, ced, ep, inc, paid, ibnr, rbns,
                    acq, adm, cnt, expo, scr, own, inv
                ])
    return pd.DataFrame(rows, columns=[
        "date", "cedant", "lob", "region", "gross_premium", "ceded_premium", "earned_premium",
        "incurred_claims", "paid_claims", "ibnr", "rbns", "acq_expense", "adm_expense",
        "claims_count", "exposure", "scr", "own_funds", "investment_income"
    ])

def auto_map_columns(df: pd.DataFrame):
    """Détecte automatiquement les correspondances colonnes utilisateur -> schéma."""
    mapping = {}
    cols_lower = {c.lower(): c for c in df.columns}
    for key, aliases in SCHEMA.items():
        found = None
        for a in aliases:
            if a in cols_lower:
                found = cols_lower[a]
                break
        mapping[key] = found
    return mapping

def compute_kpis(d: pd.DataFrame) -> pd.DataFrame:
    """Calcule les ratios KPI techniques/financiers/risque."""
    df = d.copy()
    ep = df["earned_premium"].replace(0, np.nan)
    gwp = df.get("gross_premium", pd.Series(np.nan, index=df.index))
    ced = df.get("ceded_premium", pd.Series(0.0, index=df.index))

    df["loss_ratio"] = df["incurred_claims"] / ep
    df["acq_ratio"] = df.get("acq_expense", 0) / ep
    df["adm_ratio"] = df.get("adm_expense", 0) / ep
    df["expense_ratio"] = df["acq_ratio"].fillna(0) + df["adm_ratio"].fillna(0)
    df["combined_ratio"] = df["loss_ratio"].fillna(0) + df["expense_ratio"].fillna(0)
    df["operating_ratio"] = df["combined_ratio"] - (df.get("investment_income", 0) / ep)
    df["cession_ratio"] = ced / gwp.replace(0, np.nan)
    df["retention_ratio"] = (gwp - ced) / gwp.replace(0, np.nan)

    if {"claims_count", "exposure"}.issubset(df.columns):
        df["frequency"] = df["claims_count"] / df["exposure"].replace(0, np.nan)
    if {"incurred_claims", "claims_count"}.issubset(df.columns):
        df["severity"] = df["incurred_claims"] / df["claims_count"].replace(0, np.nan)

    if {"ibnr", "rbns"}.issubset(df.columns):
        df["total_reserves"] = df["ibnr"].fillna(0) + df["rbns"].fillna(0)
        df["reserve_coverage"] = df["total_reserves"] / df["incurred_claims"].replace(0, np.nan)

    if {"scr", "own_funds"}.issubset(df.columns):
        df["solvency_ratio"] = df["own_funds"] / df["scr"].replace(0, np.nan)

    return df

def aggregate_kpis(d: pd.DataFrame, by=["date"]) -> pd.DataFrame:
    """Agrège par dimensions et recalcule les KPI au niveau agrégé."""
    grp = d.groupby(by, dropna=False).agg({
        "gross_premium": "sum", "ceded_premium": "sum", "earned_premium": "sum",
        "incurred_claims": "sum", "paid_claims": "sum", "ibnr": "sum", "rbns": "sum",
        "acq_expense": "sum", "adm_expense": "sum", "investment_income": "sum",
        "claims_count": "sum", "exposure": "sum", "scr": "sum", "own_funds": "sum"
    }).reset_index()
    grp = compute_kpis(grp)
    return grp

def sarimax_forecast(ts: pd.Series, steps: int, order=(1,1,1), seasonal=(0,1,1,4)) -> pd.Series:
    """Prévision SARIMAX avec fallback naïf si historique insuffisant."""
    ts = ts.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if ts.shape[0] < max(24, steps):
        last = ts.iloc[-1] if ts.shape[0] else 0.0
        idx = pd.date_range(datetime.today(), periods=steps, freq="MS")
        return pd.Series([last] * steps, index=idx)
    try:
        model = SARIMAX(ts, order=order, seasonal_order=seasonal,
                        enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        fc = res.get_forecast(steps=steps).predicted_mean
        return fc
    except Exception:
        last = ts.iloc[-1] if ts.shape[0] else 0.0
        idx = pd.date_range(ts.index[-1] + pd.offsets.MonthBegin(1), periods=steps, freq="MS")
        return pd.Series([last] * steps, index=idx)

def add_month_start(df: pd.DataFrame) -> pd.DataFrame:
    """Aligne les dates sur le début de mois."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    return out

def download_button(df: pd.DataFrame, filename: str):
    """Lien de téléchargement CSV."""
    csv = df.to_csv(index=False).encode("utf-8")
    b64 = base64.b64encode(csv).decode()
    st.markdown(
        f'<a download="{filename}" href="data:file/csv;base64,{b64}">📥 Télécharger {filename}</a>',
        unsafe_allow_html=True
    )

# =============================================================================
# PAGES LUDIQUES
# =============================================================================

def page_introduction_ludique():
    """Version ludique de l'introduction avec éléments interactifs"""
    
    st.markdown('<div class="main-header">🎯 Introduction à la Réassurance</div>', unsafe_allow_html=True)
    st.markdown("### *Découvrez le monde fascinant du partage des risques*")
    
    # Bannière d'accueil avec animation CSS
    st.markdown("""
    <style>
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .welcome-banner {
        animation: fadeIn 1s ease-in;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
    
    <div class="welcome-banner">
        <h2>🚀 Bienvenue dans l'univers de la Réassurance</h2>
        <p>Voyagez au cœur de la gestion des risques avec des outils interactifs et des explications claires</p>
    </div>
    """, unsafe_allow_html=True)

    # Cartes interactives
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container():
            st.markdown("""
            <div style='background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); 
                        padding: 1.5rem; border-radius: 15px; text-align: center; 
                        box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
                <h3>🎓 Public</h3>
                <p><b>Débutants, étudiants, professionnels</b></p>
                <p style='font-size: 0.9em;'>Tous niveaux acceptés !</p>
            </div>
            """, unsafe_allow_html=True)
            
    with c2:
        with st.container():
            st.markdown("""
            <div style='background: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%); 
                        padding: 1.5rem; border-radius: 15px; text-align: center; 
                        box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
                <h3>🎯 Objectif</h3>
                <p><b>Vision claire et pratique</b></p>
                <p style='font-size: 0.9em;'>Apprendre en pratiquant</p>
            </div>
            """, unsafe_allow_html=True)
            
    with c3:
        with st.container():
            st.markdown("""
            <div style='background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%); 
                        padding: 1.5rem; border-radius: 15px; text-align: center; 
                        box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
                <h3>📈 Résultat</h3>
                <p><b>Compréhension des mécanismes</b></p>
                <p style='font-size: 0.9em;'>Maîtriser le transfert de risque</p>
            </div>
            """, unsafe_allow_html=True)

    # Section témoignage interactif
    st.markdown("---")
    col_text, col_viz = st.columns([2, 1])
    
    with col_text:
        st.write("""
        ## 💡 Le saviez-vous ?
        
        En tant que spécialiste de la réassurance régionale, j'ai collaboré avec des acteurs majeurs du marché.
        Ce module vise à vous offrir une compréhension **claire et pratique** de la réassurance en tant qu'outil de
        stabilité et de gestion du risque.
        
        ### 🎮 Apprendre en s'amusant
        - **Simulations interactives** pour comprendre les concepts
        - **Quiz personnalisés** pour tester vos connaissances  
        - **Cas concrets** du terrain
        - **Outils visuels** pour une mémorisation facile
        """)
        
        # Badge de progression
        st.markdown("""
        <div style='background: #f8f9fa; padding: 1rem; border-radius: 10px; border-left: 4px solid #28a745;'>
            <h4>🏆 Votre progression</h4>
            <p>Commencez votre aventure dans la réassurance !</p>
            <div style='background: #e9ecef; height: 10px; border-radius: 5px; margin: 10px 0;'>
                <div style='background: #28a745; width: 10%; height: 100%; border-radius: 5px;'></div>
            </div>
            <p><small>10% complété - Continuez !</small></p>
        </div>
        """, unsafe_allow_html=True)

    with col_viz:
        # Mini jeu interactif : comprendre le flux de risque
        st.subheader("🎮 Mini-Lab : Flux du Risque")
        
        prime_totale = st.slider("💰 Prime totale collectée", 500000, 2000000, 1000000, step=100000)
        part_cedee = st.slider("📤 Part cédée au réassureur", 10, 60, 40)
        
        prime_conservee = prime_totale * (100 - part_cedee) / 100
        prime_cedee = prime_totale * part_cedee / 100
        
        # Graphique interactif
        fig = go.Figure(data=[go.Pie(
            labels=[f'Conservé par assureur ({100-part_cedee}%)', f'Cédé au réassureur ({part_cedee}%)'],
            values=[prime_conservee, prime_cedee],
            hole=.3,
            marker_colors=['#FF9999', '#66B2FF']
        )])
        fig.update_layout(
            title="Répartition des primes",
            annotations=[dict(text=f'{prime_totale:,.0f}€', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"""
        **📊 Résultat de votre simulation :**
        - Prime conservée : **{prime_conservee:,.0f} €**
        - Prime cédée : **{prime_cedee:,.0f} €**
        """)

    # Section concept clé avec animation
    st.markdown("---")
    st.markdown("""
    <div style='background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%); 
                padding: 2rem; border-radius: 15px; border-left: 6px solid #ffc107;'>
        <h2>💡 Concept Clé à Retenir</h2>
        <p style='font-size: 1.2em; font-weight: bold;'>
        La réassurance est un mécanisme de <span style='color: #dc3545;'>PARTAGE DU RISQUE</span> entre assureurs et réassureurs
        pour préserver la <span style='color: #28a745;'>SOLVABILITÉ</span> et la <span style='color: #17a2b8;'>CONFIANCE</span> du système financier.
        </p>
        <p style='text-align: center; margin-top: 1rem;'>
        <span style='font-size: 3em;'>🛡️</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Diagramme Sankey interactif
    st.markdown("---")
    st.subheader("🌊 Visualisation des Flux de Réassurance")
    
    # Contrôles interactifs pour le diagramme
    col_controls = st.columns(3)
    with col_controls[0]:
        sinistres_directs = st.slider("Sinistres directs", 40, 80, 60)
    with col_controls[1]:
        part_reassureur = st.slider("Part réassureur", 20, 60, 40)
    with col_controls[2]:
        portefeuille_total = st.slider("Portefeuille total", 80, 120, 100)

    # Données pour le diagramme Sankey
    df_flow = pd.DataFrame({
        "source": ["Portefeuille", "Assureur", "Assureur"],
        "target": ["Assureur", "Sinistres courants", "Réassureur"],
        "value": [portefeuille_total, sinistres_directs, part_reassureur]
    })
    
    labels = ["Portefeuille", "Assureur", "Réassureur", "Sinistres courants"]
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    
    sankey_fig = go.Figure(data=[go.Sankey(
        node=dict(
            label=labels,
            pad=18,
            thickness=18,
            line=dict(color="#cfe0ee", width=1),
            color=["#4CAF50", "#2196F3", "#FF9800", "#F44336"]
        ),
        link=dict(
            source=[label_to_idx[s] for s in df_flow["source"]],
            target=[label_to_idx[t] for t in df_flow["target"]],
            value=df_flow["value"],
            color="rgba(0,0,0,0.2)"
        )
    )])
    
    sankey_fig.update_layout(
        height=400, 
        title="Flux simplifié du risque entre assureur et réassureur",
        font=dict(size=12)
    )
    
    st.plotly_chart(sankey_fig, use_container_width=True)
    
    # Légende interactive
    st.markdown("""
    <div style='background: #f8f9fa; padding: 1rem; border-radius: 10px; margin-top: 1rem;'>
        <h4>🎯 Comment lire ce diagramme :</h4>
        <ul>
            <li>🟢 <b>Portefeuille</b> : Tous les risques assurés</li>
            <li>🔵 <b>Assureur</b> : La compagnie qui garde une partie du risque</li>
            <li>🟠 <b>Réassureur</b> : Celui qui prend le risque excédentaire</li>
            <li>🔴 <b>Sinistres</b> : Les pertes qui surviennent</li>
        </ul>
        <p><i>💡 Plus le flux est épais, plus le montant est important !</i></p>
    </div>
    """, unsafe_allow_html=True)

    # Appel à l'action
    st.markdown("---")
    col_cta1, col_cta2, col_cta3 = st.columns(3)
    
    with col_cta1:
        if st.button("📚 Commencer la formation", use_container_width=True):
            st.session_state.current_page = "Principes Fondamentaux"
            st.rerun()
    
    with col_cta2:
        if st.button("🎮 Voir les simulateurs", use_container_width=True):
            st.session_state.current_page = "Calculateurs Avancés"
            st.rerun()
    
    with col_cta3:
        if st.button("📊 Explorer les données", use_container_width=True):
            st.session_state.current_page = "Analyse Data Science"
            st.rerun()

def page_principes_ludique():
    """Version ludique des principes fondamentaux"""
    
    st.markdown('<div class="main-header">🎓 Principes Fondamentaux de la Réassurance</div>', unsafe_allow_html=True)
    st.markdown("### *Comprendre les bases essentielles de la gestion du risque*")
    
    # Barre de progression du chapitre
    st.markdown("""
    <div style='background: #e9ecef; padding: 0.5rem; border-radius: 10px; margin-bottom: 2rem;'>
        <div style='display: flex; justify-content: space-between; align-items: center;'>
            <span><b>Progression du chapitre :</b></span>
            <span>25% complété</span>
        </div>
        <div style='background: #28a745; width: 25%; height: 8px; border-radius: 4px; margin-top: 5px;'></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Introduction avec animation
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; padding: 2rem; border-radius: 15px; margin-bottom: 2rem;'>
        <h2>🤔 Qu'est-ce que la réassurance ?</h2>
        <p style='font-size: 1.2em;'>
        C'est un <b>CONTRAT</b> par lequel une compagnie d'assurance transfère une partie de ses risques
        à un réassureur. Cela permet de <b>PARTAGER</b> et <b>ABSORBER LES CHOCS</b> financiers.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sections interactives avec onglets
    tab1, tab2, tab3 = st.tabs(["🎯 Généralités", "🔄 Les Formes", "💡 Le Savoir"])
    
    with tab1:
        st.subheader("1️⃣ Les Bases Essentielles")
        
        # Cartes interactives avec hover effects
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style='background: white; padding: 1.5rem; border-radius: 15px; 
                        box-shadow: 0 4px 15px rgba(0,0,0,0.1); border-top: 4px solid #007bff;
                        transition: transform 0.3s ease;'
                        onmouseover="this.style.transform='translateY(-5px)'" 
                        onmouseout="this.style.transform='translateY(0)'">
                <div style='text-align: center; font-size: 2em;'>🛡️</div>
                <h4 style='text-align: center;'>Protection</h4>
                <p>Le réassureur protège l'assureur contre des pertes élevées</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div style='background: white; padding: 1.5rem; border-radius: 15px; 
                        box-shadow: 0 4px 15px rgba(0,0,0,0.1); border-top: 4px solid #28a745;
                        transition: transform 0.3s ease;'
                        onmouseover="this.style.transform='translateY(-5px)'" 
                        onmouseout="this.style.transform='translateY(0)'">
                <div style='text-align: center; font-size: 2em;'>🏢</div>
                <h4 style='text-align: center;'>Cédante</h4>
                <p>L'assureur qui transfère le risque augmente sa capacité</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div style='background: white; padding: 1.5rem; border-radius: 15px; 
                        box-shadow: 0 4px 15px rgba(0,0,0,0.1); border-top: 4px solid #ffc107;
                        transition: transform 0.3s ease;'
                        onmouseover="this.style.transform='translateY(-5px)'" 
                        onmouseout="this.style.transform='translateY(0)'">
                <div style='text-align: center; font-size: 2em;'>💪</div>
                <h4 style='text-align: center;'>Renforcement</h4>
                <p>La réassurance renforce la solvabilité et la confiance</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Timeline interactive
        st.subheader("📅 Évolution du Concept")
        
        timeline_data = {
            'Période': ['14e siècle', '17e siècle', '19e siècle', '20e siècle', 'Aujourd\'hui'],
            'Événement': [
                'Premières formes à Londres',
                'Développement à Hambourg', 
                'Compagnies spécialisées',
                'Marché global',
                'Solutions complexes'
            ],
            'Impact': ['Faible', 'Moyen', 'Important', 'Majeur', 'Critique']
        }
        
        fig_timeline = px.scatter(timeline_data, x='Période', y='Impact', 
                                size=[10, 20, 30, 40, 50], color='Impact',
                                title="Évolution historique de la réassurance")
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    with tab2:
        st.subheader("2️⃣ Les Deux Grandes Familles")
        
        # Comparaison visuelle interactive
        col_prop, col_nonprop = st.columns(2)
        
        with col_prop:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%); 
                        padding: 2rem; border-radius: 15px; height: 100%;'>
                <h3>🟢 Réassurance Proportionnelle</h3>
                <div style='text-align: center; font-size: 4em;'>⚖️</div>
                <ul>
                    <li><b>Partage</b> primes & sinistres</li>
                    <li>Selon un <b>pourcentage</b></li>
                    <li>Exemple : <i>Quota Share, Surplus</i></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Mini simulateur proportionnel
            st.subheader("🎮 Mini-Lab Proportionnel")
            prime_base = st.number_input("Prime de base", 100000, 1000000, 500000, step=50000)
            taux_cession = st.slider("Taux de cession", 10, 90, 40)
            
            prime_cedee = prime_base * taux_cession / 100
            st.metric("Prime cédée", f"{prime_cedee:,.0f} €")
        
        with col_nonprop:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%); 
                        padding: 2rem; border-radius: 15px; height: 100%;'>
                <h3>🔵 Réassurance Non-Proportionnelle</h3>
                <div style='text-align: center; font-size: 4em;'>🎯</div>
                <ul>
                    <li>Intervention au-delà d'un <b>seuil</b></li>
                    <li>Protection contre les <b>gros sinistres</b></li>
                    <li>Exemple : <i>Excess of Loss, Stop Loss</i></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Mini simulateur non-proportionnel
            st.subheader("🎮 Mini-Lab Non-Proportionnel")
            sinistre = st.number_input("Montant sinistre", 100000, 2000000, 750000, step=50000)
            retention = st.slider("Rétention", 100000, 500000, 200000)
            
            if sinistre > retention:
                couverture = sinistre - retention
            else:
                couverture = 0
                
            st.metric("Couverture réassureur", f"{couverture:,.0f} €")
    
    with tab3:
        st.subheader("💡 Le Coin du Expert")
        
        # Citation inspirante
        st.markdown("""
        <div style='background: #f8f9fa; padding: 2rem; border-radius: 15px; border-left: 6px solid #17a2b8;'>
            <blockquote style='font-style: italic; font-size: 1.2em; color: #555;'>
            "La réassurance n'est pas une dépense, c'est un investissement dans la stabilité."
            </blockquote>
            <p style='text-align: right; margin-top: 1rem; font-weight: bold;'>
            — Expert en gestion des risques
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quiz interactif
        st.subheader("🧠 Vérifiez Vos Connaissances")
        
        quiz_question = """
        **Question :** Quelle est la principale différence entre réassurance proportionnelle et non-proportionnelle ?
        """
        
        st.markdown(quiz_question)
        
        col_quiz1, col_quiz2 = st.columns(2)
        
        with col_quiz1:
            if st.button("A - Le partage systématique vs protection seuil", use_container_width=True):
                st.success("🎉 Exact ! La proportionnelle partage tout, la non-proportionnelle protège au-delà d'un seuil.")
        
        with col_quiz2:
            if st.button("B - Le type de risques couverts", use_container_width=True):
                st.error("❌ Pas tout à fait. Les deux types peuvent couvrir les mêmes risques, mais avec des mécanismes différents.")
        
        # Récompense
        st.markdown("""
        <div style='background: linear-gradient(135deg, #ffd89b 0%, #19547b 100%); 
                    color: white; padding: 1.5rem; border-radius: 15px; text-align: center; margin-top: 2rem;'>
            <h3>🏆 Félicitations !</h3>
            <p>Vous maîtrisez maintenant les principes fondamentaux de la réassurance !</p>
            <p><b>Prochaine étape :</b> Découvrir les types de contrats</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Navigation entre pages
    st.markdown("---")
    col_nav1, col_nav2, col_nav3 = st.columns(3)
    
    with col_nav1:
        if st.button("⬅️ Page précédente", use_container_width=True):
            st.session_state.current_page = "Introduction"
            st.rerun()
    
    with col_nav2:
        st.markdown("""
        <div style='text-align: center;'>
            <p><b>Position actuelle</b></p>
            <p>🎓 Principes Fondamentaux</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_nav3:
        if st.button("Page suivante ➡️", use_container_width=True):
            st.session_state.current_page = "Types de Contrats"
            st.rerun()

# =============================================================================
# INTERFACE PRINCIPALE
# =============================================================================

# Sidebar Navigation
st.sidebar.title("🔍 Navigation")

section = st.sidebar.radio("Modules", [
    "🏠 Accueil & Présentation",
    "🎯 Introduction Ludique",
    "🎓 Principes Ludiques", 
    "📝 Types de Contrats Ludiques",
    "🏛️ Acteurs & Flux Ludiques",
    "📚 Concepts Fondamentaux", 
    "📈 Traités Proportionnels",
    "⚡ Traités Non-Proportionnels",
    "💰 Tarification Technique",
    "📊 Comptabilité Technique",
    "🌪️ Gestion des Catastrophes",
    "🛡️ Solvabilité & Réglementation",
    "📋 Études de Cas Concrets",
    "📊 Analyse Data Science",
    "🧮 Calculateurs Avancés"
])

# Titre principal
st.markdown('<div class="main-header">🏛️ PLATEFORME COMPLÈTE DE RÉASSURANCE</div>', unsafe_allow_html=True)
st.markdown("### *Théorie, Pratique et Data Science pour Professionnels et Apprenants*")

# =============================================================================
# ROUTING DES PAGES
# =============================================================================

if section == "🏠 Accueil & Présentation":
    # ... (le code de la page d'accueil unifiée que nous avons déjà)
    pass

elif section == "🎯 Introduction Ludique":
    page_introduction_ludique()

elif section == "🎓 Principes Ludiques":
    page_principes_ludique()

elif section == "📝 Types de Contrats Ludiques":
    # ... (implémentez cette fonction similaire aux autres)
    st.info("Page en cours de développement...")

elif section == "🏛️ Acteurs & Flux Ludiques":
    # ... (implémentez cette fonction similaire aux autres)
    st.info("Page en cours de développement...")

elif section == "📚 Concepts Fondamentaux":
    # ... (votre code existant)
    pass

elif section == "📈 Traités Proportionnels":
    # ... (votre code existant)
    pass

elif section == "⚡ Traités Non-Proportionnels":
    # ... (votre code existant)
    pass

elif section == "💰 Tarification Technique":
    # ... (votre code existant)
    pass

elif section == "📊 Comptabilité Technique":
    # ... (votre code existant)
    pass

elif section == "🌪️ Gestion des Catastrophes":
    # ... (votre code existant)
    pass

elif section == "🛡️ Solvabilité & Réglementation":
    # ... (votre code existant)
    pass

elif section == "📋 Études de Cas Concrets":
    # ... (votre code existant)
    pass

elif section == "📊 Analyse Data Science":
    # ... (votre code existant)
    pass

elif section == "🧮 Calculateurs Avancés":
    # ... (votre code existant)
    pass

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")

col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    st.markdown("**📚 Références Techniques**")
    st.markdown("""
    - Code des Assurances
    - Directive Solvabilité II
    - Normes IFRS 17
    - Principes Actuariels
    - Standards de réassurance
    """)

with col_f2:
    st.markdown("**🔍 Glossaire Technique**")
    st.markdown("""
    - Cédante / Réassureur
    - Traités / Facultatif
    - Prime / Commission
    - Rétention / Cession
    - SCR / MCR
    """)

with col_f3:
    st.markdown("**📞 Support Pédagogique**")
    st.markdown("""
    Xataxeli MBA - Programme Réassurance  
    📧 ibugueye@ngorweb.com  
    🌐 www.ngorweb.com
    """)

st.markdown("---")
st.markdown(
    "**Plateforme pédagogique Xataxeli MBA - Réassurance & Data Science** | "
    "© 2024 - Tous droits réservés | "
    "**Version Professionnelle 4.0**"
)

# =============================================================================
# FONCTIONNALITÉS AVANCÉES SIDEBAR
# =============================================================================
st.sidebar.markdown("---")
st.sidebar.subheader("🛠️ Outils Professionnels")

if st.sidebar.button("📥 Exporter l'Analyse Complète"):
    st.sidebar.success("Fonctionnalité d'export activée")

if st.sidebar.button("🔄 Réinitialiser les Données"):
    st.experimental_rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("**🔐 Session Utilisateur**")
st.sidebar.info("Connecté en tant que : Étudiant BIGDAA MBA")

# Mode démo avancé
demo_mode = st.sidebar.checkbox("Mode Démonstration Avancé")
if demo_mode:
    st.sidebar.info("""
    **Fonctionnalités démo activées:**
    - Données de test complètes
    - Simulations avancées
    - Scénarios pré-configurés
    """)
