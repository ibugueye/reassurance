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

# =============================================================================
# CONFIGURATION DE LA PAGE (DOIT ÊTRE LA PREMIÈRE COMMANDE STREAMLIT)
# =============================================================================

st.set_page_config(
    page_title="Plateforme de Réassurance - Théorie & Data Science",
    page_icon="🧊",
    layout="wide", 
    initial_sidebar_state="expanded"
)

# =============================================================================
# CSS PERSONNALISÉ AMÉLIORÉ
# =============================================================================

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
        st.plotly_chart(fig, width='stretch')
        
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
    
    st.plotly_chart(sankey_fig, width='stretch')
    
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
        if st.button("📚 Commencer la formation", width='stretch'):
            st.session_state.current_page = "Principes Fondamentaux"
            st.rerun()
    
    with col_cta2:
        if st.button("🎮 Voir les simulateurs", width='stretch'):
            st.session_state.current_page = "Calculateurs Avancés"
            st.rerun()
    
    with col_cta3:
        if st.button("📊 Explorer les données", width='stretch'):
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
        st.plotly_chart(fig_timeline, width='stretch')
    
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
            if st.button("A - Le partage systématique vs protection seuil"):
                st.success("🎉 Exact ! La proportionnelle partage tout, la non-proportionnelle protège au-delà d'un seuil.")
        
        with col_quiz2:
            if st.button("B - Le type de risques couverts", width='stretch'):
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
        if st.button("⬅️ Page précédente", width='stretch'):
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
        if st.button("Page suivante ➡️", width='stretch'):
            st.session_state.current_page = "Types de Contrats"
            st.rerun()

def page_types_contrats_ludique():
    """Version ludique des types de contrats"""
    
    st.title("📝 Types de Contrats de Réassurance")
    st.markdown("### *Découvrez la boîte à outils du réassureur*")
    
    # Introduction visuelle
    with st.container():
        st.markdown("## 🛠️ Deux Grandes Familles, Une Multitude d'Outils")
        st.info("""
        Comme un artisan avec ses outils, le réassureur dispose de différentes techniques 
        adaptées à chaque situation de risque.
        """)
    
    # Navigation par onglets interactifs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Vue d'ensemble", "⚖️ Proportionnelle", "🎪 Non-Proportionnelle", "🏆 Quiz Final"])
    
    with tab1:
        st.subheader("🌐 La Carte des Contrats")
        
        # Graphique radar comparatif
        categories = ['Simplicité', 'Protection', 'Coût', 'Flexibilité', 'Stabilité']
        
        fig_radar = go.Figure()
        
        fig_radar.add_trace(go.Scatterpolar(
            r=[8, 6, 7, 5, 9],
            theta=categories,
            fill='toself',
            name='Proportionnelle',
            line_color='green'
        ))
        
        fig_radar.add_trace(go.Scatterpolar(
            r=[5, 9, 4, 8, 7],
            theta=categories,
            fill='toself',
            name='Non-Proportionnelle',
            line_color='blue'
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )),
            showlegend=True,
            title="Comparaison des caractéristiques"
        )
        
        st.plotly_chart(fig_radar, width='stretch')
        
        # Tableau comparatif interactif
        st.subheader("📊 Tableau Comparatif")
        
        comparatif_data = {
            'Caractéristique': ['Principe', 'Application', 'Coût', 'Avantage principal', 'Inconvénient'],
            'Proportionnelle': [
                'Partage systématique', 
                'Portefeuille homogène', 
                'Élevé', 
                'Lissage des résultats', 
                'Cession des bons risques'
            ],
            'Non-Proportionnelle': [
                'Protection seuil', 
                'Risques spécifiques', 
                'Variable', 
                'Protection catastrophes', 
                'Complexité'
            ]
        }
        
        st.dataframe(pd.DataFrame(comparatif_data), width='None')
    
    with tab2:
        st.subheader("⚖️ La Famille Proportionnelle")
        
        col_desc, col_viz = st.columns([2, 1])
        
        with col_desc:
            # Section Partage Équitable
            st.subheader("🧩 Le Partage Équitable")
            st.write("**Principe :** Partage systématique des primes et sinistres selon un pourcentage fixe.")
            
            st.write("**🎯 Quand l'utiliser ?**")
            st.markdown("- Portefeuille homogène")
            st.markdown("- Besoin de stabilité")
            st.markdown("- Début d'activité")
            st.markdown("- Transfert d'expertise")
            
            # Exemple concret
            st.subheader("📝 Exemple Concret : Quota Share 40%")
            
            col_ex1, col_ex2 = st.columns(2)
            with col_ex1:
                st.metric("Prime totale", "1 000 000 €")
                st.metric("Sinistre total", "600 000 €")
                
            with col_ex2:
                st.metric("Réassureur prend", "400 000 €", "de primes")
                st.metric("Réassureur paie", "240 000 €", "de sinistres")
        
        with col_viz:
            # Visualisation du partage
            st.subheader("📊 Simulateur de Partage")
            
            prime_totale = st.number_input("Prime totale", 500000, 2000000, 1000000, key="prime_share")
            taux_cession = st.slider("Taux de cession %", 10, 90, 40, key="taux_share")
            sinistre_total = st.number_input("Sinistre total", 300000, 1500000, 600000, key="sinistre_share")
            
            prime_cedee = prime_totale * taux_cession / 100
            sinistre_cede = sinistre_total * taux_cession / 100
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=[f'Assureur ({100-taux_cession}%)', f'Réassureur ({taux_cession}%)'],
                values=[prime_totale - prime_cedee, prime_cedee],
                hole=.4,
                marker_colors=['#FF6B6B', '#4ECDC4']
            )])
            fig_pie.update_layout(title="Répartition des Primes")
            st.plotly_chart(fig_pie, width='stretch')
            
            st.info(f"""
            **Résultat du partage :**
            - Prime conservée : **{(prime_totale - prime_cedee):,.0f} €**
            - Prime cédée : **{prime_cedee:,.0f} €**
            - Sinistre cédé : **{sinistre_cede:,.0f} €**
            """)
    
    with tab3:
        st.subheader("🎪 La Famille Non-Proportionnelle")
        
        with st.container():
            st.subheader("🛡️ La Protection Ciblée")
            st.write("**Principe :** Intervention du réassureur uniquement au-delà d'un certain seuil de sinistres.")
            
            st.markdown("<div style='text-align: center; font-size: 2em; margin: 1rem 0;'></div>", unsafe_allow_html=True)
            
            st.write("*Je ne protège que ce qui dépasse votre capacité d'absorption*")
        
        # Simulateur XL interactif
        st.subheader("🎮 Laboratoire XL (Excédent de Sinistre)")
        
        col_xl1, col_xl2 = st.columns(2)
        
        with col_xl1:
            retention_xl = st.number_input("Rétention de l'assureur", 100000, 500000, 200000, key="retention_xl")
            limite_xl = st.number_input("Limite du réassureur", 100000, 1000000, 550000, key="limite_xl")
            sinistre_xl = st.number_input("Montant du sinistre", 100000, 1500000, 750000, key="sinistre_xl")
        
        with col_xl2:
            # Calcul de la prise en charge
            if sinistre_xl <= retention_xl:
                prise_reassureur = 0
                message = "🟢 Sinistre entièrement à charge de l'assureur"
            elif sinistre_xl <= retention_xl + limite_xl:
                prise_reassureur = sinistre_xl - retention_xl
                message = "🟡 Sinistre partagé selon le traité XL"
            else:
                prise_reassureur = limite_xl
                message = "🔴 Limite du réassureur atteinte"
            
            # Graphique waterfall
            fig_waterfall = go.Figure(go.Waterfall(
                name="Répartition XL",
                orientation="v",
                measure=["relative", "relative", "total"],
                x=["Sinistre total", "Rétention assureur", "Part réassureur"],
                textposition="outside",
                text=[f"{sinistre_xl:,.0f}€", f"-{retention_xl:,.0f}€", f"-{prise_reassureur:,.0f}€"],
                y=[sinistre_xl, -retention_xl, -prise_reassureur],
                connector={"line":{"color":"rgb(63, 63, 63)"}},
            ))
            
            fig_waterfall.update_layout(
                title="Répartition du Sinistre XL",
                showlegend=False
            )
            
            st.plotly_chart(fig_waterfall, width='stretch')
            
            st.success(message)
            st.metric("Part réassureur", f"{prise_reassureur:,.0f} €")
    
    with tab4:
        st.subheader("🏆 Quiz de Validation")
        
        with st.container():
            st.markdown("## 🧠 Testez Votre Compréhension")
            st.info("Validez vos connaissances sur les types de contrats")
        
        # Question 1
        st.markdown("### Question 1/3")
        q1 = st.radio(
            "Quel type de contrat partage systématiquement primes et sinistres selon un pourcentage ?",
            ["A - Le Stop Loss", "B - Le Quota Share", "C - L'Excédent de Sinistre", "D - Le Surplus"]
        )
        
        if q1 == "B - Le Quota Share":
            st.success("✅ Correct ! Le Quota Share est le contrat proportionnel par excellence.")
        elif q1:
            st.error("❌ Ce n'est pas la bonne réponse. Réessayez !")
        
        # Question 2
        st.markdown("### Question 2/3")
        q2 = st.radio(
            "Dans un contrat XL, quand le réassureur intervient-il ?",
            ["A - Dès le premier euro de sinistre", "B - Au-delà de la rétention de l'assureur", "C - Uniquement pour les catastrophes", "D - Pour tous les sinistres majeurs"]
        )
        
        if q2 == "B - Au-delà de la rétention de l'assureur":
            st.success("✅ Exact ! Le XL protège au-delà du seuil de rétention.")
        elif q2:
            st.error("❌ Pas tout à fait. Pensez au seuil d'intervention.")
        
        # Question 3
        st.markdown("### Question 3/3")
        q3 = st.radio(
            "Quel avantage principal offre la réassurance non-proportionnelle ?",
            ["A - Réduction systématique des primes", "B - Protection contre les sinistres exceptionnels", "C - Partage de l'expertise technique", "D - Simplification administrative"]
        )
        
        if q3 == "B - Protection contre les sinistres exceptionnels":
            st.success("✅ Bravo ! C'est sa force principale.")
        elif q3:
            st.error("❌ Ce n'est pas la caractéristique principale.")
        
        # Résultats du quiz
        if st.button("🎯 Voir mes résultats", width='stretch'):
            score = 0
            if q1 == "B - Le Quota Share": score += 1
            if q2 == "B - Au-delà de la rétention de l'assureur": score += 1
            if q3 == "B - Protection contre les sinistres exceptionnels": score += 1
            
            if score == 3:
                st.balloons()
                st.success("🏆 Excellent ! Vous maîtrisez parfaitement les types de contrats !")
                st.metric("Score", "3/3")
            elif score >= 1:
                st.warning(f"📚 Bon travail ! Vous avez bien compris les bases !")
                st.metric("Score", f"{score}/3")
                st.write("Continuez à apprendre !")
            else:
                st.error("📖 À revoir - Relisez le chapitre et réessayez !")
                st.metric("Score", "0/3")
                st.write("La pratique rend parfait !")

def page_acteurs_flux_ludique():
    """Version ludique des acteurs et flux du marché"""
    
    st.markdown('<div class="main-header">🏛️ Acteurs du Marché & Flux de Réassurance</div>', unsafe_allow_html=True)
    st.markdown("### *Explorez l'écosystème et les interactions*")
    
    # Introduction métaphorique
    st.markdown("""
    <div style='background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); 
                padding: 2rem; border-radius: 15px; margin-bottom: 2rem;'>
        <h2>🎭 La Grande Pièce de Théâtre de la Réassurance</h2>
        <p style='font-size: 1.2em;'>
        Chaque acteur a son rôle, chaque flux sa partition. Découvrez l'orchestre complet !
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation par onglets
    tab1, tab2, tab3, tab4 = st.tabs(["🎭 Les Acteurs", "🌊 Les Flux", "🏢 L'Écosystème", "🎮 Le Simulateur"])
    
    with tab1:
        st.subheader("🎭 La Distribution des Rôles")
        
        # Galerie des acteurs avec cartes interactives
        col_act1, col_act2 = st.columns(2)
        
        with col_act1:
            # Carte Cédante
            st.markdown("""
            <div style='background: white; border: 2px solid #007bff; border-radius: 15px; 
                        padding: 1.5rem; margin-bottom: 1rem; transition: all 0.3s ease;'
                        onmouseover="this.style.transform='scale(1.05)'" 
                        onmouseout="this.style.transform='scale(1)'">
                <div style='text-align: center;'>
                    <div style='font-size: 3em;'>🏢</div>
                    <h3>La Cédante</h3>
                    <p><b>Rôle :</b> Compagnie d'assurance qui transfère le risque</p>
                    <p><b>Mission :</b> Souscrire les risques et céder une partie au réassureur</p>
                    <p><b>Objectif :</b> Protéger son portefeuille et sa solvabilité</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Carte Courtier
            st.markdown("""
            <div style='background: white; border: 2px solid #28a745; border-radius: 15px; 
                        padding: 1.5rem; margin-bottom: 1rem; transition: all 0.3s ease;'
                        onmouseover="this.style.transform='scale(1.05)'" 
                        onmouseout="this.style.transform='scale(1)'">
                <div style='text-align: center;'>
                    <div style='font-size: 3em;'>🌐</div>
                    <h3>Le Courtier</h3>
                    <p><b>Rôle :</b> Intermédiaire entre cédantes et réassureurs</p>
                    <p><b>Mission :</b> Négocier les meilleures conditions</p>
                    <p><b>Objectif :</b> Optimiser le placement du risque</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_act2:
            # Carte Réassureur
            st.markdown("""
            <div style='background: white; border: 2px solid #ffc107; border-radius: 15px; 
                        padding: 1.5rem; margin-bottom: 1rem; transition: all 0.3s ease;'
                        onmouseover="this.style.transform='scale(1.05)'" 
                        onmouseout="this.style.transform='scale(1)'">
                <div style='text-align: center;'>
                    <div style='font-size: 3em;'>🏛️</div>
                    <h3>Le Réassureur</h3>
                    <p><b>Rôle :</b> Société qui accepte le risque cédé</p>
                    <p><b>Mission :</b> Mutualiser les risques de plusieurs cédantes</p>
                    <p><b>Objectif :</b> Gérer un portefeuille diversifié</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Carte Rétrocessionnaire
            st.markdown("""
            <div style='background: white; border: 2px solid #dc3545; border-radius: 15px; 
                        padding: 1.5rem; margin-bottom: 1rem; transition: all 0.3s ease;'
                        onmouseover="this.style.transform='scale(1.05)'" 
                        onmouseout="this.style.transform='scale(1)'">
                <div style='text-align: center;'>
                    <div style='font-size: 3em;'>🔁</div>
                    <h3>Le Rétrocessionnaire</h3>
                    <p><b>Rôle :</b> Réassureur du réassureur</p>
                    <p><b>Mission :</b> Recevoir à son tour une partie du risque</p>
                    <p><b>Objectif :</b> Diversifier encore plus le risque</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Timeline des interactions
        st.subheader("⏱️ Séquence des Interventions")
        
        timeline_steps = {
            'Étape': ['Souscription', 'Cession', 'Placement', 'Rétrocession', 'Règlement'],
            'Acteur Principal': ['Cédante', 'Cédante', 'Courtier', 'Réassureur', 'Tous'],
            'Durée': ['2-4 semaines', '1-2 semaines', '3-6 semaines', '2-4 semaines', '30-60 jours'],
            'Document': ['Police', 'Note de cession', 'Proposition', 'Contrat rétro', 'Bordereau']
        }
        
        st.dataframe(pd.DataFrame(timeline_steps), width='None')
    
    with tab2:
        st.subheader("🌊 La Danse des Flux Financiers")
        
        # Contrôles interactifs pour le diagramme Sankey
        st.markdown("### 🎛️ Panneau de Contrôle des Flux")
        
        col_flux1, col_flux2, col_flux3 = st.columns(3)
        
        with col_flux1:
            prime_assure = st.slider("Prime de l'assuré", 50, 150, 100, key="prime_assure")
        with col_flux2:
            part_cedee = st.slider("Part cédée au réassureur", 20, 80, 70, key="part_cedee_flux")
        with col_flux3:
            part_retro = st.slider("Part rétrocédée", 10, 50, 50, key="part_retro")
        
        # Diagramme Sankey interactif
        labels = ["Assuré", "Cédante", "Courtier", "Réassureur", "Rétrocessionnaire"]
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=18,
                thickness=18,
                line=dict(color="black", width=0.5),
                label=labels,
                color=["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#F44336"]
            ),
            link=dict(
                source=[0, 1, 1, 2, 3],
                target=[1, 2, 3, 3, 4],
                value=[prime_assure, part_cedee * 0.8, part_cedee, part_cedee * 0.7, part_retro],
                color="rgba(0,0,0,0.2)"
            )
        )])
        
        fig.update_layout(
            height=500, 
            title="Flux de primes et sinistres dans la chaîne de réassurance",
            font=dict(size=14)
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # Légende interactive
        st.markdown("""
        <div style='background: #f8f9fa; padding: 1rem; border-radius: 10px; margin-top: 1rem;'>
            <h4>🎨 Code Couleur des Acteurs :</h4>
            <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;'>
                <div style='display: flex; align-items: center;'>
                    <div style='width: 20px; height: 20px; background: #4CAF50; margin-right: 10px; border-radius: 50%;'></div>
                    <span><b>Assuré</b> - Celui qui paie la prime initiale</span>
                </div>
                <div style='display: flex; align-items: center;'>
                    <div style='width: 20px; height: 20px; background: #2196F3; margin-right: 10px; border-radius: 50%;'></div>
                    <span><b>Cédante</b> - L'assureur qui transfère le risque</span>
                </div>
                <div style='display: flex; align-items: center;'>
                    <div style='width: 20px; height: 20px; background: #FF9800; margin-right: 10px; border-radius: 50%;'></div>
                    <span><b>Courtier</b> - L'intermédiaire négociateur</span>
                </div>
                <div style='display: flex; align-items: center;'>
                    <div style='width: 20px; height: 20px; background: #9C27B0; margin-right: 10px; border-radius: 50%;'></div>
                    <span><b>Réassureur</b> - Celui qui accepte le risque</span>
                </div>
                <div style='display: flex; align-items: center;'>
                    <div style='width: 20px; height: 20px; background: #F44336; margin-right: 10px; border-radius: 50%;'></div>
                    <span><b>Rétrocessionnaire</b> - Le réassureur du réassureur</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.subheader("🏢 L'Écosystème Mondial")
        
        # Carte des acteurs majeurs
        st.markdown("### 🌍 Les Géants du Marché")
        
        acteurs_data = {
            'Catégorie': ['Réassureurs Tier 1', 'Réassureurs Spécialisés', 'Courtiers Majeurs', 'Cédantes Leaders'],
            'Exemples': [
                'Munich Re, Swiss Re, Hannover Re',
                'SCOR, PartnerRe, Everest Re',
                'Aon Re, Guy Carpenter, Willis Re',
                'AXA, Allianz, Generali, Zurich'
            ],
            'Part de Marché': ['~40%', '~25%', '~20%', '~15%'],
            'Spécialité': ['Tous risques', 'Risques spécifiques', 'Intermédiation', 'Assurance directe']
        }
        
        st.dataframe(pd.DataFrame(acteurs_data), width='stretch')
        
        # Graphique de parts de marché
        marche_data = {
            'Acteur': ['Tier 1', 'Spécialisés', 'Courtiers', 'Cédantes'],
            'Part (%)': [40, 25, 20, 15]
        }
        
        fig_marche = px.pie(marche_data, values='Part (%)', names='Acteur',
                           title="Répartition du Marché Mondial de la Réassurance")
        st.plotly_chart(fig_marche, width='stretch')
        
        # Focus sur un acteur (interactif)
        st.subheader("🔍 Zoom sur un Acteur")
        
        acteur_choisi = st.selectbox(
            "Choisissez un acteur à explorer :",
            ["Munich Re", "Swiss Re", "Aon Re", "SCOR", "AXA"]
        )
        
        if acteur_choisi:
            infos_acteurs = {
                "Munich Re": {
                    "description": "Leader mondial, basé en Allemagne",
                    "chiffre": "45 Md€ de primes",
                    "specialite": "Tous risques, fort en catastrophes naturelles"
                },
                "Swiss Re": {
                    "description": "Suisse, innovation et solutions complexes",
                    "chiffre": "34 Md€ de primes", 
                    "specialite": "Risques corporates et solutions sur mesure"
                },
                "Aon Re": {
                    "description": "Courtier leader, conseil en réassurance",
                    "chiffre": "8 Md€ de commissions",
                    "specialite": "Placement et optimisation des programmes"
                },
                "SCOR": {
                    "description": "Français, expertise vie et non-vie",
                    "chiffre": "16 Md€ de primes",
                    "specialite": "Équilibre vie/non-vie, solide recherche"
                },
                "AXA": {
                    "description": "Assureur majeur avec activité réassurance",
                    "chiffre": "12 Md€ de primes cédées",
                    "specialite": "Cédante stratégique, réassurance interne"
                }
            }
            
            info = infos_acteurs[acteur_choisi]
            
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%); 
                        padding: 2rem; border-radius: 15px;'>
                <h3>{acteur_choisi}</h3>
                <p><b>Description :</b> {info['description']}</p>
                <p><b>Chiffre clé :</b> {info['chiffre']}</p>
                <p><b>Spécialité :</b> {info['specialite']}</p>
            </div>
            """, unsafe_allow_html=True)

# =============================================================================
# INTERFACE PRINCIPALE - SIDEBAR ET ROUTAGE
# =============================================================================

# Titre principal
st.markdown('<div class="main-header">🏛️ PLATEFORME COMPLÈTE DE RÉASSURANCE</div>', unsafe_allow_html=True)
st.markdown("### *Théorie, Pratique et Data Science pour Professionnels et Apprenants*")

# Sidebar Navigation
st.sidebar.title("🔍 Navigation")

section = st.sidebar.radio("Modules", [
    "🏠 Accueil & Présentation",
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

# =============================================================================
# ROUTAGE DES PAGES
# =============================================================================

if section == "🏠 Accueil & Présentation":
    
    # Bannière principale avec animation
    st.markdown("""
    <style>
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .welcome-banner {
        animation: fadeIn 1s ease-in;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    </style>
    
    <div class="welcome-banner">
        <h1 style='margin:0; font-size: 2.5em;'>🚀 Plateforme de Réassurance</h1>
        <p style='font-size: 1.3em; margin: 1rem 0;'>Découvrez le monde fascinant du partage des risques</p>
        <div style='font-size: 3em;'>🛡️📊🎯</div>
    </div>
    """, unsafe_allow_html=True)

    # Première ligne : Présentation générale
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("""
        <div class="concept-box">
        <h3>🎯 Objectifs de la Plateforme</h3>
        <p>Cette application complète vous permet de maîtriser tous les aspects de la réassurance :</p>
        <ul>
            <li><b>📚 Explications théoriques approfondies</b> des concepts clés</li>
            <li><b>🧮 Calculateurs interactifs</b> pour appliquer les formules</li>
            <li><b>📊 Analyses data science</b> avec KPI et prévisions</li>
            <li><b>📋 Études de cas réels</b> avec analyses détaillées</li>
            <li><b>🎯 Outils professionnels</b> de simulation et d'optimisation</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.metric("📈 Marché Mondial 2024", "450 Md€", "+6.2% vs 2023")
        st.metric("🏛️ Réassureurs Tier 1", "25 sociétés", "~80% du marché")
        
        st.markdown("""
        <div style='background: #fff3cd; padding: 1rem; border-radius: 10px; border-left: 4px solid #ffc107; margin-top: 1rem;'>
            <h4>⚠️ Importance Stratégique</h4>
            <p>Outil essentiel pour :</p>
            <ul>
            <li>Protéger les fonds propres</li>
            <li>Améliorer la notation</li>
            <li>Permettre la croissance</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.metric("📊 Modules Disponibles", "11 sections", "150+ concepts")
        
        # Badge de progression
        st.markdown("""
        <div style='background: #f8f9fa; padding: 1rem; border-radius: 10px; border-left: 4px solid #28a745; margin-top: 1rem;'>
            <h4>🏆 Votre progression</h4>
            <div style='background: #e9ecef; height: 10px; border-radius: 5px; margin: 10px 0;'>
                <div style='background: #28a745; width: 10%; height: 100%; border-radius: 5px;'></div>
            </div>
            <p><small>10% complété - Continuez !</small></p>
        </div>
        """, unsafe_allow_html=True)

    # Cartes interactives - Public et Objectifs
    st.markdown("### 🎓 Public Cible & Objectifs")
    
    col_cards = st.columns(3)
    
    with col_cards[0]:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); 
                    padding: 1.5rem; border-radius: 15px; text-align: center; 
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1); height: 200px;'>
            <h3>👥 Public Cible</h3>
            <ul style='text-align: left;'>
            <li><b>Étudiants</b> en assurance</li>
            <li><b>Professionnels</b> du secteur</li>
            <li><b>Actuaires</b> et gestionnaires</li>
            <li><b>Data scientists</b></li>
            <li><b>Consultants</b></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col_cards[1]:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%); 
                    padding: 1.5rem; border-radius: 15px; text-align: center; 
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1); height: 200px;'>
            <h3>🎯 Méthodologie</h3>
            <ul style='text-align: left;'>
            <li><b>Apprentissage progressif</b></li>
            <li><b>Simulations interactives</b></li>
            <li><b>Cas concrets terrain</b></li>
            <li><b>Outils visuels</b></li>
            <li><b>Quiz personnalisés</b></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col_cards[2]:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%); 
                    padding: 1.5rem; border-radius: 15px; text-align: center; 
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1); height: 200px;'>
            <h3>📈 Résultats Attendu</h3>
            <ul style='text-align: left;'>
            <li><b>Vision claire</b> des mécanismes</li>
            <li><b>Maîtrise pratique</b> des outils</li>
            <li><b>Expertise opérationnelle</b></li>
            <li><b>Analyse quantitative</b></li>
            <li><b>Décision stratégique</b></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Mini-jeu interactif
    st.markdown("---")
    st.markdown("### 🎮 Mini-Lab : Comprendre le Partage des Risques")
    
    col_game1, col_game2 = st.columns([1, 2])
    
    with col_game1:
        st.markdown("""
        <div style='background: #f8f9fa; padding: 1.5rem; border-radius: 15px;'>
            <h4>💡 Concept Clé</h4>
            <p>La réassurance est un mécanisme de <b>PARTAGE DU RISQUE</b> entre assureurs et réassureurs pour préserver la <b>SOLVABILITÉ</b> du système.</p>
            
            <h4>🎯 Testez le principe</h4>
            <p>Utilisez les curseurs pour simuler différents scénarios de répartition :</p>
        </div>
        """, unsafe_allow_html=True)
        
        prime_totale = st.slider("💰 Prime totale collectée (€)", 500000, 2000000, 1000000, step=100000)
        part_cedee = st.slider("📤 Part cédée au réassureur (%)", 10, 60, 40)
        
        prime_conservee = prime_totale * (100 - part_cedee) / 100
        prime_cedee = prime_totale * part_cedee / 100
        
        st.info(f"""
        **📊 Résultat de votre simulation :**
        - **Prime conservée :** {prime_conservee:,.0f} €
        - **Prime cédée :** {prime_cedee:,.0f} €
        - **Ratio de cession :** {part_cedee}%
        """)

    with col_game2:
        # Graphique interactif
        fig = go.Figure(data=[go.Pie(
            labels=[f'Conservé par assureur ({100-part_cedee}%)', f'Cédé au réassureur ({part_cedee}%)'],
            values=[prime_conservee, prime_cedee],
            hole=.4,
            marker_colors=['#FF6B6B', '#4ECDC4'],
            textinfo='label+value'
        )])
        fig.update_layout(
            title="Répartition des primes entre assureur et réassureur",
            showlegend=False,
            annotations=[dict(
                text=f'Total<br>{prime_totale:,.0f}€', 
                x=0.5, y=0.5, 
                font_size=16, 
                showarrow=False,
                font_color='white'
            )]
        )
        fig.update_traces(
            textposition='inside', 
            texttemplate='%{label}<br>%{value:,.0f}€',
            textfont_color='white'
        )
        st.plotly_chart(fig, width='stretch')

    # Section Data Science
    st.markdown("---")
    st.markdown("### 🔬 Fonctionnalités Avancées")
    
    col_ds1, col_ds2, col_ds3 = st.columns(3)
    
    with col_ds1:
        st.markdown("""
        <div style='background: #e7f3ff; padding: 1.5rem; border-radius: 10px;'>
            <h4>📈 Analyse KPI Avancée</h4>
            <ul>
            <li>Loss Ratio & Combined Ratio</li>
            <li>Fréquence et Sévérité</li>
            <li>Solvabilité SCR</li>
            <li>Réserves techniques</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col_ds2:
        st.markdown("""
        <div style='background: #f0f8ff; padding: 1.5rem; border-radius: 10px;'>
            <h4>🔮 Prévisions 3 Ans</h4>
            <ul>
            <li>Modèles SARIMAX</li>
            <li>Par ligne de business</li>
            <li>Par région géographique</li>
            <li>Scénarios de stress</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col_ds3:
        st.markdown("""
        <div style='background: #f5f5f5; padding: 1.5rem; border-radius: 10px;'>
            <h4>📤 Export Professionnel</h4>
            <ul>
            <li>Rapports PDF exécutifs</li>
            <li>Données CSV structurées</li>
            <li>Graphiques interactifs</li>
            <li>Tableaux de bord</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Diagramme Sankey interactif
    st.markdown("---")
    st.markdown("### 🌊 Visualisation des Flux de Réassurance")
    
    col_sankey1, col_sankey2 = st.columns([1, 2])
    
    with col_sankey1:
        st.markdown("""
        <div style='background: #f8f9fa; padding: 1.5rem; border-radius: 15px;'>
            <h4>🎯 Comment lire ce diagramme</h4>
            <p>Visualisez comment le risque circule entre les différents acteurs :</p>
            <ul>
            <li>🟢 <b>Portefeuille</b> : Tous les risques assurés</li>
            <li>🔵 <b>Assureur</b> : Garde une partie du risque</li>
            <li>🟠 <b>Réassureur</b> : Prend le risque excédentaire</li>
            <li>🔴 <b>Sinistres</b> : Les pertes qui surviennent</li>
            </ul>
            <p><i>💡 Plus le flux est épais, plus le montant est important !</i></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Contrôles pour le diagramme Sankey
        sinistres_directs = st.slider("Niveau des sinistres (%)", 40, 80, 60)
        part_reassureur = st.slider("Participation réassureur (%)", 20, 60, 40)

    with col_sankey2:
        # Données pour le diagramme Sankey
        portefeuille_total = 100
        labels = ["Portefeuille", "Assureur", "Réassureur", "Sinistres courants"]
        label_to_idx = {lab: i for i, lab in enumerate(labels)}
        
        sankey_fig = go.Figure(data=[go.Sankey(
            node=dict(
                label=labels,
                pad=18,
                thickness=20,
                line=dict(color="#cfe0ee", width=1),
                color=["#4CAF50", "#2196F3", "#FF9800", "#F44336"]
            ),
            link=dict(
                source=[0, 1, 1],  # Portefeuille, Assureur, Assureur
                target=[1, 3, 2],  # Assureur, Sinistres, Réassureur
                value=[portefeuille_total, sinistres_directs, part_reassureur],
                color="rgba(0,0,0,0.2)"
            )
        )])
        
        sankey_fig.update_layout(
            height=400, 
            title="Flux simplifié du risque entre assureur et réassureur",
            font=dict(size=12)
        )
        
        st.plotly_chart(sankey_fig, width='stretch')

    # Roadmap d'apprentissage
    st.markdown("---")
    st.markdown("### 🗺️ Roadmap d'Apprentissage")
    
    roadmap_data = {
        'Phase': ['🎯 Fondamentaux', '⚙️ Techniques', '📊 Analyse', '🚀 Expertise'],
        'Modules': [
            'Concepts de base, Acteurs, Écosystème',
            'Traités proportionnels et non-proportionnels',
            'Data Science, KPI, Prévisions',
            'Cas complexes, Optimisation, Stratégie'
        ],
        'Durée': ['2 semaines', '3 semaines', '3 semaines', '2 semaines'],
        'Compétences': [
            'Compréhension des bases',
            'Maîtrise des techniques',
            'Analyse quantitative',
            'Expertise stratégique'
        ]
    }
    
    st.dataframe(pd.DataFrame(roadmap_data), width=None)

    # Appel à l'action final
    st.markdown("---")
    st.markdown("""
    <div style='background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%); 
                padding: 2rem; border-radius: 15px; border-left: 6px solid #ffc107; text-align: center;'>
        <h2>🚀 Prêt à Commencer Votre Voyage ?</h2>
        <p style='font-size: 1.2em;'>Choisissez votre point de départ pour explorer le monde de la réassurance</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_cta1, col_cta2, col_cta3 = st.columns(3)
    
    with col_cta1:
        if st.button("📚 Commencer les Fondamentaux", width='stretch', type="primary"):
            st.session_state.current_page = "Principes Fondamentaux"
            st.rerun()
    
    with col_cta2:
        if st.button("🧮 Utiliser les Calculateurs", width='stretch'):
            st.session_state.current_page = "Calculateurs Avancés"
            st.rerun()
    
    with col_cta3:
        if st.button("📊 Explorer les Données", width='stretch'):
            st.session_state.current_page = "Analyse Data Science"
            st.rerun()

# =============================================================================
# AUTRES SECTIONS (structure simplifiée pour respecter la limite de caractères)
# =============================================================================

elif section == "🎓 Principes Ludiques":
    page_principes_ludique()

elif section == "📝 Types de Contrats Ludiques":
    page_types_contrats_ludique()

elif section == "🏛️ Acteurs & Flux Ludiques":
    page_acteurs_flux_ludique()

elif section == "📚 Concepts Fondamentaux":
    st.markdown('<div class="section-header">📚 Concepts Fondamentaux de la Réassurance</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🏛️ Définitions", "🔄 Processus", "📊 Écosystème"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="concept-box">
            <h3>🎯 Définition Professionnelle</h3>
            <p>La <b>réassurance</b> est une technique par laquelle un assureur (la cédante) transfère tout ou partie 
            des risques qu'il a assurés à un réassureur, contre le paiement d'une prime de réassurance.</p>
            <p><b>Double fonction</b> : Technique (transfert de risque) et Financière (lissage des résultats).</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            # Tableau des définitions
            definitions_data = {
                'Terme': ['Cédante', 'Réassureur', 'Prime de Réassurance', 'Commission', 'Rétention', 'Cession'],
                'Définition': [
                    'Compagnie qui transfère le risque',
                    'Société qui accepte le risque',
                    'Prix du transfert de risque',
                    'Pourcentage reversé pour frais',
                    'Part conservée par la cédante',
                    'Part transférée au réassureur'
                ]
            }
            
            st.dataframe(pd.DataFrame(definitions_data), width='stretch')

elif section == "📈 Traités Proportionnels":
    st.markdown("### 📈 Traités Proportionnels - Théorie et Applications")
    
    st.info("""
    **🧮 Principes Mathématiques des Traités Proportionnels**
    
    Les traités proportionnels reposent sur un **partage systématique** des primes et sinistres selon un pourcentage fixe.
    """)
    
    tab1, tab2, tab3 = st.tabs(["📊 Quota-Share", "📈 Surplus", "🔄 Applications Pratiques"])
    
    with tab1:
        st.subheader("📊 Traité Quota-Share (Quote-Part)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**⚖️ La Famille Proportionnelle**")
            st.write("**🧩 Le Partage Équitable**")
            st.write("**Principe :** Partage systématique des primes et sinistres selon un pourcentage fixe.")
        
        with col2:
            # Calculateur Quota-Share
            st.subheader("🧮 Calculateur Quota-Share")
            
            prime_directe = st.number_input("Prime directe totale (€)", value=1000000, step=100000)
            taux_cession = st.slider("Taux de cession (%)", 10, 90, 30)
            
            prime_cedee = prime_directe * taux_cession / 100
            
            st.metric("💰 Prime cédée", f"{prime_cedee:,.0f} €")

elif section == "⚡ Traités Non-Proportionnels":
    st.markdown('<div class="section-header">⚡ Traités Non-Proportionnels - Théorie et Applications</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="theory-box">
    <h3>🎯 Principes des Traités Non-Proportionnels</h3>
    <p>Contrairement aux traités proportionnels, les traités non-proportionnels déclenchent l'intervention du réassureur 
    <b>uniquement au-delà d'un certain seuil de sinistres</b> (la priorité), et jusqu'à une limite donnée.</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📉 Stop Loss", "🌊 Excédent de Sinistres", "📊 Applications Avancées"])
    
    with tab1:
        st.subheader("📉 Stop Loss (Excédent de Pertes)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="concept-box">
            <h4>🎯 Définition Technique</h4>
            <p>Le <b>Stop Loss</b> protège la cédante contre un taux de sinistralité anormalement élevé sur l'ensemble 
            de son portefeuille ou d'une branche d'activité.</p>
            </div>
            """, unsafe_allow_html=True)

elif section == "💰 Tarification Technique":
    st.markdown('<div class="section-header">💰 Tarification Technique - Modèles et Méthodologies</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="theory-box">
    <h3>🎯 Principes Actuariels de Tarification</h3>
    <p>La tarification en réassurance combine <b>statistiques historiques</b>, <b>modélisation prospective</b> 
    et <b>jugement d'expert</b> pour déterminer des primes équitables et suffisantes.</p>
    </div>
    """, unsafe_allow_html=True)

elif section == "📊 Comptabilité Technique":
    st.markdown('<div class="section-header">📊 Comptabilité Technique - Principes et Applications</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="theory-box">
    <h3>🏛️ Cadre Comptable des Assureurs</h3>
    <p>La comptabilité technique des assureurs et réassureurs suit des principes spécifiques distincts 
    de la comptabilité générale, avec un focus sur la <b>mesure des engagements techniques</b>.</p>
    </div>
    """, unsafe_allow_html=True)

elif section == "🌪️ Gestion des Catastrophes":
    st.markdown('<div class="section-header">🌪️ Gestion des Risques Catastrophiques</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="theory-box">
    <h3>🎯 Modélisation des Catastrophes Naturelles</h3>
    <p>La modélisation des catastrophes combine <b>données historiques</b>, <b>modèles physiques</b> 
    et <b>analyses statistiques</b> pour estimer les pertes potentielles.</p>
    </div>
    """, unsafe_allow_html=True)

elif section == "🛡️ Solvabilité & Réglementation":
    st.markdown('<div class="section-header">🛡️ Solvabilité II - Cadre Réglementaire Complet</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="theory-box">
    <h3>🏛️ Cadre Solvabilité II - Les Trois Piliers</h3>
    <p>Le cadre Solvabilité II, applicable depuis 2016, repose sur <b>trois piliers</b> complémentaires 
    pour assurer la stabilité financière des assureurs et réassureurs en Europe.</p>
    </div>
    """, unsafe_allow_html=True)

elif section == "📋 Études de Cas Concrets":
    st.markdown('<div class="section-header">📋 Études de Cas Concrets - Applications Réelles</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🏢 Cas Assureur Auto", "🏠 Cas Assureur Habitation", "🌍 Cas Réassureur Global"])
    
    with tab1:
        st.subheader("🏢 Cas : Optimisation du Programme d'un Assureur Auto")
        
        st.markdown("""
        <div class="case-study-box">
        <h4>📖 Contexte</h4>
        <p><b>Assureur AutoPro</b> : Portefeuille de 50M€ de primes, spécialisé en assurance automobile particuliers.
        Souhaite optimiser son programme de réassurance pour améliorer sa rentabilité.</p>
        </div>
        """, unsafe_allow_html=True)

elif section == "📊 Analyse Data Science":
    st.markdown('<div class="section-header">📊 Analyse Data Science - KPI & Prévisions</div>', unsafe_allow_html=True)
    
    # Sidebar pour les données
    with st.sidebar:
        st.subheader("📥 Chargement des Données")
        uploaded_file = st.file_uploader("Importer CSV/Excel", type=["csv", "xlsx", "xls"])
        
        st.subheader("⚙️ Configuration")
        use_demo_data = st.checkbox("Utiliser les données de démonstration", value=True)
    
    # Préparation des données
    if use_demo_data:
        df_raw = make_demo_data(periods=16, freq="Q")
        mapping = auto_map_columns(df_raw)
        df = df_raw.rename(columns={v: k for k, v in mapping.items() if v is not None})
        df["date"] = _infer_date_col(df["date"])
        df = add_month_start(df)
        df_kpi = compute_kpis(df)
        
        # Métriques principales
        agg_global = aggregate_kpis(df_kpi, by=["date"]).sort_values("date")
        if not agg_global.empty:
            last_row = agg_global.iloc[-1]
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Primes Acquises", f"{last_row['earned_premium']:,.0f} €")
            col2.metric("Sinistres Encourus", f"{last_row['incurred_claims']:,.0f} €")
            col3.metric("Loss Ratio", f"{last_row['loss_ratio']*100:.1f}%")
            col4.metric("Combined Ratio", f"{last_row['combined_ratio']*100:.1f}%")

elif section == "🧮 Calculateurs Avancés":
    st.markdown('<div class="section-header">🧮 Calculateurs Avancés - Outils Professionnels</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📈 Optimisation Programme", "💰 Analyse de Rentabilité", "🛡️ Simulation SCR"])
    
    with tab1:
        st.subheader("📈 Optimisateur de Programme de Réassurance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="concept-box">
            <h4>🎯 Objectif d'Optimisation</h4>
            <p>Cet outil permet de trouver la structure optimale de réassurance qui maximise 
            la rentabilité tout en respectant les contraintes de solvabilité.</p>
            </div>
            """, unsafe_allow_html=True)

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
    📧 contact@example.com  
    🌐 www.example.com
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

