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
    page_title="Plateforme Complète de Réassurance - Théorie & Data Science",
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
# INTERFACE PRINCIPALE
# =============================================================================

# Titre principal
st.markdown('<div class="main-header">🏛️ PLATEFORME COMPLÈTE DE RÉASSURANCE</div>', unsafe_allow_html=True)
st.markdown("### *Théorie, Pratique et Data Science pour Professionnels et Apprenants*")

# Sidebar Navigation
st.sidebar.title("🔍 Navigation")

section = st.sidebar.radio("Modules", [
    "🏠 Accueil & Présentation",
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
# SECTION 1: ACCUEIL & PRÉSENTATION
# =============================================================================
if section == "🏠 Accueil & Présentation":
    st.markdown('<div class="section-header">🎯 Bienvenue sur la Plateforme de Réassurance</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="concept-box">
        <h3>📖 Objectifs de la Plateforme</h3>
        <p>Cette application complète vous permet de maîtriser tous les aspects de la réassurance à travers :</p>
        <ul>
            <li><b>📚 Explications théoriques approfondies</b> des concepts clés</li>
            <li><b>🧮 Calculateurs interactifs</b> pour appliquer les formules</li>
            <li><b>📊 Analyses data science</b> avec KPI et prévisions</li>
            <li><b>📋 Études de cas réels</b> avec analyses détaillées</li>
            <li><b>🎯 Outils professionnels</b> de simulation et d'optimisation</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="theory-box">
        <h3>🎓 Public Cible</h3>
        <ul>
            <li><b>Étudiants en assurance et réassurance</b></li>
            <li><b>Professionnels du secteur</b> souhant se perfectionner</li>
            <li><b>Actuaires</b> et <b>gestionnaires de risques</b></li>
            <li><b>Data scientists</b> en secteur assurance</li>
            <li><b>Consultants</b> en finance et assurance</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric("📈 Marché Mondial 2024", "450 Md€", "+6.2% vs 2023")
        st.metric("🏛️ Réassureurs Tier 1", "25 sociétés", "~80% du marché")
        st.metric("📊 Modules Disponibles", "11 sections", "150+ concepts")
        
        st.markdown("""
        <div class="warning-box">
        <h4>⚠️ Importance Stratégique</h4>
        <p>La réassurance est un <b>outil de gestion du capital</b> essentiel pour :</p>
        <ul>
        <li>Protéger les fonds propres</li>
        <li>Améliorer la notation financière</li>
        <li>Permettre la croissance</li>
        <li>Gérer la volatilité des résultats</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Nouveautés Data Science
    st.markdown("### 🔬 Nouveautés Data Science")
    
    col_ds1, col_ds2, col_ds3 = st.columns(3)
    
    with col_ds1:
        st.markdown("""
        <div class="data-box">
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
        <div class="data-box">
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
        <div class="data-box">
        <h4>📤 Export Professionnel</h4>
        <ul>
        <li>Rapports PDF exécutifs</li>
        <li>Données CSV structurées</li>
        <li>Graphiques interactifs</li>
        <li>Tableaux de bord</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    # Roadmap d'apprentissage
    st.markdown("### 🗺️ Roadmap d'Apprentissage")
    
    roadmap_data = {
        'Phase': ['Fondamentaux', 'Techniques', 'Analyse', 'Expertise'],
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
    
    st.dataframe(pd.DataFrame(roadmap_data), use_container_width=True)

# =============================================================================
# SECTION 2: CONCEPTS FONDAMENTAUX
# =============================================================================
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
            
            st.markdown("""
            <div class="theory-box">
            <h3>🧠 Théorie : Principe de Mutualisation</h3>
            <p>La réassurance s'appuie sur la <b>loi des grands nombres</b> :</p>
            <div class="formula-box">
            σ_portefeuille = σ_risque / √n
            </div>
            <p>Où σ représente la volatilité et n le nombre de risques. En mutualisant, le réassureur réduit la variabilité des résultats.</p>
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
                ],
                'Impact Comptable': [
                    'Compte 62 - Acceptations',
                    'Compte 61 - Cessions',
                    'Charge de réassurance',
                    'Produit de réassurance',
                    'Actif du bilan',
                    'Passif du bilan'
                ]
            }
            
            st.dataframe(pd.DataFrame(definitions_data), use_container_width=True)
            
            st.markdown("""
            <div class="warning-box">
            <h4>📈 Types de Réassurance</h4>
            <ul>
            <li><b>Facultative</b> : Risque par risque</li>
            <li><b>Traditionnelle</b> : Contrat global</li>
            <li><b>Proportionnelle</b> : Partage des primes et sinistres</li>
            <li><b>Non-proportionnelle</b> : Au-delà d'un seuil</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("""
        <div class="concept-box">
        <h3>🔄 Processus de Réassurance</h3>
        <p>Le cycle complet de la réassurance comprend 5 étapes principales :</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Processus détaillé
        processus_data = {
            'Étape': ['1. Analyse du Risque', '2. Structuration', '3. Négociation', '4. Gestion', '5. Règlement'],
            'Activités': [
                'Évaluation technique du portefeuille',
                'Définition des traités et couvertures',
                'Détermination des primes et commissions',
                'Suivi et administration des traités',
                'Règlement des sinistres et commissions'
            ],
            'Documents': [
                'Notes techniques, Scorings',
                'Projet de traité, Conditions',
                'Placements, Contrats',
                'Systèmes de gestion, Reporting',
                'Bordereaux, Contrôles'
            ],
            'Délai': [
                '2-4 semaines',
                '1-2 semaines',
                '3-6 semaines',
                'Continue',
                '30-60 jours'
            ]
        }
        
        st.dataframe(pd.DataFrame(processus_data), use_container_width=True)
        
        # Schéma du processus
        st.markdown("### 📊 Schéma du Flux de Réassurance")
        
        fig_process = go.Figure(go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=["Assuré", "Cédante", "Réassureur Direct", "Rétrocessionnaire", "Marché"]
            ),
            link=dict(
                source=[0, 1, 1, 2, 2, 3],
                target=[1, 2, 3, 3, 4, 4],
                value=[100, 70, 30, 20, 50, 20],
                label=["Prime", "Cession", "Rétrocession", "Sinistre", "Commission"]
            )
        ))
        
        fig_process.update_layout(title_text="Flux des Risques et des Primes", font_size=10)
        st.plotly_chart(fig_process, use_container_width=True)
    
    with tab3:
        st.markdown("""
        <div class="concept-box">
        <h3>🏢 Écosystème de la Réassurance</h3>
        <p>Le marché de la réassurance est structuré en plusieurs niveaux d'acteurs :</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
            <h4>🎯 Réassureurs Directs</h4>
            <ul>
            <li>Munich Re</li>
            <li>Swiss Re</li>
            <li>Hannover Re</li>
            <li>SCOR</li>
            </ul>
            <p><b>Rôle</b> : Acceptation directe des risques</p>
            <p><b>Part de marché</b> : ~65%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
            <h4>🔄 Réassureurs Rétrocessionnaires</h4>
            <ul>
            <li>Lloyd's</li>
            <li>Berkshire Hathaway</li>
            <li>PartnerRe</li>
            </ul>
            <p><b>Rôle</b> : Réassurance des réassureurs</p>
            <p><b>Part de marché</b> : ~20%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
            <h4>📊 Courtiers en Réassurance</h4>
            <ul>
            <li>Aon Re</li>
            <li>Guy Carpenter</li>
            <li>Willis Re</li>
            </ul>
            <p><b>Rôle</b> : Intermédiation et conseil</p>
            <p><b>Part de marché</b> : ~15%</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Statistiques du marché
        st.markdown("### 📈 Chiffres Clés du Marché 2024")
        
        marche_data = {
            'Segment': ['Vie', 'Non-Vie', 'Catastrophes', 'Santé', 'Transport'],
            'Prime Globale (Md€)': [180, 220, 35, 12, 8],
            'Croissance (%)': [4.2, 6.8, 12.5, 5.3, 3.7],
            'Profitabilité (%)': [8.2, 6.5, 15.3, 7.8, 4.2]
        }
        
        st.dataframe(pd.DataFrame(marche_data), use_container_width=True)

# =============================================================================
# SECTION 3: TRAITÉS PROPORTIONNELS
# =============================================================================
elif section == "📈 Traités Proportionnels":
    st.markdown('<div class="section-header">📈 Traités Proportionnels - Théorie et Applications</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="theory-box">
    <h3>🧮 Principes Mathématiques des Traités Proportionnels</h3>
    <p>Les traités proportionnels reposent sur un <b>partage systématique</b> des primes et sinistres selon un pourcentage fixe.</p>
    
    <div class="formula-box">
    <b>Formules fondamentales :</b><br>
    Prime cédée = Prime directe × Taux de cession<br>
    Sinistre cédé = Sinistre direct × Taux de cession<br>
    Commission = Prime cédée × Taux de commission
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📊 Quota-Share", "📈 Surplus", "🔄 Applications Pratiques"])
    
    with tab1:
        st.subheader("📊 Traité Quota-Share (Quote-Part)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="concept-box">
            <h4>🎯 Définition Technique</h4>
            <p>Le <b>Quota-Share</b> est un traité par lequel la cédante cède une fraction fixe de tous les risques 
            d'une catégorie déterminée, et le réassureur en accepte la même fraction.</p>
            
            <h4>📝 Caractéristiques</h4>
            <ul>
            <li>Taux de cession unique et constant</li>
            <li>Application à l'ensemble du portefeuille</li>
            <li>Partage systématique des primes et sinistres</li>
            <li>Commission de réassurance généralement élevée</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="warning-box">
            <h4>⚠️ Avantages et Inconvénients</h4>
            <p><b>Avantages :</b> Simplicité, lissage efficace, réduction du besoin en capital</p>
            <p><b>Inconvénients :</b> Cession même des bons risques, coût pour les petits sinistres</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Calculateur Quota-Share
            st.subheader("🧮 Calculateur Quota-Share")
            
            prime_directe = st.number_input("Prime directe totale (€)", value=1000000, step=100000)
            sinistre_attendu = st.number_input("Sinistre attendu (€)", value=600000, step=50000)
            taux_cession = st.slider("Taux de cession (%)", 10, 90, 30)
            taux_commission = st.slider("Taux de commission (%)", 15, 40, 25)
            
            # Calculs détaillés
            prime_cedee = prime_directe * taux_cession / 100
            sinistre_cede = sinistre_attendu * taux_cession / 100
            commission = prime_cedee * taux_commission / 100
            prime_nette_reassureur = prime_cedee - commission
            
            # Affichage des résultats
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.metric("💰 Prime cédée", f"{prime_cedee:,.0f} €")
                st.metric("⚡ Sinistre cédé", f"{sinistre_cede:,.0f} €")
            with col_res2:
                st.metric("💸 Commission", f"{commission:,.0f} €")
                st.metric("📊 Prime nette réassureur", f"{prime_nette_reassureur:,.0f} €")
            
            # Graphique de répartition
            labels = ['Cédé au réassureur', 'Commission', 'Conservé par cédante']
            values = [prime_nette_reassureur, commission, prime_directe - prime_cedee]
            
            fig = px.pie(values=values, names=labels, 
                        title="Répartition de la Prime Directe",
                        color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig, use_container_width=True)
            
            # Analyse de rentabilité
            benefice_cedeant = (prime_directe - prime_cedee) - (sinistre_attendu - sinistre_cede) + commission
            taux_rentabilite = (benefice_cedeant / prime_directe) * 100
            
            st.metric("📈 Bénéfice net cédante", f"{benefice_cedeant:,.0f} €")
            st.metric("🎯 Taux de rentabilité", f"{taux_rentabilite:.1f}%")
    
    with tab2:
        st.subheader("📈 Traité de Surplus")
        
        st.markdown("""
        <div class="theory-box">
        <h3>🎯 Principe du Surplus</h3>
        <p>Le traité de <b>surplus</b> permet à la cédante de ne céder que la partie des risques qui dépasse sa rétention, 
        avec des lignes de surplus multiples pour les très gros risques.</p>
        
        <div class="formula-box">
        <b>Calcul du surplus :</b><br>
        Ligne = Rétention × Multiple<br>
        Cession = Min(Capital assuré - Rétention, Ligne disponible)<br>
        Taux de cession = Cession / Capital assuré
        </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Paramètres du surplus
            st.subheader("⚙️ Paramètres du Traité")
            
            retention = st.number_input("Rétention par risque (€)", value=500000, step=50000)
            multiple_ligne = st.slider("Multiple de la ligne", 2, 10, 4)
            nb_lignes = st.number_input("Nombre de lignes disponibles", value=5, min_value=1, max_value=20)
            
            capacite_surplus = retention * multiple_ligne * nb_lignes
            
            st.metric("📦 Capacité totale surplus", f"{capacite_surplus:,.0f} €")
            st.metric("🎯 Plus gros risque couvert", f"{retention + capacite_surplus:,.0f} €")
        
        with col2:
            # Simulation de risque
            st.subheader("🎲 Simulation de Risque")
            
            capital_assure = st.number_input("Capital assuré (€)", value=2500000, step=100000)
            
            if capital_assure <= retention:
                st.info("💰 Risque entièrement conservé - Pas de cession")
                part_cedee = 0
                taux_cession = 0
            else:
                part_cedee = min(capital_assure - retention, capacite_surplus)
                taux_cession = (part_cedee / capital_assure) * 100
                
                st.metric("📤 Part cédée en surplus", f"{part_cedee:,.0f} €")
                st.metric("📊 Taux de cession effectif", f"{taux_cession:.1f} %")
            
            # Tableau de répartition
            repartition_data = {
                'Élément': ['Rétention cédante', 'Surplus cédé', 'Total risque'],
                'Montant (€)': [min(retention, capital_assure), part_cedee, capital_assure],
                'Pourcentage': [
                    min(retention, capital_assure) / capital_assure * 100,
                    part_cedee / capital_assure * 100,
                    100
                ]
            }
            
            st.dataframe(pd.DataFrame(repartition_data))
            
            # Graphique de répartition
            if capital_assure > 0:
                fig_repartition = px.pie(
                    values=[min(retention, capital_assure), part_cedee],
                    names=['Rétention', 'Surplus cédé'],
                    title="Répartition du Risque"
                )
                st.plotly_chart(fig_repartition, use_container_width=True)
    
    with tab3:
        st.subheader("🔄 Applications Pratiques et Cas d'Usage")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="case-study-box">
            <h4>🏢 Cas d'Usage 1 : Début d'Activité</h4>
            <p><b>Contexte</b> : Nouvel assureur avec peu de fonds propres</p>
            <p><b>Solution</b> : Quota-share à 50% pour :</p>
            <ul>
            <li>Limiter l'engagement en capital</li>
            <li>Bénéficier de l'expertise du réassureur</li>
            <li>Construire un historique</li>
            </ul>
            <p><b>Résultat</b> : Croissance maîtrisée et rentabilité préservée</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="case-study-box">
            <h4>🌪️ Cas d'Usage 2 : Exposition Catastrophe</h4>
            <p><b>Contexte</b> : Assureur avec forte exposition aux catastrophes naturelles</p>
            <p><b>Solution</b> : Programme combiné Quota-Share + Surplus</p>
            <ul>
            <li>Quota-share pour le portefeuille standard</li>
            <li>Surplus pour les risques exceptionnels</li>
            <li>Couche catastrophe spécifique</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="theory-box">
            <h4>📊 Optimisation du Programme</h4>
            <p>Critères pour choisir entre Quota-Share et Surplus :</p>
            
            <div class="formula-box">
            <b>Matrice de décision :</b><br>
            Homogénéité du portefeuille → Quota-Share<br>
            Hétérogénéité des risques → Surplus<br>
            Besoin de liquidité → Quota-Share<br>
            Optimisation capital → Surplus
            </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Calculateur d'optimisation
            st.subheader("🎯 Calculateur d'Optimisation")
            
            taille_portefeuille = st.selectbox("Taille du portefeuille", 
                                             ["< 10M€", "10-50M€", "50-200M€", "> 200M€"])
            homogeneite = st.slider("Homogénéité des risques", 1, 10, 7)
            exposition_cat = st.slider("Exposition catastrophes", 1, 10, 3)
            expertise_interne = st.slider("Expertise technique interne", 1, 10, 5)
            
            # Logique de recommandation
            score_quota = homogeneite + expertise_interne
            score_surplus = (10 - homogeneite) + exposition_cat
            
            if score_quota > score_surplus:
                recommendation = "QUOTA-SHARE"
                ratio_optimal = "30-50%"
            else:
                recommendation = "SURPLUS"
                ratio_optimal = "Rétention adaptée aux risques"
            
            st.metric("🎯 Recommandation", recommendation)
            st.metric("📊 Ratio optimal", ratio_optimal)

# =============================================================================
# SECTION 4: TRAITÉS NON-PROPORTIONNELS
# =============================================================================
elif section == "⚡ Traités Non-Proportionnels":
    st.markdown('<div class="section-header">⚡ Traités Non-Proportionnels - Théorie et Applications</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="theory-box">
    <h3>🎯 Principes des Traités Non-Proportionnels</h3>
    <p>Contrairement aux traités proportionnels, les traités non-proportionnels déclenchent l'intervention du réassureur 
    <b>uniquement au-delà d'un certain seuil de sinistres</b> (la priorité), et jusqu'à une limite donnée.</p>
    
    <div class="formula-box">
    <b>Formule d'intervention :</b><br>
    Prise réassureur = Max(0, Min(Limite, Sinistres agrégés - Priorité))
    </div>
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
            
            <h4>📝 Caractéristiques Clés</h4>
            <ul>
            <li>Protège le <b>résultat technique</b></li>
            <li>Se déclenche sur sinistres <b>agrégés</b></li>
            <li>Priorité généralement exprimée en % des primes</li>
            <li>Coût élevé mais protection forte</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="warning-box">
            <h4>⚠️ Applications Typiques</h4>
            <p><b>Scénarios de déclenchement :</b></p>
            <ul>
            <li>Catastrophes naturelles multiples</li>
            <li>Dégradation soudaine de la sinistralité</li>
            <li>Événements systémiques</li>
            <li>Erreurs de tarification massives</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Calculateur Stop Loss
            st.subheader("🧮 Calculateur Stop Loss")
            
            primes_portefeuille = st.number_input("Primes du portefeuille (€)", value=5000000, step=100000)
            priorite_pourcentage = st.slider("Priorité (% des primes)", 100, 130, 110)
            limite_stoploss = st.number_input("Limite Stop Loss (€)", value=2000000, step=100000)
            sinistres_reels = st.number_input("Sinistres réels du portefeuille (€)", value=6200000, step=100000)
            
            # Calculs détaillés
            priorite_absolue = primes_portefeuille * priorite_pourcentage / 100
            taux_sinistralite = (sinistres_reels / primes_portefeuille) * 100
            
            if sinistres_reels <= priorite_absolue:
                prise_reassureur = 0
            else:
                prise_reassureur = min(limite_stoploss, sinistres_reels - priorite_absolue)
            
            sinistre_reste_cedeant = sinistres_reels - prise_reassureur
            
            # Affichage des résultats
            st.metric("📊 Taux de sinistralité", f"{taux_sinistralite:.1f}%")
            st.metric("⚡ Sinistre à charge réassureur", f"{prise_reassureur:,.0f} €")
            st.metric("💰 Sinistre net cédante", f"{sinistre_reste_cedeant:,.0f} €")
            
            # Graphique waterfall
            fig = go.Figure(go.Waterfall(
                name="Répartition Stop Loss",
                orientation="v",
                measure=["relative", "relative", "relative", "total"],
                x=["Primes", "Sinistres", "Priorité", "Part réassureur", "Résultat net"],
                textposition="outside",
                y=[primes_portefeuille, -sinistres_reels, priorite_absolue, prise_reassureur, -sinistre_reste_cedeant]
            ))
            fig.update_layout(title="Analyse Stop Loss - Répartition des Flux")
            st.plotly_chart(fig, use_container_width=True)
            
            # Analyse de la protection
            protection_obtenue = (prise_reassureur / sinistres_reels) * 100 if sinistres_reels > 0 else 0
            st.metric("🛡️ Niveau de protection", f"{protection_obtenue:.1f}%")
    
    with tab2:
        st.subheader("🌊 Traité XL (Excédent de Sinistre)")
        
        st.markdown("""
        <div class="theory-box">
        <h3>🏗️ Architecture en Couches XL</h3>
        <p>Les traités XL sont structurés en <b>couches successives</b>, chaque réassureur prenant une tranche de sinistre 
        entre une priorité et une limite. Cette structure permet une optimisation fine de la protection.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Simulateur de couches XL
        st.subheader("🎛️ Simulateur de Programme XL")
        
        nb_couches = st.slider("Nombre de couches XL", 1, 5, 3)
        
        couches_data = []
        priorite_cumulee = 0
        
        for i in range(nb_couches):
            st.markdown(f"### Couche {i+1}")
            col_c1, col_c2, col_c3 = st.columns([2,2,1])
            
            with col_c1:
                priorite = st.number_input(f"Priorité couche {i+1} (€)", 
                                         value=1000000 * (i+1), 
                                         key=f"priorite_{i}")
            with col_c2:
                limite = st.number_input(f"Limite couche {i+1} (€)", 
                                       value=500000, 
                                       key=f"limite_{i}")
            with col_c3:
                prix = st.number_input(f"Prix (%)", 
                                     value=2.5 + i*0.5, 
                                     key=f"prix_{i}",
                                     min_value=0.1, max_value=20.0, step=0.1)
            
            couches_data.append({
                'Couche': f"XL {i+1}",
                'Priorité': priorite,
                'Limite': limite,
                'Prix (%)': prix,
                'Plage': f"{priorite:,.0f} € - {priorite + limite:,.0f} €"
            })
            priorite_cumulee += limite
        
        # Simulation de sinistre
        st.subheader("📊 Répartition par Couche")
        
        sinistre_xl = st.number_input("Montant du sinistre principal (€)", value=1200000, step=100000)
        
        resultats_couches = []
        sinistre_restant = sinistre_xl
        cout_total = 0
        
        for couche in couches_data:
            if sinistre_restant <= couche['Priorité']:
                prise_couche = 0
            else:
                prise_couche = min(couche['Limite'], sinistre_restant - couche['Priorité'])
            
            cout_couche = couche['Limite'] * couche['Prix (%)'] / 100
            cout_total += cout_couche
            
            resultats_couches.append({
                'Couche': couche['Couche'],
                'Plage de Couverture': couche['Plage'],
                'Prise Réassureur': prise_couche,
                'Coût Annuel': cout_couche,
                'Sinistre Restant': sinistre_restant - prise_couche
            })
            sinistre_restant -= prise_couche
        
        df_resultats = pd.DataFrame(resultats_couches)
        st.dataframe(df_resultats, use_container_width=True)
        
        col_cout1, col_cout2 = st.columns(2)
        with col_cout1:
            st.metric("💸 Coût total du programme", f"{cout_total:,.0f} €")
        with col_cout2:
            st.metric("📈 Coût en % des primes", f"{(cout_total/5000000)*100:.2f}%")
    
    with tab3:
        st.subheader("📊 Applications Avancées et Optimisation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="case-study-box">
            <h4>🏭 Cas Complexe : Programme Combiné</h4>
            <p><b>Contexte</b> : Grand groupe industriel avec risques diversifiés</p>
            <p><b>Solution</b> : Programme à 3 niveaux :</p>
            <ul>
            <li><b>Niveau 1</b> : Quota-share 20% pour le portefeuille standard</li>
            <li><b>Niveau 2</b> : Surplus pour les risques spécifiques</li>
            <li><b>Niveau 3</b> : Stop Loss global pour les sinistres agrégés</li>
            </ul>
            <p><b>Résultat</b> : Optimisation coût/protection à -15% vs programme antérieur</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="theory-box">
            <h4>🎯 Stratégie d'Optimisation</h4>
            <p><b>Étapes clés :</b></p>
            <ol>
            <li>Analyse détaillée du portefeuille</li>
            <li>Identification des points de rupture</li>
            <li>Construction en couches successives</li>
            <li>Négociation par tranche</li>
            <li>Monitoring et ajustement</li>
            </ol>
            
            <div class="formula-box">
            <b>Règle d'or :</b><br>
            Coût réassurance ≤ Gain en capital libéré × Coût du capital
            </div>
            </div>
            """, unsafe_allow_html=True)

# =============================================================================
# SECTION 5: TARIFICATION TECHNIQUE
# =============================================================================
elif section == "💰 Tarification Technique":
    st.markdown('<div class="section-header">💰 Tarification Technique - Modèles et Méthodologies</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="theory-box">
    <h3>🎯 Principes Actuariels de Tarification</h3>
    <p>La tarification en réassurance combine <b>statistiques historiques</b>, <b>modélisation prospective</b> 
    et <b>jugement d'expert</b> pour déterminer des primes équitables et suffisantes.</p>
    
    <div class="formula-box">
    <b>Équation fondamentale :</b><br>
    Prime Commerciale = Prime Pure × (1 + Chargement Sécurité) + Frais + Marge Bénéficiaire
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🎯 Prime Pure", "📊 Prime Commerciale", "🔄 Commissions"])
    
    with tab1:
        st.subheader("🎯 Calcul de la Prime Pure")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="concept-box">
            <h4>📖 Définition de la Prime Pure</h4>
            <p>La <b>prime pure</b> représente l'espérance mathématique du coût des sinistres, 
            sans aucun chargement pour frais, sécurité ou bénéfice.</p>
            
            <div class="formula-box">
            Prime Pure = Fréquence × Coût Moyen Sinistre
            </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Calculateur prime pure
            st.subheader("🧮 Calculateur Prime Pure")
            
            frequence_sinistres = st.slider("Fréquence sinistres (%)", 0.1, 10.0, 2.5)
            cout_moyen_sinistre = st.number_input("Coût moyen sinistre (€)", value=50000)
            
            prime_pure = (frequence_sinistres / 100) * cout_moyen_sinistre
            
            st.metric("🎯 Prime pure calculée", f"{prime_pure:,.0f} €")
            
            # Analyse de sensibilité
            st.subheader("📈 Analyse de Sensibilité")
            
            variation_frequence = st.slider("Variation fréquence (%)", -50, 50, 10)
            variation_severite = st.slider("Variation sévérité (%)", -50, 50, 15)
            
            nouvelle_frequence = frequence_sinistres * (1 + variation_frequence/100)
            nouvelle_severite = cout_moyen_sinistre * (1 + variation_severite/100)
            nouvelle_prime_pure = (nouvelle_frequence / 100) * nouvelle_severite
            
            variation_prime = ((nouvelle_prime_pure - prime_pure) / prime_pure) * 100
            
            st.metric("🔄 Nouvelle prime pure", f"{nouvelle_prime_pure:,.0f} €")
            st.metric("📊 Variation", f"{variation_prime:+.1f}%")
        
        with col2:
            st.markdown("""
            <div class="theory-box">
            <h4>🧮 Méthodes d'Estimation</h4>
            <ul>
            <li><b>Méthode fréquentiste</b> : Basée sur l'expérience historique</li>
            <li><b>Méthode bayésienne</b> : Combinaison expérience propre/collective</li>
            <li><b>Crédibility Theory</b> : Poids accordé à différentes sources</li>
            <li><b>Modèles de risque</b> : Distributions probabilistes avancées</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Distribution des sinistres
            st.subheader("📊 Distribution des Sinistres")
            
            lambda_poisson = st.slider("Paramètre λ (fréquence)", 0.1, 5.0, 2.0)
            mu_lognormal = st.slider("μ lognormal", 9.0, 12.0, 10.5)
            sigma_lognormal = st.slider("σ lognormal", 0.1, 2.0, 1.0)
            
            # Simulation de la distribution
            n_simulations = 10000
            n_sinistres = np.random.poisson(lambda_poisson, n_simulations)
            couts_sinistres = np.random.lognormal(mu_lognormal, sigma_lognormal, n_simulations)
            
            fig_dist = px.histogram(couts_sinistres, nbins=50, 
                                  title="Distribution des Coûts de Sinistres",
                                  labels={'value': 'Coût du sinistre (€)', 'count': 'Fréquence'})
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Statistiques descriptives
            stats_data = {
                'Métrique': ['Moyenne', 'Médiane', 'Écart-type', 'VaR 95%', 'VaR 99%'],
                'Valeur': [
                    f"{np.mean(couts_sinistres):.2f}",
                    f"{np.median(couts_sinistres):.2f}",
                    f"{np.std(couts_sinistres):.2f}",
                    f"{np.percentile(couts_sinistres, 95):.2f}",
                    f"{np.percentile(couts_sinistres, 99):.2f}"
                ]
            }
            st.dataframe(pd.DataFrame(stats_data))
    
    with tab2:
        st.subheader("📊 Prime Commerciale")
        
        st.markdown("""
        <div class="concept-box">
        <h4>🏷️ Composition de la Prime Commerciale</h4>
        <p>La prime commerciale inclut tous les éléments nécessaires à la viabilité économique du contrat :</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Paramètres de tarification
            prime_pure_base = st.number_input("Prime pure de base (€)", value=50000)
            chargement_securite = st.slider("Chargement sécurité (%)", 5, 30, 15)
            frais_acquisition = st.slider("Frais d'acquisition (%)", 10, 25, 15)
            frais_gestion = st.slider("Frais de gestion (%)", 5, 20, 10)
            marge_benefice = st.slider("Marge bénéficiaire (%)", 5, 20, 10)
            
            # Calculs
            prime_risque = prime_pure_base * (1 + chargement_securite/100)
            prime_chargement_frais = prime_risque / (1 - (frais_acquisition + frais_gestion + marge_benefice)/100)
            
            # Détail des composants
            detail_chargement = {
                'Composant': ['Prime pure', 'Chargement sécurité', 'Frais acquisition', 'Frais gestion', 'Marge bénéficiaire'],
                'Montant (€)': [
                    prime_pure_base,
                    prime_risque - prime_pure_base,
                    prime_chargement_frais * frais_acquisition/100,
                    prime_chargement_frais * frais_gestion/100,
                    prime_chargement_frais * marge_benefice/100
                ],
                'Pourcentage': [
                    (prime_pure_base / prime_chargement_frais) * 100,
                    ((prime_risque - prime_pure_base) / prime_chargement_frais) * 100,
                    frais_acquisition,
                    frais_gestion,
                    marge_benefice
                ]
            }
            
            st.dataframe(pd.DataFrame(detail_chargement))
            
        with col2:
            st.metric("🎯 Prime pure", f"{prime_pure_base:,.0f} €")
            st.metric("🛡️ Prime de risque", f"{prime_risque:,.0f} €")
            st.metric("🏷️ Prime commerciale", f"{prime_chargement_frais:,.0f} €")
            
            # Graphique de composition
            composition = {
                'Élément': ['Prime pure', 'Chargement sécurité', 'Frais acquisition', 'Frais gestion', 'Marge bénéficiaire'],
                'Valeur (€)': [
                    prime_pure_base,
                    prime_risque - prime_pure_base,
                    prime_chargement_frais * frais_acquisition/100,
                    prime_chargement_frais * frais_gestion/100,
                    prime_chargement_frais * marge_benefice/100
                ]
            }
            
            fig_compo = px.pie(composition, values='Valeur (€)', names='Élément', 
                             title="Composition de la Prime Commerciale")
            st.plotly_chart(fig_compo, use_container_width=True)
            
            # Analyse de rentabilité
            ratio_combine_attendu = (prime_risque / prime_chargement_frais) * 100
            marge_nette = prime_chargement_frais - prime_risque
            
            st.metric("📈 Ratio combiné attendu", f"{ratio_combine_attendu:.1f}%")
            st.metric("💰 Marge nette attendue", f"{marge_nette:,.0f} €")

# =============================================================================
# SECTION 6: COMPTABILITÉ TECHNIQUE
# =============================================================================
elif section == "📊 Comptabilité Technique":
    st.markdown('<div class="section-header">📊 Comptabilité Technique - Principes et Applications</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="theory-box">
    <h3>🏛️ Cadre Comptable des Assureurs</h3>
    <p>La comptabilité technique des assureurs et réassureurs suit des principes spécifiques distincts 
    de la comptabilité générale, avec un focus sur la <b>mesure des engagements techniques</b>.</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Provisions Techniques", "📊 Ratios Clés", "💰 Résultat Technique", "🛡️ Solvabilité II"])
    
    with tab1:
        st.subheader("📈 Provisions Techniques")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="concept-box">
            <h4>📖 Types de Provisions</h4>
            <ul>
            <li><b>PSAP</b> : Provision pour Sinistres À Payer</li>
            <li><b>PPNA</b> : Provision pour Primes Non Acquises</li>
            <li><b>PRA</b> : Provision pour Risques en Cours</li>
            <li><b>Provision pour Equalisation</b></li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Calculateur PSAP
            st.subheader("🧮 Calculateur PSAP - Méthode Chain Ladder")
            
            sinistres_payes = st.number_input("Sinistres payés à ce jour (€)", value=2000000)
            sinistres_survenus = st.number_input("Sinistres survenus estimés (€)", value=3500000)
            
            provision_sinistres = max(0, sinistres_survenus - sinistres_payes)
            
            st.metric("📊 Provision pour sinistres (PSAP)", f"{provision_sinistres:,.0f} €")
            
            # Calculateur PPNA
            st.subheader("📅 Calculateur PPNA")
            
            primes_annee = st.number_input("Primes de l'année (€)", value=5000000)
            duree_moyenne = st.slider("Durée moyenne contrats (mois)", 1, 12, 6)
            
            ppna = primes_annee * (12 - duree_moyenne) / 12
            
            st.metric("📅 Provision pour primes non acquises", f"{ppna:,.0f} €")
        
        with col2:
            st.markdown("""
            <div class="theory-box">
            <h4>📐 Méthode Chain Ladder</h4>
            <p>Méthode actuarielle pour l'estimation des sinistres à payer :</p>
            <div class="formula-box">
            PSAP = Sinistres Survenus - Sinistres Payés
            </div>
            <p>Les facteurs de développement sont estimés à partir de l'historique des sinistres.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Tableau de développement
            st.subheader("📈 Tableau de Développement")
            
            # Données simulées pour le développement
            annees = [2019, 2020, 2021, 2022, 2023]
            developpement_data = {
                'Année': annees,
                'Développement 1 an': [1.8, 1.7, 1.9, 1.8, 1.7],
                'Développement 2 ans': [1.4, 1.3, 1.5, 1.4, None],
                'Développement 3 ans': [1.2, 1.1, 1.2, None, None],
                'Développement final': [1.1, 1.1, None, None, None]
            }
            
            st.dataframe(pd.DataFrame(developpement_data), use_container_width=True)
            
            st.markdown("""
            <div class="warning-box">
            <h4>⚠️ Points d'Attention</h4>
            <ul>
            <li>Qualité des données historiques</li>
            <li>Stabilité du portefeuille</li>
            <li>Changements réglementaires</li>
            <li>Événements exceptionnels</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.subheader("📊 Ratios Techniques Clés")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            primes_emises = st.number_input("Primes émises (€)", value=5000000)
            sinistres_payes = st.number_input("Sinistres payés (€)", value=3000000)
            ratio_sinistralite = (sinistres_payes / primes_emises) * 100
            st.metric("📈 Ratio de sinistralité", f"{ratio_sinistralite:.1f}%")
        
        with col2:
            frais_gestion = st.number_input("Frais de gestion (€)", value=1500000)
            ratio_frais = (frais_gestion / primes_emises) * 100
            st.metric("💼 Ratio de frais", f"{ratio_frais:.1f}%")
        
        with col3:
            resultat_technique = primes_emises - sinistres_payes - frais_gestion
            ratio_combined = ratio_sinistralite + ratio_frais
            st.metric("⚖️ Ratio combiné", f"{ratio_combined:.1f}%")
            st.metric("💰 Résultat technique", f"{resultat_technique:,.0f} €")
        
        # Analyse détaillée
        st.subheader("📈 Analyse des Ratios")
        
        ratios_data = {
            'Ratio': ['Sinistralité', 'Frais', 'Combined', 'Rentabilité'],
            'Valeur': [ratio_sinistralite, ratio_frais, ratio_combined, (resultat_technique/primes_emises)*100],
            'Cible': [65, 25, 90, 10],
            'Écart': [ratio_sinistralite-65, ratio_frais-25, ratio_combined-90, (resultat_technique/primes_emises)*100-10]
        }
        
        df_ratios = pd.DataFrame(ratios_data)
        st.dataframe(df_ratios, use_container_width=True)
        
        # Graphique des ratios
        fig_ratios = px.bar(df_ratios, x='Ratio', y=['Valeur', 'Cible'], 
                          barmode='group', title="Comparaison Ratios Réels vs Cibles")
        st.plotly_chart(fig_ratios, use_container_width=True)

# =============================================================================
# SECTION 7: GESTION DES CATASTROPHES
# =============================================================================
elif section == "🌪️ Gestion des Catastrophes":
    st.markdown('<div class="section-header">🌪️ Gestion des Risques Catastrophiques</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="theory-box">
    <h3>🎯 Modélisation des Catastrophes Naturelles</h3>
    <p>La modélisation des catastrophes combine <b>données historiques</b>, <b>modèles physiques</b> 
    et <b>analyses statistiques</b> pour estimer les pertes potentielles.</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🎯 Modélisation", "📊 Couverture", "🛡️ Gestion de Crise"])
    
    with tab1:
        st.subheader("🎯 Modélisation des Catastrophes")
        
        col1, col2 = st.columns(2)
        
        with col1:
            type_catastrophe = st.selectbox("Type de catastrophe", [
                "Séisme", "Ouragan", "Inondation", "Incendie", "Grêle"
            ])
            
            intensite = st.slider("Intensité", 1, 10, 7)
            zone_affectee = st.number_input("Zone affectée (km²)", value=5000)
            densite_construction = st.slider("Densité construction", 0.1, 1.0, 0.7)
            valeur_par_km2 = st.number_input("Valeur assurée par km² (M€)", value=50)
            
            # Calcul dommages estimés
            dommage_base = {
                "Séisme": 1.5,
                "Ouragan": 1.2, 
                "Inondation": 0.8,
                "Incendie": 0.6,
                "Grêle": 0.3
            }
            
            dommage_estime = dommage_base[type_catastrophe] * intensite * densite_construction * zone_affectee * valeur_par_km2 * 1000000
            
            st.metric("💥 Dommage total estimé", f"{dommage_estime:,.0f} €")
            
            # Probabilité d'occurrence
            proba_annee = {
                "Séisme": 2,
                "Ouragan": 5,
                "Inondation": 10,
                "Incendie": 8,
                "Grêle": 15
            }
            
            st.metric("📅 Probabilité annuelle", f"{proba_annee[type_catastrophe]}%")
            
        with col2:
            st.markdown("""
            <div class="concept-box">
            <h4>📊 Modèles de Référence</h4>
            <ul>
            <li><b>RMS</b> : Risk Management Solutions</li>
            <li><b>AIR</b> : Applied Insurance Research</li>
            <li><b>EQECAT</b> : Modèles sismiques</li>
            <li><b>Modèles propriétaires</b> : Développés en interne</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Courbe PML (Probable Maximum Loss)
            st.subheader("📈 Courbe PML")
            
            periods = [1, 10, 50, 100, 250, 500]
            pml_values = [dommage_estime * p/100 for p in [80, 50, 20, 10, 4, 2]]
            
            fig_pml = px.line(x=periods, y=pml_values, 
                            labels={'x': 'Période de retour (ans)', 'y': 'PML (€)'},
                            title="Courbe Probable Maximum Loss")
            st.plotly_chart(fig_pml, use_container_width=True)
    
    with tab2:
        st.subheader("📊 Couverture Catastrophe")
        
        col1, col2 = st.columns(2)
        
        with col1:
            priorite_cat = st.number_input("Priorité programme cat (€)", value=100000000)
            limite_cat = st.number_input("Limite programme cat (€)", value=200000000)
            prime_cat = st.number_input("Prime catastrophe (€)", value=5000000)
            
            prise_reassureur_cat = max(0, min(limite_cat, dommage_estime - priorite_cat))
            
            st.metric("🛡️ Part cédante", f"{min(dommage_estime, priorite_cat):,.0f} €")
            st.metric("🤝 Part réassureurs", f"{prise_reassureur_cat:,.0f} €")
            st.metric("💰 Prime catastrophe", f"{prime_cat:,.0f} €")
            
            # Taux de prime
            taux_prime = (prime_cat / (priorite_cat + limite_cat)) * 100
            st.metric("📊 Taux de prime", f"{taux_prime:.2f}%")
        
        with col2:
            # Graphique de couverture
            fig_cat = go.Figure(go.Waterfall(
                name="Répartition sinistre cat",
                orientation="v",
                measure=["relative", "relative", "total"],
                x=["Dommage total", "Priorité cédante", "Part réassureur"],
                textposition="outside",
                y=[dommage_estime, -priorite_cat, -prise_reassureur_cat]
            ))
            fig_cat.update_layout(title="Répartition du Sinistre Catastrophe")
            st.plotly_chart(fig_cat, use_container_width=True)
            
            # Analyse coût-bénéfice
            esperance_sinistre = dommage_estime * (proba_annee[type_catastrophe] / 100)
            benefice_protection = min(prise_reassureur_cat, esperance_sinistre - priorite_cat)
            ratio_cout_benefice = prime_cat / benefice_protection if benefice_protection > 0 else float('inf')
            
            st.metric("📈 Espérance de sinistre", f"{esperance_sinistre:,.0f} €")
            st.metric("🎯 Bénéfice de protection", f"{benefice_protection:,.0f} €")
            st.metric("⚖️ Ratio coût/bénéfice", f"{ratio_cout_benefice:.2f}")

# =============================================================================
# SECTION 8: SOLVABILITÉ & RÉGLEMENTATION (COMPLÉTÉE)
# =============================================================================
elif section == "🛡️ Solvabilité & Réglementation":
    st.markdown('<div class="section-header">🛡️ Solvabilité II - Cadre Réglementaire Complet</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="theory-box">
    <h3>🏛️ Cadre Solvabilité II - Les Trois Piliers</h3>
    <p>Le cadre Solvabilité II, applicable depuis 2016, repose sur <b>trois piliers</b> complémentaires 
    pour assurer la stabilité financière des assureurs et réassureurs en Europe.</p>
    
    <div class="formula-box">
    <b>Objectif fondamental :</b><br>
    Protéger les assurés avec une probabilité de 99.5% sur un an horizon<br>
    Niveau de confiance : VaR 99.5% (Value at Risk)
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📊 Pilier I - Quantitatif", "🎯 Pilier II - Qualitatif", "📋 Pilier III - Transparence"])
    
    with tab1:
        st.subheader("📊 Pilier I - Exigences Quantitatives")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="concept-box">
            <h4>🎯 Solvency Capital Requirement (SCR)</h4>
            <p><b>Définition :</b> Capital requis pour absorber les chocs avec une probabilité de 99.5% sur un an.</p>
            
            <div class="formula-box">
            <b>Formule standard détaillée :</b><br>
            SCR = √(∑∑ρ_ij × SCR_i × SCR_j)<br><br>
            <b>Où :</b><br>
            • ρ_ij = coefficient de corrélation entre modules i et j<br>
            • SCR_i = capital requis pour le module i<br>
            • Somme double sur tous les modules de risque
            </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Explication détaillée des modules
            st.markdown("""
            <div class="theory-box">
            <h4>🔍 Modules de Risque Principaux</h4>
            <ul>
            <li><b>Module Souscription</b> : Risque de souscription vie et non-vie</li>
            <li><b>Module Marché</b> : Risque de marché (actions, taux, immobilier)</li>
            <li><b>Module Contrepartie</b> : Risque de défaut des contreparties</li>
            <li><b>Module Opérationnel</b> : Risques opérationnels divers</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Calculateur SCR détaillé
            st.subheader("🧮 Calculateur SCR Détaillé")
            
            st.markdown("**Module Souscription**")
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                scr_vie = st.number_input("SCR Vie (€)", value=30000000)
            with col_s2:
                scr_non_vie = st.number_input("SCR Non-Vie (€)", value=40000000)
            
            st.markdown("**Module Marché**")
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                scr_actions = st.number_input("SCR Actions (€)", value=20000000)
            with col_m2:
                scr_taux = st.number_input("SCR Taux (€)", value=15000000)
            with col_m3:
                scr_immobilier = st.number_input("SCR Immobilier (€)", value=10000000)
            
            st.markdown("**Autres Modules**")
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                scr_contrepartie = st.number_input("SCR Contrepartie (€)", value=10000000)
            with col_a2:
                scr_operationnel = st.number_input("SCR Opérationnel (€)", value=5000000)
            
            # Calcul SCR avec matrice de corrélation complète
            # Matrice de corrélation standard Solvabilité II
            corr_matrix = {
                'souscription_vie': [1.0, 0.5, 0.25, 0.25, 0.25, 0.25],
                'souscription_non_vie': [0.5, 1.0, 0.25, 0.25, 0.25, 0.25],
                'marche_actions': [0.25, 0.25, 1.0, 0.5, 0.5, 0.25],
                'marche_taux': [0.25, 0.25, 0.5, 1.0, 0.5, 0.25],
                'marche_immobilier': [0.25, 0.25, 0.5, 0.5, 1.0, 0.25],
                'contrepartie': [0.25, 0.25, 0.25, 0.25, 0.25, 1.0]
            }
            
            # Calcul détaillé
            scr_souscription = math.sqrt(scr_vie**2 + scr_non_vie**2 + 2*0.5*scr_vie*scr_non_vie)
            scr_marche = math.sqrt(scr_actions**2 + scr_taux**2 + scr_immobilier**2 + 
                                 2*0.5*scr_actions*scr_taux + 2*0.5*scr_actions*scr_immobilier + 
                                 2*0.5*scr_taux*scr_immobilier)
            
            # SCR global
            scr_global = math.sqrt(
                scr_souscription**2 + scr_marche**2 + scr_contrepartie**2 + scr_operationnel**2 +
                2*0.25*scr_souscription*scr_marche +
                2*0.25*scr_souscription*scr_contrepartie +
                2*0.25*scr_souscription*scr_operationnel +
                2*0.25*scr_marche*scr_contrepartie +
                2*0.25*scr_marche*scr_operationnel +
                2*0.25*scr_contrepartie*scr_operationnel
            )
            
            st.metric("🛡️ SCR Souscription", f"{scr_souscription:,.0f} €")
            st.metric("📈 SCR Marché", f"{scr_marche:,.0f} €")
            st.metric("🛡️ SCR Global Calculé", f"{scr_global:,.0f} €")
        
        with col2:
            st.markdown("""
            <div class="theory-box">
            <h4>📈 Minimum Capital Requirement (MCR)</h4>
            <p><b>Définition :</b> Niveau de capital minimum en dessous duquel l'autorité de contrôle intervient immédiatement.</p>
            
            <div class="formula-box">
            <b>Formule MCR :</b><br>
            MCR = Max(25% × SCR, MCR_plancher)<br><br>
            <b>Où :</b><br>
            • 25% × SCR = partie liée au risque<br>
            • MCR_plancher = minimum absolu (2.2M€ pour vie, 1.5M€ pour non-vie)<br>
            • Plafond : 45% × SCR
            </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Calculateur MCR
            st.subheader("📋 Calculateur MCR")
            
            type_assureur = st.selectbox("Type d'assureur", ["Vie", "Non-Vie", "Mixte"])
            
            if type_assureur == "Vie":
                mcr_plancher = 2200000
            elif type_assureur == "Non-Vie":
                mcr_plancher = 1500000
            else:  # Mixte
                mcr_plancher = 2500000
            
            mcr_calc = max(scr_global * 0.25, mcr_plancher)
            mcr_plafond = scr_global * 0.45
            
            st.metric("📊 MCR Calculé", f"{mcr_calc:,.0f} €")
            st.metric("📈 Plancher MCR", f"{mcr_plancher:,.0f} €")
            st.metric("📉 Plafond MCR", f"{mcr_plafond:,.0f} €")
            
            # Analyse de solvabilité
            st.subheader("📊 Analyse de Solvabilité")
            
            capital_disponible = st.number_input("Capital disponible (€)", value=80000000)
            ratio_solvabilite = (capital_disponible / scr_global) * 100
            
            st.metric("💰 Capital disponible", f"{capital_disponible:,.0f} €")
            st.metric("📊 Ratio de solvabilité", f"{ratio_solvabilite:.1f}%")
            
            # Interprétation du ratio
            if ratio_solvabilite >= 150:
                st.success("✅ **Niveau excellent** - Très bon niveau de capitalisation")
                st.info("Marge de sécurité confortable au-dessus des exigences réglementaires")
            elif ratio_solvabilite >= 100:
                st.warning("⚠️ **Niveau suffisant** - Capitalisation adéquate mais à surveiller")
                st.info("Respect des exigences mais marge de sécurité limitée")
            else:
                st.error("🚨 **Niveau insuffisant** - Mesures correctives requises")
                st.info("Plan de recapitalisation nécessaire - Intervention de l'autorité de contrôle")
            
            # Impact de la réassurance sur le SCR
            st.subheader("🔄 Impact Réassurance sur SCR")
            
            reduction_scr = st.slider("Réduction SCR grâce à la réassurance (%)", 0, 50, 20)
            nouveau_scr = scr_global * (1 - reduction_scr/100)
            nouveau_ratio = (capital_disponible / nouveau_scr) * 100
            
            st.metric("🛡️ Nouveau SCR", f"{nouveau_scr:,.0f} €")
            st.metric("📈 Nouveau ratio", f"{nouveau_ratio:.1f}%")
            st.metric("📊 Amélioration", f"{(nouveau_ratio - ratio_solvabilite):+.1f} points")
            
            # Calcul du gain en capital
            gain_capital = scr_global - nouveau_scr
            st.metric("💰 Gain en capital libéré", f"{gain_capital:,.0f} €")
    
    with tab2:
        st.subheader("🎯 Pilier II - Exigences Qualitatives")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="concept-box">
            <h4>🏛️ Gouvernance et Contrôle Interne</h4>
            <p><b>Objectif :</b> Assurer une gestion saine et prudente des risques</p>
            
            <h5>📋 Éléments Clés :</h5>
            <ul>
            <li><b>Organe de surveillance</b> : Conseil d'administration compétent</li>
            <li><b>Fonction de contrôle</b> : Risk Management, Compliance, Audit interne</li>
            <li><b>Politique de rémunération</b> : Alignée sur le risque à long terme</li>
            <li><b>Système de contrôle interne</b> : Processus documentés et contrôlés</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="theory-box">
            <h4>📊 Own Risk and Solvency Assessment (ORSA)</h4>
            <p><b>Définition :</b> Processus interne d'évaluation globale des risques et de la solvabilité</p>
            
            <div class="formula-box">
            <b>Étapes de l'ORSA :</b><br>
            1. Identification des risques significatifs<br>
            2. Évaluation quantitative et qualitative<br>
            3. Détermination du capital économique interne<br>
            4. Planification stratégique et capital<br>
            5. Surveillance continue et reporting
            </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="theory-box">
            <h4>🎯 Gestion des Risques</h4>
            <p><b>Exigences principales :</b></p>
            
            <h5>🔍 Système de Gouvernance des Risques</h5>
            <ul>
            <li>Stratégie de risque définie et approuvée</li>
            <li>Appétit pour le risque quantifié</li>
            <li>Limites de risque opérationnelles</li>
            <li>Processus d'escalade défini</li>
            </ul>
            
            <h5>📈 Fonction Actuarielle</h5>
            <ul>
            <li>Évaluation techniques des provisions</li>
            <li>Calculs de solvabilité</li>
            <li>Tests de sensibilité</li>
            <li>Validation des modèles</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Auto-évaluation ORSA
            st.subheader("🧮 Auto-évaluation ORSA")
            
            st.markdown("**Évaluez votre maturité ORSA (1-5)**")
            
            col_o1, col_o2 = st.columns(2)
            with col_o1:
                gouvernance = st.slider("Gouvernance des risques", 1, 5, 3)
                identification_risques = st.slider("Identification risques", 1, 5, 3)
                capital_economique = st.slider("Capital économique", 1, 5, 2)
            with col_o2:
                planification = st.slider("Planification stratégique", 1, 5, 3)
                surveillance = st.slider("Surveillance continue", 1, 5, 2)
                reporting = st.slider("Reporting interne", 1, 5, 3)
            
            score_orsa = (gouvernance + identification_risques + capital_economique + planification + surveillance + reporting) / 6
            
            st.metric("📊 Score ORSA moyen", f"{score_orsa:.1f}/5")
            
            if score_orsa >= 4:
                st.success("✅ Maturité ORSA avancée")
            elif score_orsa >= 3:
                st.warning("⚠️ Maturité ORSA intermédiaire")
            else:
                st.error("🔴 Maturité ORSA à développer")
    
    with tab3:
        st.subheader("📋 Pilier III - Transparence et Reporting")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="concept-box">
            <h4>📊 Reporting Réglementaire</h4>
            <p><b>Objectif :</b> Assurer la transparence vis-à-vis des superviseurs et du marché</p>
            
            <h5>📋 Rapports Principaux :</h5>
            <ul>
            <li><b>Rapport de Solvabilité et de Situation Financière (RSSF)</b></li>
            <li><b>Déclarations Réglementaires Régulières (QRTs)</b></li>
            <li><b>Rapport sur la Politique de Rémunération</b></li>
            <li><b>Rapport ORSA</b> (confidentiel)</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="theory-box">
            <h4>🌐 Disclosure Public</h4>
            <p><b>Exigences de transparence marché :</b></p>
            
            <div class="formula-box">
            <b>Reportings publics obligatoires :</b><br>
            • Rapport annuel de solvabilité<br>
            • Informations sur le profil de risque<br>
            • Politique de rémunération<br>
            • Performance et capitalisation<br>
            • Informations qualitatives et quantitatives
            </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="theory-box">
            <h4>📅 Calendrier Réglementaire</h4>
            
            <h5>🗓️ Échéances Clés :</h5>
            <ul>
            <li><b>15 février</b> : Reporting trimestriel Q1-Q3</li>
            <li><b>1er mai</b> : Reporting annuel et RSSF</li>
            <li><b>30 juin</b> : Rapport ORSA</li>
            <li><b>Publication immédiate</b> : Événements significatifs</li>
            </ul>
            
            <h5>⚖️ Sanctions :</h5>
            <ul>
            <li>Retard de reporting : sanctions pécuniaires</li>
            <li>Informations erronées : suspension d'agrément</li>
            <li>Non-conformité répétée : radiation</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Checklist conformité
            st.subheader("✅ Checklist Conformité Pilier III")
            
            conformite_items = {
                'RSSF publié': st.checkbox("RSSF publié sur le site internet", value=True),
                'QRTs déposés': st.checkbox("QRTs déposés auprès de l'ACPR", value=True),
                'Politique rémunération': st.checkbox("Politique de rémunération publiée", value=False),
                'Profil risque public': st.checkbox("Profil de risque public", value=True),
                'Procédures documentées': st.checkbox("Procédures de reporting documentées", value=True),
                'Contrôles internes': st.checkbox("Contrôles internes validés", value=False)
            }
            
            score_conformite = sum(conformite_items.values()) / len(conformite_items) * 100
            
            st.metric("📊 Taux de conformité", f"{score_conformite:.0f}%")
            
            if score_conformite >= 90:
                st.success("✅ Conformité excellente")
            elif score_conformite >= 70:
                st.warning("⚠️ Conformité satisfaisante")
            else:
                st.error("🔴 Conformité insuffisante")

# =============================================================================
# SECTION 9: ÉTUDES DE CAS CONCRETS (COMPLÉTÉE)
# =============================================================================
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
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="concept-box">
            <h4>📊 Données Initiales</h4>
            <ul>
            <li>Primes totales : 50M€</li>
            <li>Sinistres attendus : 35M€</li>
            <li>Rétention actuelle : 500k€ par sinistre</li>
            <li>Programme actuel : Quota-Share 30% + Surplus</li>
            <li>Coût réassurance : 7.5M€</li>
            <li>Ratio combiné : 102%</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Analyse de la situation actuelle
            st.subheader("📈 Analyse Actuelle")
            
            primes_totales = 50000000
            sinistres_attendus = 35000000
            quote_part_actuelle = 30
            retention_actuelle = 500000
            cout_actuel = 7500000
            
            prime_cedee_actuelle = primes_totales * quote_part_actuelle / 100
            sinistre_cede_actuel = sinistres_attendus * quote_part_actuelle / 100
            
            st.metric("💰 Prime cédée actuelle", f"{prime_cedee_actuelle:,.0f} €")
            st.metric("⚡ Sinistre cédé actuel", f"{sinistre_cede_actuel:,.0f} €")
            st.metric("💸 Coût réassurance actuel", f"{cout_actuel:,.0f} €")
        
        with col2:
            st.markdown("""
            <div class="theory-box">
            <h4>🎯 Objectifs d'Optimisation</h4>
            <ul>
            <li>Réduire le coût de la réassurance de 15%</li>
            <li>Maintenir un niveau de protection adéquat</li>
            <li>Améliorer le ratio combiné de 2 points</li>
            <li>Optimiser l'utilisation du capital</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Proposition d'optimisation
            st.subheader("🔄 Proposition d'Optimisation")
            
            nouvelle_retention = st.slider("Nouvelle rétention (€)", 500000, 2000000, 750000)
            nouveau_quote_part = st.slider("Nouveau quota-share (%)", 10, 40, 20)
            
            economie_prime = prime_cedee_actuelle - (primes_totales * nouveau_quote_part / 100)
            nouveau_sinistre_cede = sinistres_attendus * nouveau_quote_part / 100
            nouveau_cout = cout_actuel * 0.85  # Réduction de 15%
            
            st.metric("💸 Économie sur primes", f"{economie_prime:,.0f} €")
            st.metric("📈 Nouveau sinistre cédé", f"{nouveau_sinistre_cede:,.0f} €")
            st.metric("💰 Nouveau coût réassurance", f"{nouveau_cout:,.0f} €")
            st.metric("📊 Économie totale", f"{cout_actuel - nouveau_cout:,.0f} €")
            
            # Impact sur la rentabilité
            benefice_supplementaire = (cout_actuel - nouveau_cout) + (prime_cedee_actuelle - (primes_totales * nouveau_quote_part / 100))
            nouveau_ratio_combine = 102 - (benefice_supplementaire / primes_totales * 100)
            
            st.metric("🎯 Nouveau ratio combiné", f"{nouveau_ratio_combine:.1f}%")
            st.metric("📈 Amélioration rentabilité", f"{benefice_supplementaire:,.0f} €")
    
    with tab2:
        st.subheader("🏠 Cas : Programme Habitation avec Exposition Catastrophe")
        
        st.markdown("""
        <div class="case-study-box">
        <h4>📖 Contexte</h4>
        <p><b>Assureur HabitatSecur</b> : Portefeuille habitation de 80M€ de primes avec forte exposition 
        aux risques naturels dans le Sud-Est de la France. Exposition significative aux inondations et séismes.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="concept-box">
            <h4>📊 Analyse des Expositions</h4>
            
            <h5>🌪️ Risques Principaux :</h5>
            <ul>
            <li><b>Inondations</b> : 25M€ d'exposition (crue centennale)</li>
            <li><b>Séismes</b> : 15M€ d'exposition (séisme 5.5)</li>
            <li><b>Tempêtes</b> : 20M€ d'exposition (tempête 1999)</li>
            <li><b>Incendies</b> : 10M€ d'exposition</li>
            </ul>
            
            <h5>📈 Données Techniques :</h5>
            <ul>
            <li>Primes totales : 80M€</li>
            <li>Sinistres normaux attendus : 48M€</li>
            <li>PML inondation : 45M€ (période retour 100 ans)</li>
            <li>PML séisme : 35M€ (période retour 200 ans)</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Calculateur exposition catastrophe
            st.subheader("🎯 Calculateur Exposition CAT")
            
            exposition_inondation = st.number_input("Exposition inondation (M€)", value=25.0)
            exposition_seisme = st.number_input("Exposition séisme (M€)", value=15.0)
            exposition_tempete = st.number_input("Exposition tempête (M€)", value=20.0)
            
            pml_inondation = exposition_inondation * 1.8  # Multiplicateur PML
            pml_seisme = exposition_seisme * 2.3
            pml_tempete = exposition_tempete * 1.4
            
            st.metric("💥 PML Inondation", f"{pml_inondation:,.1f} M€")
            st.metric("🌋 PML Séisme", f"{pml_seisme:,.1f} M€")
            st.metric("💨 PML Tempête", f"{pml_tempete:,.1f} M€")
            
            exposition_totale_cat = pml_inondation + pml_seisme + pml_tempete
            st.metric("📊 Exposition CAT totale", f"{exposition_totale_cat:,.1f} M€")
        
        with col2:
            st.markdown("""
            <div class="theory-box">
            <h4>🛡️ Programme de Réassurance Proposé</h4>
            
            <h5>🏗️ Structure en Couches :</h5>
            <ul>
            <li><b>Couche 1</b> : Quota-share 20% pour le portefeuille standard</li>
            <li><b>Couche 2</b> : Surplus pour les risques individuels élevés</li>
            <li><b>Couche 3</b> : Stop Loss global à 110% des primes</li>
            <li><b>Couche 4</b> : Programme catastrophe dédié</li>
            </ul>
            
            <h5>💰 Coût du Programme :</h5>
            <ul>
            <li>Prime totale réassurance : 9.5M€</li>
            <li>Économie vs programme actuel : 2.5M€</li>
            <li>Amélioration ratio combiné : -3.1 points</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Simulateur programme CAT
            st.subheader("🌪️ Simulateur Programme Catastrophe")
            
            priorite_cat = st.number_input("Priorité CAT (M€)", value=10.0)
            limite_cat = st.number_input("Limite CAT (M€)", value=30.0)
            prime_cat = st.number_input("Prime CAT (M€)", value=2.5)
            
            # Simulation sinistre CAT
            sinistre_cat = st.selectbox("Scénario CAT", 
                                      ["Crue moyenne (15M€)", "Crue majeure (25M€)", "Séisme modéré (20M€)", "Séisme majeur (35M€)"])
            
            sinistre_valeurs = {
                "Crue moyenne (15M€)": 15,
                "Crue majeure (25M€)": 25,
                "Séisme modéré (20M€)": 20,
                "Séisme majeur (35M€)": 35
            }
            
            sinistre_montant = sinistre_valeurs[sinistre_cat] * 1000000
            
            prise_reassureur = max(0, min(limite_cat * 1000000, sinistre_montant - priorite_cat * 1000000))
            part_cedeante = sinistre_montant - prise_reassureur
            
            st.metric("💥 Sinistre CAT", f"{sinistre_montant/1000000:,.1f} M€")
            st.metric("🛡️ Part cédante", f"{part_cedeante/1000000:,.1f} M€")
            st.metric("🤝 Part réassureurs", f"{prise_reassureur/1000000:,.1f} M€")
            
            # Analyse de la protection
            taux_couverture = (prise_reassureur / sinistre_montant) * 100
            st.metric("📊 Taux de couverture", f"{taux_couverture:.1f}%")
            
            st.markdown("""
            <div class="success-box">
            <h4>✅ Résultats Attendus</h4>
            <ul>
            <li><b>Ratio combiné</b> : 96.5% (-5.5 points vs initial)</li>
            <li><b>Économie annuelle</b> : 2.5M€</li>
            <li><b>SCR réduit</b> : -18% grâce à la réassurance CAT</li>
            <li><b>Stabilité résultats</b> : Protection contre les chocs majeurs</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        st.subheader("🌍 Cas : Réassureur Global - Gestion de Portefeuille International")
        
        st.markdown("""
        <div class="case-study-box">
        <h4>📖 Contexte</h4>
        <p><b>GlobalRe</b> : Réassureur tier 1 avec un portefeuille mondial de 5Md€ de primes. 
        Présent sur tous les continents avec des expositions diversifiées mais concentrées sur certains risques catastrophiques.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="concept-box">
            <h4>🌐 Portefeuille Mondial</h4>
            
            <h5>🗺️ Répartition Géographique :</h5>
            <ul>
            <li><b>Amérique du Nord</b> : 35% (1.75Md€) - Ouragans, séismes</li>
            <li><b>Europe</b> : 30% (1.5Md€) - Tempêtes, inondations</li>
            <li><b>Asie-Pacifique</b> : 25% (1.25Md€) - Typhons, séismes, tsunamis</li>
            <li><b>Amérique Latine</b> : 6% (0.3Md€) - Séismes, éruptions</li>
            <li><b>Afrique</b> : 4% (0.2Md€) - Risques politiques, sécheresses</li>
            </ul>
            
            <h5>📊 Mix de Produits :</h5>
            <ul>
            <li><b>Non-vie</b> : 70% (Property, Casualty)</li>
            <li><b>Vie</b> : 20% (Longévité, mortalité)</li>
            <li><b>Spécialités</b> : 10% (Credit, Aviation, Marine)</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Analyse de concentration
            st.subheader("🎯 Analyse de Concentration")
            
            regions = ['Amérique Nord', 'Europe', 'Asie-Pacifique', 'Amérique Latine', 'Afrique']
            expositions = [35, 30, 25, 6, 4]
            pml_regions = [45, 25, 35, 8, 5]  # PML en % du portefeuille
            
            fig_concentration = go.Figure(data=[
                go.Bar(name='Exposition (%)', x=regions, y=expositions),
                go.Bar(name='PML Maximal (%)', x=regions, y=pml_regions)
            ])
            fig_concentration.update_layout(title="Concentration Géographique et PML")
            st.plotly_chart(fig_concentration, use_container_width=True)
            
            # Indice de concentration Herfindahl
            herfindahl = sum([(exp/100)**2 for exp in expositions]) * 10000
            st.metric("📊 Indice Herfindahl", f"{herfindahl:.0f}")
            
            if herfindahl > 2500:
                st.warning("⚠️ Concentration élevée - Diversification recommandée")
            else:
                st.success("✅ Bonne diversification géographique")
        
        with col2:
            st.markdown("""
            <div class="theory-box">
            <h4>🏗️ Stratégie de Réassurance du Réassureur</h4>
            
            <h5>🔄 Programme de Rétrocession :</h5>
            <ul>
            <li><b>Quota-share</b> : 15% du portefeuille global</li>
            <li><b>Surplus</b> : Pour les risques concentrés</li>
            <li><b>Stop Loss</b> : Protection agrégée du portefeuille</li>
            <li><b>Cat Bonds</b> : 5% de l'exposition CAT via marchés capitaux</li>
            <li><b>Sidecars</b> : Financement alternatif pour pics de capacité</li>
            </ul>
            
            <h5>💰 Optimisation du Capital :</h5>
            <ul>
            <li><b>SCR initial</b> : 1.2Md€</li>
            <li><b>SCR après rétrocession</b> : 950M€</li>
            <li><b>Économie de capital</b> : 250M€</li>
            <li><b>Coût rétrocession</b> : 85M€/an</li>
            <li><b>ROE cible</b> : >12% après optimisation</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Calculateur optimisation capital
            st.subheader("🧮 Calculateur Optimisation Capital")
            
            scr_initial = st.number_input("SCR initial (Md€)", value=1.2)
            taux_retrocession = st.slider("Taux de rétrocession (%)", 5, 30, 15)
            cout_retrocession = st.number_input("Coût rétrocession (M€)", value=85)
            
            scr_reduit = scr_initial * (1 - taux_retrocession/100)
            economie_capital = scr_initial - scr_reduit
            
            # Calcul ROE
            resultat_net = st.number_input("Résultat net attendu (M€)", value=180)
            capital_libere = economie_capital * 1000  # Conversion en M€
            cout_capital = 0.10  # 10% de coût du capital
            
            gain_capital = capital_libere * cout_capital
            resultat_ameliore = resultat_net + gain_capital - cout_retrocession
            roe_initial = (resultat_net / (scr_initial * 1000)) * 100
            roe_final = (resultat_ameliore / (scr_reduit * 1000)) * 100
            
            st.metric("🛡️ SCR après optimisation", f"{scr_reduit:.2f} Md€")
            st.metric("💰 Économie de capital", f"{economie_capital:.2f} Md€")
            st.metric("📈 ROE initial", f"{roe_initial:.1f}%")
            st.metric("🎯 ROE final", f"{roe_final:.1f}%")
            st.metric("📊 Amélioration ROE", f"{roe_final - roe_initial:+.1f} points")
            
            st.markdown("""
            <div class="success-box">
            <h4>✅ Stratégie Recommandée</h4>
            <p><b>Optimisation du programme de rétrocession :</b></p>
            <ul>
            <li>Maintenir le quota-share à 15% pour la stabilité</li>
            <li>Développer les solutions alternatives (Cat Bonds, ILS)</li>
            <li>Renforcer la surveillance des concentrations</li>
            <li>Optimiser le coût du capital via la rétrocession</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

# =============================================================================
# SECTION 10: ANALYSE DATA SCIENCE
# =============================================================================
elif section == "📊 Analyse Data Science":
    st.markdown('<div class="section-header">📊 Analyse Data Science - KPI & Prévisions</div>', unsafe_allow_html=True)
    
    # Sidebar pour les données
    with st.sidebar:
        st.subheader("📥 Chargement des Données")
        uploaded_file = st.file_uploader("Importer CSV/Excel", type=["csv", "xlsx", "xls"])
        
        st.subheader("⚙️ Configuration")
        use_demo_data = st.checkbox("Utiliser les données de démonstration", value=True)
        freq = st.selectbox("Fréquence des données", ["Trimestrielle", "Mensuelle", "Annuelle"], index=0)
        forecast_years = st.slider("Années de prévision", 1, 5, 3)
    
    # Préparation des données
    if use_demo_data:
        df_raw = make_demo_data(periods=16, freq="Q" if freq == "Trimestrielle" else "M")
        mapping = auto_map_columns(df_raw)
    elif uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)
        mapping = auto_map_columns(df_raw)
        
        # Interface de mapping manuel
        st.sidebar.subheader("🎯 Mapping des Colonnes")
        for key in REQUIRED_BASE:
            available_cols = [None] + list(df_raw.columns)
            default_idx = 0
            if mapping.get(key) in df_raw.columns:
                default_idx = list(df_raw.columns).index(mapping[key]) + 1
            mapping[key] = st.sidebar.selectbox(
                f"Colonne pour {key}", 
                available_cols,
                index=default_idx
            )
    else:
        st.info("📊 Veuillez importer un fichier ou utiliser les données de démonstration")
        st.stop()
    
    # Application du mapping
    if mapping:
        rename_dict = {v: k for k, v in mapping.items() if v is not None}
        df = df_raw.rename(columns=rename_dict)
        df["date"] = _infer_date_col(df["date"])
        df = add_month_start(df)
        df_kpi = compute_kpis(df)
    
    # Métriques principales
    agg_global = aggregate_kpis(df_kpi, by=["date"]).sort_values("date")
    if not agg_global.empty:
        last_row = agg_global.iloc[-1]
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Primes Acquises", f"{last_row['earned_premium']:,.0f} €")
        col2.metric("Sinistres Encourus", f"{last_row['incurred_claims']:,.0f} €")
        col3.metric("Loss Ratio", f"{last_row['loss_ratio']*100:.1f}%")
        col4.metric("Combined Ratio", f"{last_row['combined_ratio']*100:.1f}%")
        if 'solvency_ratio' in last_row:
            col5.metric("Solvabilité", f"{last_row['solvency_ratio']*100:.1f}%")
    
    # Onglets d'analyse
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 KPI Dynamiques", "🔮 Prévisions", "🧪 Stress Tests", "🗂️ Structure Portefeuille", "📤 Export"])
    
    with tab1:
        st.subheader("📈 Analyse des KPI par Dimension")
        
        dimensions = []
        if "lob" in df_kpi.columns:
            dimensions.append("lob")
        if "region" in df_kpi.columns:
            dimensions.append("region")
        if "cedant" in df_kpi.columns:
            dimensions.append("cedant")
            
        selected_dims = st.multiselect("Regrouper par", dimensions, default=dimensions[:1] if dimensions else [])
        
        if selected_dims:
            grouped_data = aggregate_kpis(df_kpi, by=["date"] + selected_dims)
            
            # Sélecteur de KPI
            kpi_options = {
                "Loss Ratio": "loss_ratio",
                "Expense Ratio": "expense_ratio", 
                "Combined Ratio": "combined_ratio",
                "Operating Ratio": "operating_ratio",
                "Cession Ratio": "cession_ratio"
            }
            selected_kpi = st.selectbox("KPI à analyser", list(kpi_options.keys()))
            kpi_column = kpi_options[selected_kpi]
            
            fig = px.line(grouped_data, x="date", y=kpi_column, color=selected_dims[0], 
                         title=f"Évolution du {selected_kpi} par {selected_dims[0]}",
                         markers=True)
            st.plotly_chart(fig, use_container_width=True)
            
            # Heatmap des corrélations
            st.subheader("📊 Matrice de Corrélation")
            numeric_cols = grouped_data.select_dtypes(include=[np.number]).columns
            corr_matrix = grouped_data[numeric_cols].corr()
            fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                               title="Corrélations entre Variables Numériques")
            st.plotly_chart(fig_corr, use_container_width=True)
    
    with tab2:
        st.subheader("🔮 Prévisions SARIMAX")
        
        target_var = st.selectbox("Variable à prévoir", 
                                 ["earned_premium", "incurred_claims", "combined_ratio", "loss_ratio"])
        
        forecast_dim = st.selectbox("Dimension de prévision", 
                                   ["Global"] + [d for d in ["lob", "region"] if d in df_kpi.columns])
        
        def generate_forecast(data_subset, target, steps):
            """Génère les prévisions pour un sous-ensemble de données"""
            aggregated = aggregate_kpis(data_subset, by=["date"]).sort_values("date")
            if aggregated.empty:
                return pd.DataFrame()
                
            ts_data = aggregated.set_index("date")[target]
            
            # Déterminer le nombre de pas selon la fréquence
            if freq == "Trimestrielle":
                steps_calc = 4 * steps
            elif freq == "Mensuelle":
                steps_calc = 12 * steps
            else:  # Annuelle
                steps_calc = steps
                
            forecast = sarimax_forecast(ts_data, steps_calc)
            
            # Préparation des résultats
            historical = pd.DataFrame({
                'date': ts_data.index,
                'value': ts_data.values,
                'type': 'Historique'
            })
            
            future = pd.DataFrame({
                'date': forecast.index,
                'value': forecast.values,
                'type': 'Prévision'
            })
            
            return pd.concat([historical, future], ignore_index=True)
        
        if forecast_dim == "Global":
            forecast_data = generate_forecast(df_kpi, target_var, forecast_years)
            if not forecast_data.empty:
                fig_forecast = px.line(forecast_data, x='date', y='value', color='type',
                                     title=f"Prévision {target_var} - Global")
                st.plotly_chart(fig_forecast, use_container_width=True)
        else:
            unique_vals = df_kpi[forecast_dim].dropna().unique()
            for val in unique_vals:
                subset = df_kpi[df_kpi[forecast_dim] == val]
                forecast_data = generate_forecast(subset, target_var, forecast_years)
                if not forecast_data.empty:
                    fig_forecast = px.line(forecast_data, x='date', y='value', color='type',
                                         title=f"Prévision {target_var} - {forecast_dim}: {val}")
                    st.plotly_chart(fig_forecast, use_container_width=True)
    
    with tab3:
        st.subheader("🧪 Tests de Résistance (Stress Tests)")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            freq_shock = st.slider("Choc Fréquence (%)", -50, 200, 20)
        with col2:
            sev_shock = st.slider("Choc Sévérité (%)", -50, 300, 30)
        with col3:
            cat_event = st.slider("Événement CAT (multiplicateur)", 1.0, 10.0, 3.0)
        
        # Application des chocs
        df_stress = df_kpi.copy()
        
        if "claims_count" in df_stress.columns:
            df_stress["claims_count"] = df_stress["claims_count"] * (1 + freq_shock/100)
            
        df_stress["incurred_claims"] = df_stress["incurred_claims"] * (1 + sev_shock/100)
        
        # Application d'un événement CAT sur la dernière période
        last_date = df_stress["date"].max()
        cat_mask = df_stress["date"] == last_date
        df_stress.loc[cat_mask, "incurred_claims"] = df_stress.loc[cat_mask, "incurred_claims"] * cat_event
        
        # Comparaison baseline vs stress
        base_kpi = aggregate_kpis(df_kpi, by=["date"])
        stress_kpi = aggregate_kpis(df_stress, by=["date"])
        
        col1, col2 = st.columns(2)
        with col1:
            fig_base = px.line(base_kpi, x="date", y="combined_ratio", 
                             title="Combined Ratio - Baseline")
            st.plotly_chart(fig_base, use_container_width=True)
        with col2:
            fig_stress = px.line(stress_kpi, x="date", y="combined_ratio",
                               title="Combined Ratio - Stress Test")
            st.plotly_chart(fig_stress, use_container_width=True)
        
        # Impact sur la solvabilité
        if {"scr", "own_funds"}.issubset(df_kpi.columns):
            base_solv = base_kpi["own_funds"].sum() / base_kpi["scr"].sum()
            stress_solv = stress_kpi["own_funds"].sum() / stress_kpi["scr"].sum()
            
            st.metric("Ratio de Solvabilité Baseline", f"{base_solv:.2%}")
            st.metric("Ratio de Solvabilité Stress", f"{stress_solv:.2%}", 
                     delta=f"{(stress_solv - base_solv):.2%}")
    
    with tab4:
        st.subheader("🗂️ Structure du Portefeuille")
        
        # Répartition par LOB
        if "lob" in df_kpi.columns:
            lob_analysis = aggregate_kpis(df_kpi, by=["lob"])
            fig_lob = px.pie(lob_analysis, values="earned_premium", names="lob",
                           title="Répartition des Primes par Ligne de Business")
            st.plotly_chart(fig_lob, use_container_width=True)
        
        # Répartition géographique
        if "region" in df_kpi.columns:
            region_analysis = aggregate_kpis(df_kpi, by=["region"])
            fig_region = px.bar(region_analysis, x="region", y="earned_premium",
                              title="Primes par Région")
            st.plotly_chart(fig_region, use_container_width=True)
        
        # Analyse fréquence vs sévérité
        if {"frequency", "severity"}.issubset(df_kpi.columns):
            freq_sev_analysis = aggregate_kpis(df_kpi, by=["lob"] if "lob" in df_kpi.columns else ["region"])
            fig_scatter = px.scatter(freq_sev_analysis, x="frequency", y="severity",
                                   size="earned_premium", hover_name=freq_sev_analysis.index,
                                   title="Fréquence vs Sévérité par Segment")
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tab5:
        st.subheader("📤 Export des Données et Rapports")
        
        # Export CSV
        st.markdown("### 📊 Données Brutes avec KPI")
        st.dataframe(df_kpi.head(100))
        download_button(df_kpi, "donnees_reassurance_avec_kpi.csv")
        
        # Export agrégé
        st.markdown("### 📈 Données Agrégées")
        aggregated_data = aggregate_kpis(df_kpi, by=["date"])
        st.dataframe(aggregated_data)
        download_button(aggregated_data, "kpi_agreges.csv")
        
        # Rapport PDF (simplifié)
        st.markdown("### 📄 Rapport PDF")
        if st.button("Générer le Rapport d'Analyse"):
            # Simulation de génération de rapport
            st.success("📋 Rapport généré avec succès!")
            st.info("""
            **Contenu du rapport:**
            - Synthèse des KPI principaux
            - Analyse des tendances
            - Prévisions sur 3 ans
            - Tests de résistance
            - Recommandations stratégiques
            """)

# =============================================================================
# SECTION 11: CALCULATEURS AVANCÉS
# =============================================================================
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
            
            # Paramètres du portefeuille
            st.subheader("📊 Paramètres du Portefeuille")
            
            primes_portefeuille = st.number_input("Primes du portefeuille (€)", value=10000000)
            sinistres_attendus = st.number_input("Sinistres attendus (€)", value=7000000)
            volatilite_sinistres = st.slider("Volatilité des sinistres (%)", 10, 50, 25)
            capital_disponible = st.number_input("Capital disponible (€)", value=3000000)
            cout_capital = st.slider("Coût du capital (%)", 8, 15, 10)
        
        with col2:
            st.markdown("""
            <div class="theory-box">
            <h4>⚙️ Contraintes d'Optimisation</h4>
            <ul>
            <li>Ratio de solvabilité ≥ 100%</li>
            <li>Probabilité de ruine ≤ 0.5%</li>
            <li>Coût réassurance ≤ 15% des primes</li>
            <li>Rétention ≥ 500k€</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Lancement de l'optimisation
            if st.button("🚀 Lancer l'optimisation"):
                # Simulation d'optimisation
                st.subheader("📊 Résultats de l'Optimisation")
                
                resultats_opti = {
                    'Paramètre': ['Quote-Share optimal', 'Rétention optimale', 'Stop Loss priorité', 'Coût réassurance', 'SCR après réassurance', 'Gain en capital'],
                    'Valeur': ['25%', '750k€', '115% des primes', '12.5% des primes', '2.1M€', '450k€'],
                    'Impact': ['↘️ Coût -15%', '↗️ Protection +10%', '🛡️ Sécurité +20%', '💰 Économie 250k€', '📈 Solvabilité +25%', '📊 ROE +2.5%']
                }
                
                st.dataframe(pd.DataFrame(resultats_opti), use_container_width=True)
                
                # Graphique des gains
                gains_data = {
                    'Élément': ['Économie coût réassurance', 'Gain en capital libéré', 'Amélioration rentabilité', 'Réduction volatilité'],
                    'Montant (k€)': [250, 450, 180, 320]
                }
                
                fig_gains = px.bar(gains_data, x='Élément', y='Montant (k€)',
                                 title="Gains de l'Optimisation")
                st.plotly_chart(fig_gains, use_container_width=True)
    
    with tab2:
        st.subheader("💰 Analyse de Rentabilité par Ligne de Business")
        
        # Calculateur ROE par ligne
        lignes_business = st.multiselect("Lignes de business à analyser", 
                                       ['Auto', 'Habitation', 'Santé', 'RC Pro', 'Vie'],
                                       default=['Auto', 'Habitation'])
        
        if lignes_business:
            data_roe = []
            for ligne in lignes_business:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    primes = st.number_input(f"Primes {ligne} (€)", value=2000000, key=f"primes_{ligne}")
                with col2:
                    sinistres = st.number_input(f"Sinistres {ligne} (€)", value=1400000, key=f"sinistres_{ligne}")
                with col3:
                    capital_alloue = st.number_input(f"Capital alloué {ligne} (€)", value=800000, key=f"capital_{ligne}")
                
                resultat_technique = primes - sinistres
                roe = (resultat_technique / capital_alloue) * 100 if capital_alloue > 0 else 0
                
                data_roe.append({
                    'Ligne': ligne,
                    'Primes': primes,
                    'Sinistres': sinistres,
                    'Résultat Technique': resultat_technique,
                    'Capital Alloué': capital_alloue,
                    'ROE Technique': roe
                })
            
            df_roe = pd.DataFrame(data_roe)
            st.dataframe(df_roe, use_container_width=True)
            
            # Graphique ROE
            fig_roe = px.bar(df_roe, x='Ligne', y='ROE Technique', 
                           title="Rentabilité par Ligne de Business")
            st.plotly_chart(fig_roe, use_container_width=True)
            
            # Analyse de la performance
            roe_moyen = df_roe['ROE Technique'].mean()
            meilleure_ligne = df_roe.loc[df_roe['ROE Technique'].idxmax()]
            moins_rentable = df_roe.loc[df_roe['ROE Technique'].idxmin()]
            
            col_perf1, col_perf2, col_perf3 = st.columns(3)
            with col_perf1:
                st.metric("📈 ROE Moyen", f"{roe_moyen:.1f}%")
            with col_perf2:
                st.metric("🏆 Meilleure ligne", f"{meilleure_ligne['Ligne']} ({meilleure_ligne['ROE Technique']:.1f}%)")
            with col_perf3:
                st.metric("📉 Ligne à améliorer", f"{moins_rentable['Ligne']} ({moins_rentable['ROE Technique']:.1f}%)")

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