# =============================================================================
# IMPORTS ET CONFIGURATION - DOIT ÊTRE EN PREMIER
# =============================================================================
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

# Configuration de la page - DOIT ÊTRE LA PREMIÈRE COMMANDE STREAMLIT
st.set_page_config(
    page_title="Plateforme de Réassurance - Théorie & Data Science",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CONSTANTES ET CONFIGURATION
# =============================================================================
PAGE_CONFIG = {
    "page_title": "Plateforme de Réassurance - Théorie & Data Science",
    "page_icon": "🏛️", 
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

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

# =============================================================================
# CLASSES DE GESTION DES DONNÉES
# =============================================================================
class DataProcessor:
    """Classe pour le traitement des données de réassurance"""
    
    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def aggregate_kpis(d: pd.DataFrame, by=["date"]) -> pd.DataFrame:
        """Agrège par dimensions et recalcule les KPI au niveau agrégé."""
        grp = d.groupby(by, dropna=False).agg({
            "gross_premium": "sum", "ceded_premium": "sum", "earned_premium": "sum",
            "incurred_claims": "sum", "paid_claims": "sum", "ibnr": "sum", "rbns": "sum",
            "acq_expense": "sum", "adm_expense": "sum", "investment_income": "sum",
            "claims_count": "sum", "exposure": "sum", "scr": "sum", "own_funds": "sum"
        }).reset_index()
        grp = DataProcessor.compute_kpis(grp)
        return grp

    @staticmethod
    def add_month_start(df: pd.DataFrame) -> pd.DataFrame:
        """Aligne les dates sur le début de mois."""
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        return out

class DataGenerator:
    """Classe pour générer des données de démonstration"""
    
    @staticmethod
    def make_demo_data(periods=16, seed=42, freq="Q"):
        """Jeu de données de démonstration."""
        rng = np.random.default_rng(seed)
        idx = pd.period_range("2022Q1", periods=periods, freq=freq).to_timestamp()
        lobs = ["Property Cat", "Casualty", "Vie", "Santé"]
        regions = ["EU", "NA", "Asia"]
        rows = []
        for dt in idx:
            for lob in lobs:
                for region in regions[:2]:
                    gwp = rng.normal(50, 8) * 100000
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

# =============================================================================
# CLASSES DE PRÉVISION
# =============================================================================
class ForecastEngine:
    """Moteur de prévision pour les données de réassurance"""
    
    @staticmethod
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

# =============================================================================
# CLASSES D'INTERFACE UTILISATEUR
# =============================================================================
class UIStyles:
    """Gestion des styles CSS et de l'interface utilisateur"""
    
    @staticmethod
    def load_css():
        """Charge les styles CSS personnalisés"""
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

class Navigation:
    """Gestion de la navigation de l'application"""
    
    SECTIONS = [
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
    ]
    
    @staticmethod
    def setup_sidebar():
        """Configure la barre latérale de navigation"""
        st.sidebar.title("🔍 Navigation")
        section = st.sidebar.radio("Modules", Navigation.SECTIONS)
        return section

class FileHandler:
    """Gestion des fichiers et téléchargements"""
    
    @staticmethod
    def download_button(df: pd.DataFrame, filename: str):
        """Lien de téléchargement CSV."""
        csv = df.to_csv(index=False).encode("utf-8")
        b64 = base64.b64encode(csv).decode()
        st.markdown(
            f'<a download="{filename}" href="data:file/csv;base64,{b64}">📥 Télécharger {filename}</a>',
            unsafe_allow_html=True
        )

# =============================================================================
# PAGES DE L'APPLICATION
# =============================================================================
class PageManager:
    """Gestionnaire central des pages de l'application"""
    
    def __init__(self):
        self.processor = DataProcessor()
        self.generator = DataGenerator()
        self.forecaster = ForecastEngine()
    
    def render_page(self, section):
        """Route vers la page appropriée en fonction de la section sélectionnée"""
        if section == "🏠 Accueil & Présentation":
            self._page_accueil()
        elif section == "🎓 Principes Ludiques":
            self._page_principes_ludique()
        elif section == "📝 Types de Contrats Ludiques":
            self._page_types_contrats_ludique()
        elif section == "🏛️ Acteurs & Flux Ludiques":
            self._page_acteurs_flux_ludique()
        elif section == "📚 Concepts Fondamentaux":
            self._page_concepts_fondamentaux()
        elif section == "📈 Traités Proportionnels":
            self._page_traites_proportionnels()
        elif section == "⚡ Traités Non-Proportionnels":
            self._page_traites_non_proportionnels()
        elif section == "💰 Tarification Technique":
            self._page_tarification_technique()
        elif section == "📊 Comptabilité Technique":
            self._page_comptabilite_technique()
        elif section == "🌪️ Gestion des Catastrophes":
            self._page_gestion_catastrophes()
        elif section == "🛡️ Solvabilité & Réglementation":
            self._page_solvabilite_reglementation()
        elif section == "📋 Études de Cas Concrets":
            self._page_etudes_cas_concrets()
        elif section == "📊 Analyse Data Science":
            self._page_analyse_data_science()
        elif section == "🧮 Calculateurs Avancés":
            self._page_calculateurs_avances()
    
    def _page_accueil(self):
        """Page d'accueil principale"""
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
            st.plotly_chart(fig, use_container_width=True)

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
            if st.button("📚 Commencer les Fondamentaux", use_container_width=True, type="primary"):
                st.session_state.current_page = "Principes Fondamentaux"
                st.rerun()
        
        with col_cta2:
            if st.button("🧮 Utiliser les Calculateurs", use_container_width=True):
                st.session_state.current_page = "Calculateurs Avancés"
                st.rerun()
        
        with col_cta3:
            if st.button("📊 Explorer les Données", use_container_width=True):
                st.session_state.current_page = "Analyse Data Science"
                st.rerun()

    def _page_principes_ludique(self):
        """Page des principes fondamentaux ludiques"""
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
        
        # Le reste du contenu de la page des principes ludiques...
        st.info("Page des principes ludiques - Contenu à implémenter")

    def _page_types_contrats_ludique(self):
        """Page des types de contrats ludiques"""
        st.title("📝 Types de Contrats de Réassurance")
        st.markdown("### *Découvrez la boîte à outils du réassureur*")
        st.info("Page des types de contrats ludiques - Contenu à implémenter")

    def _page_acteurs_flux_ludique(self):
        """Page des acteurs et flux ludiques"""
        st.markdown('<div class="main-header">🏛️ Acteurs du Marché & Flux de Réassurance</div>', unsafe_allow_html=True)
        st.markdown("### *Explorez l'écosystème et les interactions*")
        st.info("Page des acteurs et flux ludiques - Contenu à implémenter")

    def _page_concepts_fondamentaux(self):
        """Page des concepts fondamentaux"""
        st.markdown('<div class="section-header">📚 Concepts Fondamentaux de la Réassurance</div>', unsafe_allow_html=True)
        st.info("Page des concepts fondamentaux - Contenu à implémenter")

    def _page_traites_proportionnels(self):
        """Page des traités proportionnels"""
        st.markdown("### 📈 Traités Proportionnels - Théorie et Applications")
        st.info("Page des traités proportionnels - Contenu à implémenter")

    def _page_traites_non_proportionnels(self):
        """Page des traités non-proportionnels"""
        st.markdown('<div class="section-header">⚡ Traités Non-Proportionnels - Théorie et Applications</div>', unsafe_allow_html=True)
        st.info("Page des traités non-proportionnels - Contenu à implémenter")

    def _page_tarification_technique(self):
        """Page de tarification technique"""
        st.markdown('<div class="section-header">💰 Tarification Technique - Modèles et Méthodologies</div>', unsafe_allow_html=True)
        st.info("Page de tarification technique - Contenu à implémenter")

    def _page_comptabilite_technique(self):
        """Page de comptabilité technique"""
        st.markdown('<div class="section-header">📊 Comptabilité Technique - Principes et Applications</div>', unsafe_allow_html=True)
        st.info("Page de comptabilité technique - Contenu à implémenter")

    def _page_gestion_catastrophes(self):
        """Page de gestion des catastrophes"""
        st.markdown('<div class="section-header">🌪️ Gestion des Risques Catastrophiques</div>', unsafe_allow_html=True)
        st.info("Page de gestion des catastrophes - Contenu à implémenter")

    def _page_solvabilite_reglementation(self):
        """Page de solvabilité et réglementation"""
        st.markdown('<div class="section-header">🛡️ Solvabilité II - Cadre Réglementaire Complet</div>', unsafe_allow_html=True)
        st.info("Page de solvabilité et réglementation - Contenu à implémenter")

    def _page_etudes_cas_concrets(self):
        """Page d'études de cas concrets"""
        st.markdown('<div class="section-header">📋 Études de Cas Concrets - Applications Réelles</div>', unsafe_allow_html=True)
        st.info("Page d'études de cas concrets - Contenu à implémenter")

    def _page_analyse_data_science(self):
        """Page d'analyse data science"""
        st.markdown('<div class="section-header">📊 Analyse Data Science - KPI & Prévisions</div>', unsafe_allow_html=True)
        st.info("Page d'analyse data science - Contenu à implémenter")

    def _page_calculateurs_avances(self):
        """Page des calculateurs avancés"""
        st.markdown('<div class="section-header">🧮 Calculateurs Avancés - Outils Professionnels</div>', unsafe_allow_html=True)
        st.info("Page des calculateurs avancés - Contenu à implémenter")

# =============================================================================
# APPLICATION PRINCIPALE
# =============================================================================
class ReassuranceApp:
    """Classe principale de l'application de réassurance"""
    
    def __init__(self):
        self.page_manager = PageManager()
        self.navigation = Navigation()
        self.ui_styles = UIStyles()
    
    def setup(self):
        """Configure l'application"""
        # La configuration de la page est déjà faite au début du script
        self.ui_styles.load_css()
    
    def run(self):
        """Lance l'application"""
        self.setup()
        
        # Navigation
        section = self.navigation.setup_sidebar()
        
        # Affichage de la page sélectionnée
        self.page_manager.render_page(section)
        
        # Footer
        self._render_footer()
    
    def _render_footer(self):
        """Affiche le pied de page"""
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
# POINT D'ENTRÉE
# =============================================================================
if __name__ == "__main__":
    app = ReassuranceApp()
    app.run()
