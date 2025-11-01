# ================================================================
# 🏛️ Plateforme Intégrée de Réassurance – Théorie, Pratique & KPI
# ================================================================
# Application Streamlit unifiée :
# - Théorie & formation (10 modules pédagogiques)
# - Analyse KPI, prévisions 3 ans, stress tests, export PDF
# - Design harmonisé & professionnel
# ---------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import math
import io, base64
from datetime import datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm

# ================================================================
# CONFIGURATION GÉNÉRALE
# ================================================================
st.set_page_config(
    page_title="Plateforme Intégrée de Réassurance",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================
# STYLE CSS GLOBAL
# ================================================================
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
    .case-study-box {
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
</style>
""", unsafe_allow_html=True)

# ================================================================
# EN-TÊTE GÉNÉRAL
# ================================================================
st.markdown('<div class="main-header">🏛️ PLATEFORME INTÉGRÉE DE RÉASSURANCE</div>', unsafe_allow_html=True)
st.markdown("### *Théorie, Pratique, Analyse et Décision – Une application pour apprendre et piloter la réassurance*")

# ================================================================
# SIDEBAR NAVIGATION
# ================================================================
st.sidebar.image("https://via.placeholder.com/150x50/1f4e79/ffffff?text=BIGDAA-MBA", use_container_width=True)
st.sidebar.title("🔍 Navigation")

section = st.sidebar.radio("Sections principales", [
    "🏠 Accueil & Présentation",
    "🎓 Théorie & Formation",
    "📈 Analyse & KPI Réassurance",
    "📚 Ressources & Support"
])

# ================================================================
# =====================  MODULE 1 : ACCUEIL  ======================
# ================================================================
if section == "🏠 Accueil & Présentation":
    st.markdown('<div class="section-header">🎯 Présentation Générale</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown("""
        <div class="concept-box">
        <h3>📖 Objectifs de la Plateforme</h3>
        <p>Cette application combine la rigueur de la réassurance professionnelle avec une approche pédagogique complète :</p>
        <ul>
            <li><b>📚 Théorie structurée</b> couvrant tous les concepts essentiels</li>
            <li><b>📊 Outils analytiques interactifs</b> : KPI, prévisions, stress tests</li>
            <li><b>🧮 Calculateurs professionnels</b> intégrés</li>
            <li><b>📄 Export PDF</b> et rapport exécutif automatique</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.metric("📈 Marché mondial 2024", "450 Md€", "+6.2%")
        st.metric("🏛️ Réassureurs Tier 1", "25 sociétés", "~80% du marché")
        st.metric("🎓 Modules disponibles", "10 modules pédagogiques + 1 outil KPI")

# ================================================================
# =====================  MODULE 2 : THEORIE  ======================
# ================================================================
elif section == "🎓 Théorie & Formation":
    st.markdown('<div class="section-header">🎓 Formation Complète en Réassurance</div>', unsafe_allow_html=True)
    sous_section = st.sidebar.selectbox("Sous-modules pédagogiques", [
        "📚 Concepts Fondamentaux",
        "📈 Traités Proportionnels",
        "⚡ Traités Non-Proportionnels",
        "💰 Tarification Technique",
        "📊 Comptabilité Technique",
        "🌪️ Gestion des Catastrophes",
        "🛡️ Solvabilité & Réglementation",
        "📋 Études de Cas"
    ])
    st.write(f"### {sous_section}")

    # --- Intégration synthétique des contenus du premier code ---
    if sous_section == "📚 Concepts Fondamentaux":
        st.markdown("""
        <div class="concept-box">
        <h4>Définition</h4>
        La réassurance est un mécanisme de transfert de risque d’un assureur (cédante) vers un réassureur.
        </div>
        """, unsafe_allow_html=True)
    elif sous_section == "📈 Traités Proportionnels":
        st.markdown("Les traités proportionnels partagent primes et sinistres selon un pourcentage fixe (Quote-part, Surplus).")
    elif sous_section == "⚡ Traités Non-Proportionnels":
        st.markdown("Les traités non-proportionnels (Stop Loss, XL) déclenchent la couverture au-delà d’un seuil (priorité).")
    elif sous_section == "💰 Tarification Technique":
        st.markdown("La tarification combine statistique, actuariat et expérience du risque. Formule : Prime pure × (1 + chargements).")
    elif sous_section == "📊 Comptabilité Technique":
        st.markdown("Les provisions techniques incluent PSAP, PPNA, PRA. Ratios : sinistralité, frais, combiné.")
    elif sous_section == "🌪️ Gestion des Catastrophes":
        st.markdown("La modélisation Cat combine hazard models + données d’exposition + vulnérabilité.")
    elif sous_section == "🛡️ Solvabilité & Réglementation":
        st.markdown("Solvabilité II repose sur 3 piliers : quantitatif (SCR), qualitatif (gouvernance), reporting (transparence).")
    elif sous_section == "📋 Études de Cas":
        st.markdown("Cas pratiques : Auto, Habitation, Réassureur global. Analyse de la rentabilité et optimisation du programme.")

# ================================================================
# ==========  MODULE 3 : ANALYSE KPI & OUTILS PRO  ===============
# ================================================================
elif section == "📈 Analyse & KPI Réassurance":
    st.markdown('<div class="section-header">📈 Analyse & KPI Réassurance</div>', unsafe_allow_html=True)

    # ===================== Chargement Données =====================
    st.sidebar.subheader("📥 Import / Données")
    up = st.sidebar.file_uploader("Importer CSV/Excel", type=["csv", "xlsx"])
    demo = st.sidebar.toggle("Utiliser données démo", value=(up is None))

    def make_demo(periods=16, seed=42):
        rng = np.random.default_rng(seed)
        idx = pd.period_range("2022Q1", periods=periods, freq="Q").to_timestamp()
        data = []
        for dt in idx:
            gwp = rng.normal(50, 8)
            ced = gwp * rng.uniform(0.2, 0.4)
            ep = gwp * rng.uniform(0.8, 0.95)
            inc = gwp * rng.uniform(0.5, 0.8)
            paid = inc * rng.uniform(0.7, 0.9)
            acq = ep * rng.uniform(0.1, 0.15)
            adm = ep * rng.uniform(0.05, 0.1)
            inv = gwp * 0.02
            scr = ep * 0.35
            own = scr * 1.5
            data.append([dt, gwp, ced, ep, inc, paid, acq, adm, inv, scr, own])
        return pd.DataFrame(data, columns=[
            "date", "gross_premium", "ceded_premium", "earned_premium", "incurred_claims",
            "paid_claims", "acq_expense", "adm_expense", "investment_income", "scr", "own_funds"
        ])

    if demo:
        df = make_demo()
    elif up:
        df = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
    else:
        st.stop()

    # ===================== KPI Calculs =====================
    def compute_kpis(df):
        ep = df["earned_premium"]
        df["loss_ratio"] = df["incurred_claims"] / ep
        df["expense_ratio"] = (df["acq_expense"] + df["adm_expense"]) / ep
        df["combined_ratio"] = df["loss_ratio"] + df["expense_ratio"]
        df["operating_ratio"] = df["combined_ratio"] - (df["investment_income"] / ep)
        df["solvency_ratio"] = df["own_funds"] / df["scr"]
        return df

    df = compute_kpis(df)
    st.metric("📊 Dernier Combined Ratio", f"{df['combined_ratio'].iloc[-1]*100:.1f}%")

    # ===================== Graphique KPI =====================
    fig = px.line(df, x="date", y=["loss_ratio", "expense_ratio", "combined_ratio"],
                  markers=True, title="Évolution des ratios techniques")
    st.plotly_chart(fig, use_container_width=True)

    # ===================== Prévision SARIMAX =====================
    st.markdown("### 🔮 Prévisions (SARIMAX, 3 ans)")
    ts = df.set_index("date")["combined_ratio"]
    model = SARIMAX(ts, order=(1,1,1), seasonal_order=(0,1,1,4))
    res = model.fit(disp=False)
    fc = res.get_forecast(steps=12).predicted_mean
    fc_df = pd.DataFrame({"date": pd.date_range(ts.index[-1], periods=12, freq="Q"), "prévision": fc})
    fig_fc = px.line(fc_df, x="date", y="prévision", title="Prévision du Combined Ratio (3 ans)")
    st.plotly_chart(fig_fc, use_container_width=True)

    # ===================== Export PDF =====================
    st.markdown("### 📄 Export PDF Rapport Exécutif")
    def build_pdf(df):
        buf = io.BytesIO()
        c = pdf_canvas.Canvas(buf, pagesize=A4)
        width, height = A4
        y = height - 2*cm
        c.setFont("Helvetica-Bold", 14)
        c.drawString(2*cm, y, "Rapport Exécutif - Réassurance")
        y -= 30
        last = df.iloc[-1]
        for k, v in [
            ("Primes acquises", f"{last['earned_premium']:,.0f} €"),
            ("Sinistres encourus", f"{last['incurred_claims']:,.0f} €"),
            ("Loss Ratio", f"{last['loss_ratio']*100:.1f}%"),
            ("Expense Ratio", f"{last['expense_ratio']*100:.1f}%"),
            ("Combined", f"{last['combined_ratio']*100:.1f}%"),
            ("Solvabilité", f"{last['solvency_ratio']*100:.1f}%")
        ]:
            c.drawString(2*cm, y, f"- {k} : {v}")
            y -= 15
        c.showPage()
        c.save()
        buf.seek(0)
        return buf.read()

    if st.button("📄 Générer PDF"):
        pdf_bytes = build_pdf(df)
        st.download_button("Télécharger Rapport PDF", data=pdf_bytes,
                           file_name="rapport_reassurance.pdf", mime="application/pdf")

# ================================================================
# =====================  MODULE 4 : SUPPORT  =====================
# ================================================================
elif section == "📚 Ressources & Support":
    st.markdown('<div class="section-header">📚 Ressources et Support</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **📖 Références :**
        - Code des Assurances
        - Directive Solvabilité II
        - IFRS 17
        - Principes Actuariels & Traités
        """)
    with col2:
        st.markdown("""
        **📞 Contact :**
        Programme BIGDAA MBA – Réassurance  
        ✉️ contact@bigdaa-mba.fr  
        🌐 www.bigdaa-mba.fr
        """)

# ================================================================
# =========================== FOOTER =============================
# ================================================================
st.markdown("---")
st.markdown("""
<div style="text-align:center;">
<b>© 2025 — Plateforme Intégrée de Réassurance</b><br>
Développée par <b>Amiharbi Eyeug</b> — Ingénierie, Data Science & Réassurance<br>
Programme BIGDAA MBA | Version 4.0
</div>
""", unsafe_allow_html=True)
