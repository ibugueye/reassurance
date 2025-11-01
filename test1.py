# ============================================================
# Plateforme Complète de Réassurance - Théorie & Pratique
# ============================================================
# Auteur : Ibrahima Gueye
# Version : 1.0
# ============================================================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# -------------------- CONFIGURATION PAGE --------------------
st.set_page_config(
    page_title="Plateforme Réassurance - Théorie & Pratique",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- STYLE GLOBAL --------------------------
st.markdown("""
<style>
    .main-title{
        font-size:2.4rem;
        font-weight:800;
        color:#1f4e79;
        text-align:center;
        margin-bottom:0.2rem;
    }
    .main-sub{
        text-align:center;
        color:#2b5876;
        margin-bottom:2rem;
        font-size:1.1rem;
    }
    .card{
        background:#ffffff;
        border:1px solid #e6eef5;
        border-radius:14px;
        padding:1rem 1.25rem;
        box-shadow:0 1px 8px rgba(31,78,121,0.06);
        margin-bottom:0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# PAGE 1 : INTRODUCTION
# ============================================================
def page_introduction():
    st.markdown('<div class="main-title">Introduction à la réassurance</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-sub">Une approche stratégique & pragmatique de la gestion des risques</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="card"><b>Public :</b><br>Débutants, étudiants, professionnels</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><b>Objectif :</b><br>Vision claire et pratique</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="card"><b>Résultat :</b><br>Compréhension des mécanismes de transfert de risque</div>', unsafe_allow_html=True)

    st.write("""
    En tant que spécialiste de la réassurance régionale, j’ai collaboré avec des acteurs majeurs du marché.
    Ce module vise à vous offrir une compréhension **claire et pratique** de la réassurance en tant qu’outil de
    stabilité et de gestion du risque.
    """)

    st.info("""
    💡 La réassurance est un mécanisme de **partage du risque** entre assureurs et réassureurs
    pour préserver la solvabilité et la confiance du système financier.
    """)

    # Mini diagramme
    df_flow = pd.DataFrame({
        "source": ["Assureur", "Assureur", "Portefeuille"],
        "target": ["Sinistres courants", "Réassureur", "Assureur"],
        "value":  [60, 40, 100]
    })
    labels = ["Portefeuille", "Assureur", "Réassureur", "Sinistres courants"]
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    sankey_fig = go.Figure(data=[go.Sankey(
        node=dict(label=labels, pad=18, thickness=18, line=dict(color="#cfe0ee", width=1)),
        link=dict(
            source=[label_to_idx[s] for s in df_flow["source"]],
            target=[label_to_idx[t] for t in df_flow["target"]],
            value=df_flow["value"]
        )
    )])
    sankey_fig.update_layout(height=370, title="Flux simplifié du risque entre assureur et réassureur")
    st.plotly_chart(sankey_fig, use_container_width=True)


# ============================================================
# PAGE 2 : PRINCIPES FONDAMENTAUX
# ============================================================
def page_principes():
    st.markdown('<div class="main-title">Principes fondamentaux de la réassurance</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-sub">Comprendre les bases essentielles de la gestion du risque</div>', unsafe_allow_html=True)

    st.write("""
    La réassurance est un **contrat** par lequel une compagnie d’assurance transfère une partie de ses risques
    à un réassureur. Cela permet de **partager** et **d’absorber les chocs** financiers.
    """)

    st.subheader("1️⃣ Généralités")
    st.write("""
    - Le réassureur protège l’assureur contre des pertes élevées.  
    - L’assureur, dit **cédante**, peut alors augmenter sa capacité de souscription.  
    - La réassurance renforce la **solvabilité** et la **confiance** du marché.
    """)

    st.subheader("2️⃣ Principales formes")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **🟢 Réassurance proportionnelle**  
        - Partage primes & sinistres selon un pourcentage.  
        - Exemple : *Quota Share, Surplus*.  
        """)
    with c2:
        st.markdown("""
        **🔵 Réassurance non proportionnelle**  
        - Intervention du réassureur au-delà d’un seuil.  
        - Exemple : *Excess of Loss, Stop Loss*.  
        """)

    st.success("💡 La réassurance est un **filet de sécurité** indispensable au système assurantiel.")


# ============================================================
# PAGE 3 : TYPES DE CONTRATS
# ============================================================
def page_types_contrats():
    st.markdown('<div class="main-title">Types de contrats de réassurance</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-sub">Comparer les logiques proportionnelles et non proportionnelles</div>', unsafe_allow_html=True)

    st.write("""
    Deux grandes familles dominent la réassurance :
    - **Proportionnelle** : partage des primes et sinistres.  
    - **Non proportionnelle** : couverture des pertes au-delà d’un seuil.
    """)

    st.subheader("Réassurance proportionnelle")
    st.markdown("""
    Exemple : **Quota Share (40%)**
    - Prime totale : 1 000 000 €
    - Sinistre total : 600 000 €
    - Le réassureur prend 40 % soit 400 000 € de primes et 240 000 € de sinistres.
    """)

    st.subheader("Réassurance non proportionnelle")
    st.markdown("""
    Exemple : **Excess of Loss**
    - Rétention : 200 000 €
    - Sinistre : 750 000 €
    - Couverture réassureur : 550 000 € (au-delà du seuil).
    """)

    st.warning("🎯 La réassurance non proportionnelle agit comme un **pare-chocs financier**.")


# ============================================================
# PAGE 4 : ACTEURS DU MARCHÉ
# ============================================================
def page_acteurs_flux():
    st.markdown('<div class="main-title">Acteurs du marché & flux de réassurance</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-sub">Identifier les rôles et visualiser les flux financiers</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **🏢 Cédante (assureur)**  
        - Souscrit les risques et cède une partie à un réassureur.  
        **🌐 Courtier**  
        - Intermédiaire entre cédantes et réassureurs.  
        """)
    with col2:
        st.markdown("""
        **🏛️ Réassureur**  
        - Mutualise les risques de plusieurs cédantes.  
        **🔁 Rétrocessionnaire**  
        - Reçoit à son tour une partie du risque du réassureur.  
        """)

    labels = ["Assuré", "Cédante", "Courtier", "Réassureur", "Rétrocessionnaire"]
    fig = go.Figure(data=[go.Sankey(
        node=dict(label=labels, pad=18, thickness=18, line=dict(color="black", width=0.5)),
        link=dict(source=[0, 1, 1, 2, 3], target=[1, 2, 3, 3, 4], value=[100, 80, 80, 70, 50])
    )])
    fig.update_layout(height=450, title="Flux de primes et sinistres dans la chaîne de réassurance")
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE 5 : CAS PRATIQUES & SIMULATION
# ============================================================
def page_cas_pratiques():
    st.markdown('<div class="main-title">Cas pratiques & simulation</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-sub">Appliquer les principes de la réassurance</div>', unsafe_allow_html=True)

    st.write("Simulez les flux entre cédante et réassureur dans un contrat proportionnel :")
    prime_totale = st.number_input("Prime totale (€)", value=1_000_000)
    taux_cession = st.slider("Taux de cession (%)", 0, 100, 40)
    sinistre_total = st.number_input("Sinistre total (€)", value=600_000)

    prime_reassureur = prime_totale * taux_cession / 100
    sinistre_reassureur = sinistre_total * taux_cession / 100

    st.success(f"Prime cédée au réassureur : {prime_reassureur:,.0f} €")
    st.success(f"Sinistre pris en charge par le réassureur : {sinistre_reassureur:,.0f} €")

    st.subheader("Quiz rapide 🎓")
    q1 = st.radio("Dans un contrat proportionnel, le réassureur reçoit :", 
                  ["Une part des primes et sinistres", "Uniquement les sinistres", "Seulement les primes"])
    q2 = st.radio("Dans un contrat non proportionnel, il intervient :", 
                  ["Dès le premier euro", "Au-delà d’un seuil de pertes", "Sur tous les contrats"])
    score = 0
    if q1 == "Une part des primes et sinistres": score += 1
    if q2 == "Au-delà d’un seuil de pertes": score += 1
    if st.button("Vérifier mes réponses"):
        st.write(f"Votre score : {score}/2")


# ============================================================
# MENU LATÉRAL
# ============================================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/3/3b/Insurance_icon_blue.png", width=100)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller vers :", [
    "🏛️ Introduction",
    "📘 Principes fondamentaux",
    "📑 Types de contrats",
    "🌐 Acteurs & flux",
    "🧮 Cas pratiques & simulation"
])

if page == "🏛️ Introduction":
    page_introduction()
elif page == "📘 Principes fondamentaux":
    page_principes()
elif page == "📑 Types de contrats":
    page_types_contrats()
elif page == "🌐 Acteurs & flux":
    page_acteurs_flux()
else:
    page_cas_pratiques()
