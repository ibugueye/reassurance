import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import math

# Configuration de la page
st.set_page_config(
    page_title="Plateforme Complète de Réassurance - Théorie & Pratique",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
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

# Titre principal
st.markdown('<div class="main-header">🏛️ PLATEFORME COMPLÈTE DE RÉASSURANCE</div>', unsafe_allow_html=True)
st.markdown("### *Application pédagogique intégrant théorie, pratique et études de cas concrets*")

# Sidebar Navigation
st.sidebar.image("https://via.placeholder.com/150x50/1f4e79/ffffff?text=BIGDAA-MBA", use_column_width=True)
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
            <li><b>📊 Études de cas réels</b> avec analyses détaillées</li>
            <li><b>🎯 Outils professionnels</b> de simulation et d'analyse</li>
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
            <li><b>Consultants</b> en finance et assurance</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric("📈 Marché Mondial 2024", "450 Md€", "+6.2% vs 2023")
        st.metric("🏛️ Réassureurs Tier 1", "25 sociétés", "~80% du marché")
        st.metric("📊 Modules Disponibles", "10 sections", "100+ concepts")
        
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
    
    # Roadmap de l'application
    st.markdown("### 🗺️ Roadmap d'Apprentissage")
    
    roadmap_data = {
        'Étape': ['1. Fondamentaux', '2. Traités', '3. Tarification', '4. Comptabilité', '5. Études de Cas'],
        'Concepts': [
            'Définitions, acteurs, écosystème',
            'Proportionnels et non-proportionnels',
            'Prime pure, commerciale, commissions',
            'Provisions, ratios, Solvabilité II',
            'Cas réels, simulations, optimisations'
        ],
        'Durée Estimée': ['1h', '2h', '2h', '2h', '3h'],
        'Compétences Visées': [
            'Compréhension base',
            'Maîtrise techniques',
            'Calculs techniques',
            'Analyse financière',
            'Application pratique'
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
        st.markdown("""
        <div class="concept-box">
        <h3>🎯 Définition Professionnelle</h3>
        <p>La <b>réassurance</b> est une technique par laquelle un assureur (la cédante) transfère tout ou partie 
        des risques qu'il a assurés à un réassureur, contre le paiement d'une prime de réassurance.</p>
        <p><b>Double fonction</b> : Technique (transfert de risque) et Financière (lissage des résultats).</p>
        </div>
        """, unsafe_allow_html=True)
        
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
        <div class="theory-box">
        <h3>🧠 Théorie : Principe de Mutualisation</h3>
        <p>La réassurance s'appuie sur la <b>loi des grands nombres</b> :</p>
        <div class="formula-box">
        σ_portefeuille = σ_risque / √n
        </div>
        <p>Où σ représente la volatilité et n le nombre de risques. En mutualisant, le réassureur réduit la variabilité des résultats.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("""
        <div class="concept-box">
        <h3>🔄 Processus de Réassurance</h3>
        <p>Le cycle complet de la réassurance comprend 5 étapes principales :</p>
        </div>
        """, unsafe_allow_html=True)
        
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
                value=[100, 70, 30, 20, 50, 20]
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
            </div>
            """, unsafe_allow_html=True)

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
        
        with col2:
            # Simulation de risque
            st.subheader("🎲 Simulation de Risque")
            
            capital_assure = st.number_input("Capital assuré (€)", value=2500000, step=100000)
            
            if capital_assure <= retention:
                st.info("💰 Risque entièrement conservé - Pas de cession")
                part_cedee = 0
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
            <li>Dégradation soudaire de la sinistralité</li>
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
                y=[primes_portefeuille, -sinistres_reels, priorite_absolue, prise_reassureur, -sinistre_reste_cedeant]
            ))
            fig.update_layout(title="Analyse Stop Loss - Répartition des Flux")
            st.plotly_chart(fig, use_container_width=True)
    
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
            col_c1, col_c2 = st.columns(2)
            
            with col_c1:
                priorite = st.number_input(f"Priorité couche {i+1} (€)", 
                                         value=500000 * (i+1), 
                                         key=f"priorite_{i}")
            with col_c2:
                limite = st.number_input(f"Limite couche {i+1} (€)", 
                                       value=500000, 
                                       key=f"limite_{i}")
            
            couches_data.append({
                'Couche': f"XL {i+1}",
                'Priorité': priorite,
                'Limite': limite,
                'Priorité Cumulée': priorite_cumulee
            })
            priorite_cumulee += limite
        
        # Simulation de sinistre
        sinistre_xl = st.number_input("Montant du sinistre principal (€)", value=1200000)
        
        # Calcul des prises par couche
        st.subheader("📊 Répartition par Couche")
        
        resultats_couches = []
        sinistre_restant = sinistre_xl
        
        for couche in couches_data:
            if sinistre_restant <= couche['Priorité']:
                prise_couche = 0
            else:
                prise_couche = min(couche['Limite'], sinistre_restant - couche['Priorité'])
            
            resultats_couches.append({
                'Couche': couche['Couche'],
                'Plage de Couverture': f"{couche['Priorité']:,.0f} € - {couche['Priorité'] + couche['Limite']:,.0f} €",
                'Prise Réassureur': prise_couche,
                'Sinistre Restant': sinistre_restant - prise_couche
            })
            sinistre_restant -= prise_couche
        
        df_resultats = pd.DataFrame(resultats_couches)
        st.dataframe(df_resultats, use_container_width=True)

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
            frequence_sinistres = st.slider("Fréquence sinistres (%)", 0.1, 10.0, 2.5)
            cout_moyen_sinistre = st.number_input("Coût moyen sinistre (€)", value=50000)
            
            prime_pure = (frequence_sinistres / 100) * cout_moyen_sinistre
            
            st.metric("🎯 Prime pure calculée", f"{prime_pure:,.0f} €")
        
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
            st.subheader("📈 Distribution des Sinistres")
            
            lambda_poisson = st.slider("Paramètre λ (fréquence)", 0.1, 5.0, 2.0)
            mu_lognormal = st.slider("μ lognormal (€)", 9.0, 12.0, 10.5)
            sigma_lognormal = st.slider("σ lognormal", 0.1, 2.0, 1.0)
            
            # Simulation de la distribution
            n_simulations = 10000
            n_sinistres = np.random.poisson(lambda_poisson, n_simulations)
            couts_sinistres = np.random.lognormal(mu_lognormal, sigma_lognormal, n_simulations)
            
            fig_dist = px.histogram(couts_sinistres, nbins=50, 
                                  title="Distribution des Coûts de Sinistres")
            st.plotly_chart(fig_dist, use_container_width=True)
    
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
            
            # Calculateur PPNA
            st.subheader("📅 Calculateur PPNA")
            
            primes_annee = st.number_input("Primes de l'année (€)", value=5000000)
            duree_moyenne = st.slider("Durée moyenne contrats (mois)", 1, 12, 6)
            
            ppna = primes_annee * (12 - duree_moyenne) / 12
            
            st.metric("📅 Provision pour primes non acquises", f"{ppna:,.0f} €")
    
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
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Modélisation des Catastrophes")
        
        type_catastrophe = st.selectbox("Type de catastrophe", [
            "Séisme", "Ouragan", "Inondation", "Incendie", "Grêle"
        ])
        
        intensite = st.slider("Intensité", 1, 10, 7)
        zone_affectee = st.number_input("Zone affectée (km²)", value=5000)
        densite_construction = st.slider("Densité construction", 0.1, 1.0, 0.7)
        
        # Calcul dommages estimés
        dommage_base = {
            "Séisme": 500000000,
            "Ouragan": 300000000, 
            "Inondation": 200000000,
            "Incendie": 150000000,
            "Grêle": 80000000
        }
        
        dommage_estime = dommage_base[type_catastrophe] * intensite * densite_construction
        st.metric("💥 Dommage total estimé", f"{dommage_estime:,.0f} €")
    
    with col2:
        st.subheader("📊 Couverture Catastrophe")
        
        priorite_cat = st.number_input("Priorité programme cat (€)", value=100000000)
        limite_cat = st.number_input("Limite programme cat (€)", value=200000000)
        
        prise_reassureur_cat = max(0, min(limite_cat, dommage_estime - priorite_cat))
        
        st.metric("🛡️ Part cédante", f"{min(dommage_estime, priorite_cat):,.0f} €")
        st.metric("🤝 Part réassureurs", f"{prise_reassureur_cat:,.0f} €")
        
        # Graphique de couverture
        fig_cat = go.Figure(go.Waterfall(
            name="Répartition sinistre cat",
            orientation="v",
            measure=["relative", "relative", "total"],
            x=["Dommage total", "Priorité cédante", "Part réassureur"],
            y=[dommage_estime, -priorite_cat, -prise_reassureur_cat]
        ))
        fig_cat.update_layout(title="Répartition du Sinistre Catastrophe")
        st.plotly_chart(fig_cat, use_container_width=True)

# =============================================================================
# SECTION 8: SOLVABILITÉ & RÉGLEMENTATION
# =============================================================================
elif section == "🛡️ Solvabilité & Réglementation":
    st.markdown('<div class="section-header">🛡️ Solvabilité II et Cadre Réglementaire</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="theory-box">
    <h3>🏛️ Cadre Solvabilité II</h3>
    <p>Le cadre Solvabilité II repose sur <b>trois piliers</b> complémentaires pour assurer la stabilité 
    financière des assureurs et réassureurs en Europe.</p>
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
            <p>Capital requis pour absorber les chocs avec une probabilité de 99.5% sur un an.</p>
            
            <div class="formula-box">
            SCR = √(∑ρ_ij × SCR_i × SCR_j)
            </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Calculateur SCR simplifié
            st.subheader("🧮 Calculateur SCR Simplifié")
            
            module_souscription = st.number_input("Module souscription (€)", value=50000000)
            module_marche = st.number_input("Module marché (€)", value=30000000)
            module_contrepartie = st.number_input("Module contrepartie (€)", value=10000000)
            
            # Calcul SCR avec corrélations standard
            scr_calc = math.sqrt(
                module_souscription**2 + 
                module_marche**2 + 
                module_contrepartie**2 +
                0.5 * module_souscription * module_marche +
                0.25 * module_souscription * module_contrepartie +
                0.25 * module_marche * module_contrepartie
            )
            
            st.metric("🛡️ SCR Calculé", f"{scr_calc:,.0f} €")
        
        with col2:
            st.markdown("""
            <div class="theory-box">
            <h4>📈 Minimum Capital Requirement (MCR)</h4>
            <p>Niveau de capital minimum en dessous duquel l'autorité de contrôle intervient.</p>
            
            <div class="formula-box">
            MCR = Max(25% × SCR, MCR_plancher)
            </div>
            </div>
            """, unsafe_allow_html=True)
            
            capital_disponible = st.number_input("Capital disponible (€)", value=80000000)
            ratio_solvabilite = (capital_disponible / scr_calc) * 100
            
            st.metric("📊 Ratio de solvabilité", f"{ratio_solvabilite:.1f}%")
            
            if ratio_solvabilite >= 100:
                st.success("✅ Niveau de capital suffisant")
            elif ratio_solvabilite >= 80:
                st.warning("⚠️ Niveau de capital à surveiller")
            else:
                st.error("🚨 Niveau de capital insuffisant")

# =============================================================================
# SECTION 9: ÉTUDES DE CAS CONCRETS
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
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Analyse de la situation actuelle
            st.subheader("📈 Analyse Actuelle")
            
            primes_totales = 50000000
            sinistres_attendus = 35000000
            quote_part_actuelle = 30
            retention_actuelle = 500000
            
            prime_cedee_actuelle = primes_totales * quote_part_actuelle / 100
            sinistre_cede_actuel = sinistres_attendus * quote_part_actuelle / 100
            
            st.metric("💰 Prime cédée actuelle", f"{prime_cedee_actuelle:,.0f} €")
            st.metric("⚡ Sinistre cédé actuel", f"{sinistre_cede_actuel:,.0f} €")
        
        with col2:
            st.markdown("""
            <div class="theory-box">
            <h4>🎯 Objectifs d'Optimisation</h4>
            <ul>
            <li>Réduire le coût de la réassurance de 15%</li>
            <li>Maintenir un niveau de protection adéquat</li>
            <li>Améliorer le ratio combiné de 2 points</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Proposition d'optimisation
            st.subheader("🔄 Proposition d'Optimisation")
            
            nouvelle_retention = st.slider("Nouvelle rétention (€)", 500000, 2000000, 750000)
            nouveau_quote_part = st.slider("Nouveau quota-share (%)", 10, 40, 20)
            
            economie_prime = prime_cedee_actuelle - (primes_totales * nouveau_quote_part / 100)
            nouveau_sinistre_cede = sinistres_attendus * nouveau_quote_part / 100
            
            st.metric("💸 Économie sur primes", f"{economie_prime:,.0f} €")
            st.metric("📈 Nouveau sinistre cédé", f"{nouveau_sinistre_cede:,.0f} €")
    
    with tab2:
        st.subheader("🏠 Cas : Programme Catastrophe pour Assureur Habitation")
        
        st.markdown("""
        <div class="case-study-box">
        <h4>📖 Contexte</h4>
        <p><b>Assureur HabitatSecur</b> : Forte exposition aux risques naturels dans le Sud-Est de la France.
        Nécessite un programme catastrophe robuste pour protéger son portefeuille.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Analyse des expositions
        expositions = {
            'Risque': ['Inondation', 'Séisme', 'Tempête', 'Grêle'],
            'Exposition (M€)': [150, 80, 120, 60],
            'Probabilité Annuelle': ['1%/an', '0.5%/an', '2%/an', '3%/an'],
            'Pire Scénario (M€)': [45, 60, 35, 25]
        }
        
        st.dataframe(pd.DataFrame(expositions), use_container_width=True)
        
        # Simulation de scénario catastrophe
        st.subheader("🌪️ Simulation Scénario Catastrophe")
        
        scenario = st.selectbox("Scénario à simuler", [
            "Crue centennale Rhône",
            "Séisme modéré Nice", 
            "Tempête type 1999",
            "Grêle exceptionnelle"
        ])
        
        sinistre_scenario = {
            "Crue centennale Rhône": 35000000,
            "Séisme modéré Nice": 55000000,
            "Tempête type 1999": 28000000,
            "Grêle exceptionnelle": 18000000
        }
        
        sinistre = sinistre_scenario[scenario]
        priorite_programme = 10000000
        limite_programme = 40000000
        
        prise_cat = max(0, min(limite_programme, sinistre - priorite_programme))
        
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("💥 Sinistre scénario", f"{sinistre:,.0f} €")
        with col2: st.metric("🛡️ Part cédante", f"{min(sinistre, priorite_programme):,.0f} €")
        with col3: st.metric("🤝 Part réassureurs", f"{prise_cat:,.0f} €")

# =============================================================================
# SECTION 10: CALCULATEURS AVANCÉS
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
                    'Paramètre': ['Quote-Share optimal', 'Rétention optimale', 'Stop Loss priorité', 'Coût réassurance', 'SCR après réassurance'],
                    'Valeur': ['25%', '750k€', '115% des primes', '12.5% des primes', '2.1M€'],
                    'Impact': ['↘️ Coût -15%', '↗️ Protection +10%', '🛡️ Sécurité +20%', '💰 Économie 250k€', '📈 Solvabilité +25%']
                }
                
                st.dataframe(pd.DataFrame(resultats_opti), use_container_width=True)
    
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
                    'ROE Technique': roe
                })
            
            df_roe = pd.DataFrame(data_roe)
            st.dataframe(df_roe, use_container_width=True)
            
            # Graphique ROE
            fig_roe = px.bar(df_roe, x='Ligne', y='ROE Technique', 
                           title="Rentabilité par Ligne de Business")
            st.plotly_chart(fig_roe, use_container_width=True)

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
    """)

with col_f2:
    st.markdown("**🔍 Glossaire Technique**")
    st.markdown("""
    - Cédante / Réassureur
    - Traités / Facultatif
    - Prime / Commission
    - Rétention / Cession
    """)

with col_f3:
    st.markdown("**📞 Support Pédagogique**")
    st.markdown("""
    BIGDAA MBA - Programme Réassurance  
    📧 contact@bigdaa-mba.fr  
    🌐 www.bigdaa-mba.fr
    """)

st.markdown("---")
st.markdown(
    "**Application pédagogique développée pour le programme BIGDAA MBA** | "
    "© 2024 - Tous droits réservés | "
    "**Version Professionnelle 3.0**"
)

# =============================================================================
# FONCTIONNALITÉS AVANCÉES
# =============================================================================
st.sidebar.markdown("---")
st.sidebar.subheader("🛠️ Outils Professionnels")

if st.sidebar.button("📥 Exporter l'Analyse"):
    st.sidebar.success("Fonctionnalité d'export activée")

if st.sidebar.button("🔄 Réinitialiser les Données"):
    st.experimental_rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("**🔐 Session Utilisateur**")
st.sidebar.info("Connecté en tant que : Étudiant BIGDAA MBA")

# Fonction pour générer des rapports PDF (placeholder)
def generer_rapport_pdf():
    st.sidebar.success("Génération du rapport PDF...")

if st.sidebar.button("📄 Générer Rapport PDF"):
    generer_rapport_pdf()