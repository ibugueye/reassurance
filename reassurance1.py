import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import math

# Configuration de la page
st.set_page_config(
    page_title="Plateforme Réassurance Professionnelle - BIGDAA MBA",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour une apparence professionnelle
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
st.markdown('<div class="main-header">🏛️ PLATEFORME PROFESSIONNELLE DE RÉASSURANCE</div>', unsafe_allow_html=True)
st.markdown("### *Application pédagogique pour le programme BIGDAA MBA - Maîtrise des concepts techniques*")

# Sidebar Navigation professionnelle
st.sidebar.image("https://via.placeholder.com/150x50/1f4e79/ffffff?text=BIGDAA-MBA", use_column_width=True)
st.sidebar.title("🔍 Navigation Technique")

section = st.sidebar.radio("Modules de Formation", [
    "🏠 Accueil & Fondamentaux",
    "📊 Écosystème de la Réassurance", 
    "📈 Traités Proportionnels",
    "⚡ Traités Non-Proportionnels",
    "💰 Tarification Technique",
    "📉 Comptabilité Technique Avancée",
    "🌪️ Gestion des Catastrophes",
    "🛡️ Solvabilité & Réglementation",
    "📋 Étude de Cas Professionnelle"
])

# =============================================================================
# SECTION 1: ACCUEIL & FONDAMENTAUX
# =============================================================================
if section == "🏠 Accueil & Fondamentaux":
    st.markdown('<div class="section-header">🎯 Fondamentaux de la Réassurance</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="concept-box">
        <h3>📖 Définition Professionnelle</h3>
        <p>La <b>réassurance</b> est une technique par laquelle un assureur (la cédante) transfère tout ou partie 
        des risques qu'il a assurés à un réassureur, contre le paiement d'une prime de réassurance.</p>
        <p><b>Double fonction</b> : Technique (transfert de risque) et Financière (lissage des résultats).</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="theory-box">
        <h3>🧠 Théorie : Le Principe de Mutualisation</h3>
        <p>La réassurance s'appuie sur la <b>loi des grands nombres</b> :</p>
        <div class="formula-box">
        σ_portefeuille = σ_risque / √n
        </div>
        <p>Où σ représente la volatilité et n le nombre de risques. En mutualisant, le réassureur réduit la variabilité des résultats.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric("📈 Marché Mondial 2024", "450 Md€", "+6.2% vs 2023")
        st.metric("🏛️ Réassureurs Tier 1", "25 sociétés", "~80% du marché")
        st.metric("📊 Croissance Annuelle", "4-6%", "Projection 2024-2028")
        
        st.markdown("""
        <div class="warning-box">
        <h4>⚠️ Importance Stratégique</h4>
        <p>La réassurance est un <b>outil de gestion du capital</b> essentiel pour :</p>
        <ul>
        <li>Protéger les fonds propres</li>
        <li>Améliorer la notation financière</li>
        <li>Permettre la croissance</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Concepts fondamentaux en tableau
    st.markdown("### 📚 Concepts Clés à Maîtriser")
    
    concepts_data = {
        'Concept': [
            'Cédante', 'Réassureur', 'Prime de Réassurance', 'Commission de Réassurance',
            'Rétention', 'Cession', 'Traités Facultatifs', 'Traités Obligatoires'
        ],
        'Définition': [
            'Compagnie d\'assurance qui transfère le risque',
            'Société qui accepte le risque cédé',
            'Prix payé par la cédante pour le transfert de risque',
            'Pourcentage de prime reversé pour frais d\'acquisition',
            'Part du risque conservée par la cédante',
            'Part du risque transférée au réassureur',
            'Négociés risque par risque',
            'Couverture automatique pour un portefeuille'
        ],
        'Impact Comptable': [
            'Compte 62 - Acceptations', 'Compte 61 - Cessions',
            'Charge de réassurance', 'Produit de réassurance',
            'Actif du bilan', 'Passif du bilan',
            'Comptabilisation individuelle', 'Comptabilisation globale'
        ]
    }
    
    st.dataframe(pd.DataFrame(concepts_data), use_container_width=True)

# =============================================================================
# SECTION 2: ÉCOSYSTÈME DE LA RÉASSURANCE
# =============================================================================
elif section == "📊 Écosystème de la Réassurance":
    st.markdown('<div class="section-header">🏢 Écosystème et Acteurs de la Réassurance</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🏛️ Acteurs du Marché", "🔄 Chaîne de Valeur", "📈 Dynamiques de Marché"])
    
    with tab1:
        st.markdown("""
        <div class="concept-box">
        <h3>🏛️ Architecture du Marché</h3>
        <p>Le marché de la réassurance est structuré en plusieurs niveaux :</p>
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
    
    with tab2:
        st.markdown("""
        <div class="theory-box">
        <h3>🔄 Processus de Souscription</h3>
        <p>La chaîne de valeur de la réassurance comprend 5 étapes clés :</p>
        </div>
        """, unsafe_allow_html=True)
        
        steps = {
            'Étape': ['1. Analyse du Risque', '2. Structuration', '3. Négociation', '4. Gestion', '5. Règlement'],
            'Activité': [
                'Évaluation technique du portefeuille',
                'Définition des traités et couvertures',
                'Détermination des primes et commissions',
                'Suivi et administration des traités',
                'Règlement des sinistres et commissions'
            ],
            'Outils': [
                'Modèles actuariels, Scorings',
                'Logiciels de pricing, Bases de données',
                'Placements, Contrats types',
                'Systèmes de gestion, Reporting',
                'Processus claims, Contrôles'
            ]
        }
        
        st.dataframe(pd.DataFrame(steps), use_container_width=True)
    
    with tab3:
        st.markdown("""
        <div class="concept-box">
        <h3>📈 Cycles du Marché</h3>
        <p>Le marché de la réassurance suit des cycles alternant entre <b>hard market</b> et <b>soft market</b>.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Simulation du cycle
        st.subheader("🔄 Simulateur de Cycle du Marché")
        
        phase_marche = st.select_slider("Phase du marché", 
                                       options=['Hard Market Fort', 'Hard Market', 'Transition', 'Soft Market', 'Soft Market Prononcé'],
                                       value='Transition')
        
        caracteristiques = {
            'Hard Market Fort': {'capacite': 'Très réduite', 'primes': '+++', 'conditions': 'Très restrictives', 'couleur': '#dc3545'},
            'Hard Market': {'capacite': 'Réduite', 'primes': '++', 'conditions': 'Restrictives', 'couleur': '#ffc107'},
            'Transition': {'capacite': 'Équilibrée', 'primes': 'Stables', 'conditions': 'Normales', 'couleur': '#28a745'},
            'Soft Market': {'capacite': 'Abondante', 'primes': '-', 'conditions': 'Souples', 'couleur': '#17a2b8'},
            'Soft Market Prononcé': {'capacite': 'Excédentaire', 'primes': '--', 'conditions': 'Très souples', 'couleur': '#6f42c1'}
        }
        
        phase = caracteristiques[phase_marche]
        
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("📦 Capacité", phase['capacite'])
        with col2: st.metric("💰 Niveau des Primes", phase['primes'])
        with col3: st.metric("📝 Conditions", phase['conditions'])

# =============================================================================
# SECTION 3: TRAITÉS PROPORTIONNELS - THÉORIE AVANCÉE
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
# SECTION 4: TRAITÉS NON-PROPORTIONNELS - THÉORIE AVANCÉE
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

# =============================================================================
# SECTION 6: COMPTABILITÉ TECHNIQUE AVANCÉE
# =============================================================================
elif section == "📉 Comptabilité Technique Avancée":
    st.markdown('<div class="section-header">📉 Comptabilité Technique Avancée - Principes et Applications</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="theory-box">
    <h3>🏛️ Cadre Réglementaire et Principes</h3>
    <p>La comptabilité technique des assureurs et réassureurs est régie par des principes spécifiques distincts 
    de la comptabilité générale, notamment dans le cadre de <b>Solvabilité II</b>.</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Provisions Techniques", "📈 Ratios Clés", "💰 Résultat Technique", "🛡️ Solvabilité II"])

# =============================================================================
# SECTION 7: GESTION DES CATASTROPHES
# =============================================================================
elif section == "🌪️ Gestion des Catastrophes":
    st.markdown('<div class="section-header">🌪️ Gestion des Risques Catastrophiques - Modélisation et Couverture</div>', unsafe_allow_html=True)

# =============================================================================
# SECTION 8: SOLVABILITÉ & RÉGLEMENTATION
# =============================================================================
elif section == "🛡️ Solvabilité & Réglementation":
    st.markdown('<div class="section-header">🛡️ Solvabilité II et Gestion du Capital</div>', unsafe_allow_html=True)

# =============================================================================
# SECTION 9: ÉTUDE DE CAS PROFESSIONNELLE
# =============================================================================
elif section == "📋 Étude de Cas Professionnelle":
    st.markdown('<div class="section-header">📋 Étude de Cas Professionnelle - Optimisation de Programme</div>', unsafe_allow_html=True)

# =============================================================================
# FOOTER PROFESSIONNEL
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
    "**Version Professionnelle 2.0**"
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