import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import math

# Configuration de la page
st.set_page_config(
    page_title="Plateforme Réassurance - BIGDAA MBA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .concept-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown('<div class="main-header">📊 PLATEFORME DE RÉASSURANCE - BIGDAA MBA</div>', unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Sections", [
    "🏠 Accueil & Concepts",
    "📈 Traités Proportionnels", 
    "⚡ Traités Non-Proportionnels",
    "💰 Calcul des Primes & Commissions",
    "📊 Comptabilité Technique",
    "🌪️ Gestion Catastrophes",
    "📋 Étude de Cas Complète"
])

# Données de démonstration
@st.cache_data
def load_sample_data():
    """Charge les données de démonstration"""
    dates = pd.date_range('2023-01-01', '2024-12-31', freq='M')
    n_periods = len(dates)
    
    data = {
        'date': dates,
        'prime_directe': np.random.uniform(1000000, 2000000, n_periods),
        'sinistres_directs': np.random.uniform(500000, 1500000, n_periods),
        'prime_cedee': np.random.uniform(200000, 500000, n_periods),
        'sinistres_cedes': np.random.uniform(100000, 400000, n_periods),
        'commission_reassurance': np.random.uniform(50000, 150000, n_periods)
    }
    return pd.DataFrame(data)

df = load_sample_data()

# SECTION ACCUEIL & CONCEPTS
if section == "🏠 Accueil & Concepts":
    st.header("🎯 Fondamentaux de la Réassurance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📖 Définition")
        st.markdown("""
        <div class="concept-box">
        La <b>réassurance</b> est « l'assurance des assureurs ». C'est le mécanisme par lequel une compagnie d'assurance 
        (la cédante) transfère une partie de ses risques à une autre compagnie (le réassureur) en échange du paiement d'une prime.
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("🎯 Objectifs")
        st.markdown("""
        - **Lissage des résultats** : Réduire la volatilité des sinistres
        - **Renforcement de la capacité** : Permettre de souscrire des risques plus importants
        - **Sécurisation financière** : Protéger les fonds propres
        - **Expertise technique** : Bénéficier du savoir-faire des réassureurs
        """)
    
    with col2:
        st.subheader("🔄 Types de Traités")
        st.markdown("""
        <div class="concept-box">
        <b>Traités Proportionnels</b> : Partage des primes et sinistres selon un pourcentage fixe
        - Quota-share (quote-part)
        - Surplus
        </div>
        
        <div class="concept-box">
        <b>Traités Non-Proportionnels</b> : Couverture déclenchée au-delà d'un certain montant de sinistres
        - Stop Loss (excédent de pertes)
        - XL (Excédent de sinistres)
        </div>
        """, unsafe_allow_html=True)
    
    # Schéma conceptuel
    st.subheader("📊 Schéma du Processus de Réassurance")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h4>🏢 Assuré</h4>
        <p>Paiement de prime → Contrat d'assurance → Indemnisation en cas de sinistre</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h4>📝 Compagnie Cédante</h4>
        <p>Paiement de prime → Contrat de réassurance → Remboursement partiel des sinistres</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
        <h4>🛡️ Réassureur</h4>
        <p>Acceptation du risque → Couverture technique → Paiement des sinistres cédés</p>
        </div>
        """, unsafe_allow_html=True)

# SECTION TRAITÉS PROPORTIONNELS
elif section == "📈 Traités Proportionnels":
    st.header("📈 Traités Proportionnels")
    
    st.markdown("""
    <div class="concept-box">
    <b>Principe</b> : Partage proportionnel des primes et des sinistres entre cédant et réassureur selon un pourcentage fixe.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Quota-Share (Quote-Part)")
        quote_part = st.slider("Pourcentage de cession (%)", 10, 90, 30)
        
        prime_directe = st.number_input("Prime directe totale (€)", value=1000000)
        sinistre_total = st.number_input("Sinistre total (€)", value=500000)
        
        # Calculs Quota-Share
        prime_cedee = prime_directe * quote_part / 100
        sinistre_cede = sinistre_total * quote_part / 100
        prime_conserve = prime_directe - prime_cedee
        sinistre_conserve = sinistre_total - sinistre_cede
        
        st.metric("Prime cédée", f"{prime_cedee:,.0f} €")
        st.metric("Sinistre cédé", f"{sinistre_cede:,.0f} €")
        
        # Graphique camembert
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Cédé au réassureur', 'Conservé par la cédante'],
            values=[prime_cedee, prime_conserve],
            hole=0.4
        )])
        fig_pie.update_layout(title="Répartition des Primes")
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("📈 Calcul du Surplus")
        st.markdown("""
        **Le surplus** permet de céder uniquement la partie des risques qui dépasse la rétention de la cédante.
        """)
        
        retention = st.number_input("Rétention cédante (€)", value=500000)
        capacite_ligne = st.number_input("Capacité par ligne (€)", value=1000000)
        nombre_lignes = st.number_input("Nombre de lignes", value=3, min_value=1, max_value=10)
        
        capital_assure = st.number_input("Capital assuré (€)", value=2000000)
        
        if capital_assure > retention:
            part_cedee = min(capital_assure - retention, capacite_ligne * nombre_lignes)
            pourcentage_cession = (part_cedee / capital_assure) * 100
            
            st.metric("Part cédée en surplus", f"{part_cedee:,.0f} €")
            st.metric("Taux de cession", f"{pourcentage_cession:.1f}%")
        
        # Tableau de répartition
        st.subheader("🔄 Répartition des Risques")
        data_surplus = {
            'Type': ['Rétention cédante', 'Surplus cédé', 'Total'],
            'Montant (€)': [retention, part_cedee, retention + part_cedee],
            'Pourcentage': [retention/capital_assure*100, part_cedee/capital_assure*100, 100]
        }
        st.dataframe(pd.DataFrame(data_surplus))

# SECTION TRAITÉS NON-PROPORTIONNELS
elif section == "⚡ Traités Non-Proportionnels":
    st.header("⚡ Traités Non-Proportionnels")
    
    st.markdown("""
    <div class="concept-box">
    <b>Principe</b> : Le réassureur n'intervient qu'au-delà d'un certain montant de sinistres (priorité) et jusqu'à une limite donnée.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📉 Stop Loss (Excédent de Pertes)")
        
        prime_totale = st.number_input("Prime totale du portefeuille (€)", value=5000000)
        priorite_stoploss = st.number_input("Priorité Stop Loss (€)", value=1000000)
        limite_stoploss = st.number_input("Limite Stop Loss (€)", value=2000000)
        sinistres_portefeuille = st.number_input("Sinistres du portefeuille (€)", value=2500000)
        
        # Calcul Stop Loss
        taux_activation = (sinistres_portefeuille / prime_totale) * 100
        franchise_effective = priorite_stoploss * prime_totale / 100 if st.checkbox("Priorité en % des primes") else priorite_stoploss
        
        sinistre_reassureur = max(0, min(limite_stoploss, sinistres_portefeuille - franchise_effective))
        
        st.metric("Taux de sinistres", f"{taux_activation:.1f}%")
        st.metric("Sinistre à charge réassureur", f"{sinistre_reassureur:,.0f} €")
        
        # Graphique Stop Loss
        categories = ['Prime totale', 'Franchise', 'Sinistres totaux', 'Part réassureur']
        valeurs = [prime_totale, franchise_effective, sinistres_portefeuille, sinistre_reassureur]
        
        fig_bar = px.bar(x=categories, y=valeurs, title="Répartition Stop Loss")
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        st.subheader("🌊 Couche XL (Excédent de Sinistre)")
        
        st.markdown("**Structure en couches** : Chaque réassureur prend un tranche de sinistres entre une priorité et une limite.")
        
        priorite_xl = st.number_input("Priorité XL (€)", value=500000)
        limite_xl = st.number_input("Limite XL (€)", value=1000000)
        sinistre_principal = st.number_input("Sinistre principal (€)", value=750000)
        
        # Calcul XL
        prise_reassureur = max(0, min(limite_xl, sinistre_principal - priorite_xl))
        reste_cedeant = sinistre_principal - prise_reassureur
        
        st.metric("Part réassureur XL", f"{prise_reassureur:,.0f} €")
        st.metric("Part cédante", f"{reste_cedeant:,.0f} €")
        
        # Diagramme en couches
        layers = {
            'Couche': ['1ère (Cédante)', '2ème (Réassureur)', '3ème (Cédante)'],
            'De': [0, priorite_xl, priorite_xl + limite_xl],
            'À': [priorite_xl, priorite_xl + limite_xl, priorite_xl + limite_xl + 500000],
            'Montant Sinistre': [min(priorite_xl, sinistre_principal), prise_reassureur, max(0, sinistre_principal - priorite_xl - limite_xl)]
        }
        
        st.dataframe(pd.DataFrame(layers))

# SECTION CALCUL DES PRIMES
elif section == "💰 Calcul des Primes & Commissions":
    st.header("💰 Calcul des Primes & Commissions")
    
    tab1, tab2, tab3 = st.tabs(["🎯 Prime Pure vs Prime Commerciale", "🔄 Commissions", "📈 Profit Commission"])
    
    with tab1:
        st.subheader("🎯 Composition de la Prime")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Prime Pure** : Couvre le coût moyen des sinistres
            **Prime de Risque** : Prime pure + chargement de sécurité
            **Prime Commerciale** : Prime de risque + frais + bénéfice
            """)
            
            # CORRECTION : Définir cout_moyen_sinistre AVANT de l'utiliser
            frequence_sinistres = st.slider("Fréquence sinistres (%)", 0.1, 10.0, 2.5)
            cout_moyen_sinistre = st.number_input("Coût moyen sinistre (€)", value=50000)  # DÉFINI ICI
            chargement_securite = st.slider("Chargement sécurité (%)", 5, 30, 15)
            frais_gestion = st.slider("Frais de gestion (%)", 10, 40, 25)
            marge_beneficiaire = st.slider("Marge bénéficiaire (%)", 5, 20, 10)
            
            # MAINTENANT cout_moyen_sinistre est défini
            prime_pure = frequence_sinistres/100 * cout_moyen_sinistre
            prime_risque = prime_pure * (1 + chargement_securite/100)
            prime_commerciale = prime_risque / (1 - (frais_gestion + marge_beneficiaire)/100)
        
        with col2:
            st.metric("Prime pure", f"{prime_pure:,.0f} €")
            st.metric("Prime de risque", f"{prime_risque:,.0f} €")
            st.metric("Prime commerciale", f"{prime_commerciale:,.0f} €")
            
            # Graphique de composition
            composition = {
                'Élément': ['Prime pure', 'Chargement sécurité', 'Frais gestion', 'Marge bénéficiaire'],
                'Valeur (€)': [
                    prime_pure,
                    prime_risque - prime_pure,
                    prime_commerciale * frais_gestion/100,
                    prime_commerciale * marge_beneficiaire/100
                ]
            }
            fig_composition = px.pie(composition, values='Valeur (€)', names='Élément', 
                                   title="Composition de la Prime Commerciale")
            st.plotly_chart(fig_composition, use_container_width=True)
    
    with tab2:
        st.subheader("🔄 Commissions de Réassurance")
        
        st.markdown("""
        <div class="concept-box">
        <b>Commission de réassurance</b> : Pourcentage de la prime cédée que le réassureur reverse à la cédante 
        pour couvrir ses frais d'acquisition et de gestion.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            prime_cedee = st.number_input("Prime cédée (€)", value=300000)
            taux_commission = st.slider("Taux de commission (%)", 10, 40, 25)
            commission = prime_cedee * taux_commission / 100
            
            st.metric("Commission versée", f"{commission:,.0f} €")
            st.metric("Prime nette réassureur", f"{prime_cedee - commission:,.0f} €")
        
        with col2:
            st.subheader("📊 Commission Variable")
            st.markdown("**Commission variable** : Ajustée en fonction des résultats du portefeuille.")
            
            ratio_sinistralite = st.slider("Ratio de sinistralité (%)", 50, 150, 85)
            taux_base = st.slider("Taux de base (%)", 20, 35, 25)
            participation_resultat = st.slider("Participation résultat (%)", 10, 50, 25)
            
            if ratio_sinistralite < 100:
                commission_variable = taux_base + (100 - ratio_sinistralite) * participation_resultat / 100
            else:
                commission_variable = max(0, taux_base - (ratio_sinistralite - 100) * participation_resultat / 100)
            
            st.metric("Commission variable", f"{commission_variable:.1f}%")

# SECTION COMPTABILITÉ TECHNIQUE
elif section == "📊 Comptabilité Technique":
    st.header("📊 Comptabilité Technique de Réassurance")
    
    st.markdown("""
    <div class="concept-box">
    <b>Comptabilité technique</b> : Ensemble des méthodes et principes comptables spécifiques aux assureurs et réassureurs 
    pour mesurer la performance technique de leurs opérations, distincte de la comptabilité générale.
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Ratios Techniques", "📉 Provisionnement", "💰 Résultat Technique", "📊 État Synthétique"])
    
    with tab1:
        st.subheader("📈 Ratios Techniques Clés")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            primes_emises = st.number_input("Primes émises (€)", value=5000000)
            sinistres_payes = st.number_input("Sinistres payés (€)", value=3000000)
            ratio_sinistralite = (sinistres_payes / primes_emises) * 100
            st.metric("Ratio de sinistralité", f"{ratio_sinistralite:.1f}%")
        
        with col2:
            frais_gestion = st.number_input("Frais de gestion (€)", value=1500000)
            ratio_frais = (frais_gestion / primes_emises) * 100
            st.metric("Ratio de frais", f"{ratio_frais:.1f}%")
        
        with col3:
            resultat_technique = primes_emises - sinistres_payes - frais_gestion
            ratio_combined = ratio_sinistralite + ratio_frais
            st.metric("Ratio combiné", f"{ratio_combined:.1f}%")
            st.metric("Résultat technique", f"{resultat_technique:,.0f} €")
    
    with tab2:
        st.subheader("📉 Provisionnement Technique")
        
        st.markdown("""
        **Provisions techniques** : Montants constitués pour faire face aux engagements futurs.
        - Provision pour sinistres à payer (PSAP)
        - Provision pour primes non acquises (PPNA)
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            sinistres_regles = st.number_input("Sinistres réglés (€)", value=2000000)
            sinistres_survenus = st.number_input("Sinistres survenus (€)", value=3500000)
            provision_sinistres = sinistres_survenus - sinistres_regles
            
            st.metric("Provision pour sinistres", f"{provision_sinistres:,.0f} €")
        
        with col2:
            primes_annee = st.number_input("Primes de l'année (€)", value=5000000)
            duree_moyenne_contrats = st.slider("Durée moyenne contrats (mois)", 1, 12, 6)
            ppna = primes_annee * (12 - duree_moyenne_contrats) / 12
            
            st.metric("Provision pour primes non acquises", f"{ppna:,.0f} €")

# SECTION GESTION CATASTROPHES
elif section == "🌪️ Gestion Catastrophes":
    st.header("🌪️ Gestion des Risques Catastrophiques")
    
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
        st.metric("Dommage total estimé", f"{dommage_estime:,.0f} €")
    
    with col2:
        st.subheader("📊 Couverture Catastrophe")
        
        priorite_cat = st.number_input("Priorité programme cat (€)", value=100000000)
        limite_cat = st.number_input("Limite programme cat (€)", value=200000000)
        
        prise_reassureur_cat = max(0, min(limite_cat, dommage_estime - priorite_cat))
        
        st.metric("Part cédante", f"{min(dommage_estime, priorite_cat):,.0f} €")
        st.metric("Part réassureurs", f"{prise_reassureur_cat:,.0f} €")
        
        # Graphique de couverture
        fig_cat = go.Figure(go.Waterfall(
            name="Répartition sinistre cat",
            orientation="v",
            measure=["relative", "relative", "total"],
            x=["Dommage total", "Priorité cédante", "Part réassureur"],
            y=[dommage_estime, -priorite_cat, -prise_reassureur_cat]
        ))
        st.plotly_chart(fig_cat, use_container_width=True)

# SECTION ÉTUDE DE CAS COMPLÈTE
elif section == "📋 Étude de Cas Complète":
    st.header("📋 Étude de Cas : Compagnie ASSURPRO")
    
    st.markdown("""
    <div class="concept-box">
    <b>Scénario</b> : ASSURPRO, compagnie d'assurance IARD, souhaite optimiser son programme de réassurance pour 2024.
    Portefeuille : 10M€ de primes, concentration sur risques incendie et responsabilité civile.
    </div>
    """, unsafe_allow_html=True)
    
    # Données du cas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        primes_totales = st.number_input("Primes totales ASSURPRO (€)", value=10000000)
        sinistres_attendus = st.number_input("Sinistres attendus (€)", value=6500000)
        frais_gestion = st.number_input("Frais de gestion (€)", value=2500000)
    
    with col2:
        retention_max = st.number_input("Rétention maximale (€)", value=500000)
        capacite_surplus = st.number_input("Capacité surplus (€)", value=2000000)
        nombre_lignes = st.number_input("Nombre lignes surplus", value=5)
    
    with col3:
        priorite_stoploss = st.number_input("Priorité stop loss (%)", value=110, min_value=100, max_value=130)
        limite_stoploss = st.number_input("Limite stop loss (€)", value=2000000)
    
    # Calcul du programme optimal
    if st.button("🚀 Calculer le programme optimal"):
        
        # Calcul Quota-Share
        quote_part_optimal = min(40, (primes_totales - retention_max) / primes_totales * 100)
        prime_cedee_qs = primes_totales * quote_part_optimal / 100
        
        # Calcul besoin surplus
        besoin_surplus = capacite_surplus * nombre_lignes
        
        # Calcul stop loss
        franchise_stoploss = primes_totales * priorite_stoploss / 100
        sinistre_limite = sinistres_attendus * 1.5  # Scénario défavorable
        prise_stoploss = max(0, min(limite_stoploss, sinistre_limite - franchise_stoploss))
        
        # Affichage résultats
        st.subheader("🎯 Programme de Réassurance Recommandé")
        
        resultats = {
            'Composante': ['Quota-Share', 'Surplus', 'Stop Loss', 'Total Cédé'],
            'Taux/Montant': [f"{quote_part_optimal:.1f}%", f"{besoin_surplus:,.0f} €", f"{limite_stoploss:,.0f} €", "-"],
            'Prime Cédée': [prime_cedee_qs, primes_totales * 0.1, primes_totales * 0.02, prime_cedee_qs + primes_totales * 0.12],
            'Couverture': [f"Risques standards", "Risques importants", "Protection résultat", "Protection complète"]
        }
        
        st.dataframe(pd.DataFrame(resultats))
        
        # Graphique de répartition
        repartition = {
            'Type': ['Rétention nette', 'Quota-Share', 'Surplus', 'Stop Loss'],
            'Valeur': [
                primes_totales * (1 - quote_part_optimal/100 - 0.12),
                prime_cedee_qs,
                primes_totales * 0.1,
                primes_totales * 0.02
            ]
        }
        
        fig_repartition = px.pie(repartition, values='Valeur', names='Type', 
                               title="Répartition du Portefeuille après Réassurance")
        st.plotly_chart(fig_repartition, use_container_width=True)
        
        # Impact sur le résultat
        st.subheader("📊 Impact sur la Rentabilité")
        
        resultat_avant = primes_totales - sinistres_attendus - frais_gestion
        sinistres_apres_reassurance = sinistres_attendus * (1 - quote_part_optimal/100)
        primes_apres_reassurance = primes_totales - prime_cedee_qs - primes_totales * 0.12
        
        resultat_apres = primes_apres_reassurance - sinistres_apres_reassurance - frais_gestion
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Résultat avant réassurance", f"{resultat_avant:,.0f} €")
        with col2:
            st.metric("Résultat après réassurance", f"{resultat_apres:,.0f} €", 
                     delta=f"{(resultat_apres - resultat_avant):,.0f} €")

# Footer
st.markdown("---")
st.markdown(
    "**Application développée pour le programme BIGDAA MBA** | "
    "Plateforme pédagogique sur la réassurance | "
    "© 2024"
)

# Fonctionnalités avancées dans la sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("🛠️ Outils Avancés")

if st.sidebar.button("📥 Exporter les données"):
    # Création d'un dataframe d'export
    export_df = pd.DataFrame({
        'Paramètre': ['Primes totales', 'Sinistres attendus', 'Rétention max', 'Quote-part optimal'],
        'Valeur': [primes_totales, sinistres_attendus, retention_max, quote_part_optimal]
    })
    
    # Conversion en CSV
    csv = export_df.to_csv(index=False)
    st.sidebar.download_button(
        label="📋 Télécharger CSV",
        data=csv,
        file_name="programme_reassurance.csv",
        mime="text/csv"
    )

if st.sidebar.button("🔄 Réinitialiser"):
    st.experimental_rerun()