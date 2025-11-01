import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="Analyse Réassurance SEN-RE 2018-2022",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .kpi-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .positive {
        color: #00a86b;
        font-weight: bold;
    }
    .negative {
        color: #ff4444;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.5rem;
        color: #1f77b4;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    .norme-conforme {
        background-color: #d4edda;
        padding: 5px;
        border-radius: 5px;
        color: #155724;
    }
    .norme-alerte {
        background-color: #fff3cd;
        padding: 5px;
        border-radius: 5px;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

class ReinsuranceMultiYearAnalyzer:
    def __init__(self):
        self.years = [2018, 2019, 2020, 2021, 2022]
        self.load_multi_year_data()
        self.calculate_multi_year_ratios()
    
    def load_multi_year_data(self):
        """Chargement des données financières pour 2018-2022"""
        
        # Données du bilan 2018-2022
        self.balance_sheets = {
            2022: {
                'actif_total': 67736411528,
                'capitaux_propres': 12738350066,
                'provisions_techniques': 35157205175,
                'actif_circulant': 40497600164,
                'disponibilites': 4193211013,
                'creances_reassurance': 18741397700,
                'titres_participations': 5451434971,
                'dettes_court_terme': 16034315935
            },
            2021: {
                'actif_total': 63847390546,
                'capitaux_propres': 11805273655,
                'provisions_techniques': 34158229153,
                'actif_circulant': 39212619761,
                'disponibilites': 8612944474,
                'creances_reassurance': 17378888189,
                'titres_participations': 4084819617,
                'dettes_court_terme': 14470724315
            },
            2020: {
                'actif_total': 66140090540,
                'capitaux_propres': 11573517700,
                'provisions_techniques': 34995446739,
                'actif_circulant': 36791871917,
                'disponibilites': 5550216908,
                'creances_reassurance': 16786384349,
                'titres_participations': 5684594263,
                'dettes_court_terme': 16525949801
            },
            2019: {
                'actif_total': 61183312126,
                'capitaux_propres': 10898977425,
                'provisions_techniques': 31891692770,
                'actif_circulant': 34788951928,
                'disponibilites': 5082181152,
                'creances_reassurance': 15031494835,
                'titres_participations': 6455099576,
                'dettes_court_terme': 15088990837
            },
            2018: {
                'actif_total': 53925751172,
                'capitaux_propres': 9513876089,
                'provisions_techniques': 24906680390,
                'actif_circulant': 29172515536,
                'disponibilites': 3906891625,
                'creances_reassurance': 13657388227,
                'titres_participations': 8263745235,
                'dettes_court_terme': 15694383412
            }
        }
        
        # Données du compte de résultat 2018-2022
        self.income_statements = {
            2022: {
                'primes_vie': 1900606200,
                'primes_iard': 17805149081,
                'resultat_net': 1329281350,
                'sinistres_vie': 926401679,
                'sinistres_iard': 7051922528,
                'commissions_vie': 563041494,
                'commissions_iard': 5127080154,
                'charges_autres': 5106896886,
                'revenus_placements': 874241973,
                'produits_exceptionnels': 186419638
            },
            2021: {
                'primes_vie': 2623788596,
                'primes_iard': 15510350518,
                'resultat_net': 933076411,
                'sinistres_vie': 1357543856,
                'sinistres_iard': 6999464358,
                'commissions_vie': 756780438,
                'commissions_iard': 4064246848,
                'charges_autres': 4455329159,
                'revenus_placements': 729932706,
                'produits_exceptionnels': 66158250
            },
            2020: {
                'primes_vie': 1944213771,
                'primes_iard': 15201136511,
                'resultat_net': 565089288,
                'sinistres_vie': 37105296,
                'sinistres_iard': 9491807467,
                'commissions_vie': 444453170,
                'commissions_iard': 4198792684,
                'charges_autres': 3113888164,
                'revenus_placements': 686601005,
                'produits_exceptionnels': 313283482
            },
            2019: {
                'primes_vie': 1310741199,
                'primes_iard': 15033359616,
                'resultat_net': 723289746,
                'sinistres_vie': 741828754,
                'sinistres_iard': 9232090737,
                'commissions_vie': 222207252,
                'commissions_iard': 3950348971,
                'charges_autres': 1893519556,
                'revenus_placements': 739855397,
                'produits_exceptionnels': 26286453
            },
            2018: {
                'primes_vie': 545474423,
                'primes_iard': 13243710786,
                'resultat_net': 1385101336,
                'sinistres_vie': 386307915,
                'sinistres_iard': 7762552917,
                'commissions_vie': 293579152,
                'commissions_iard': 3842328003,
                'charges_autres': 1734749433,
                'revenus_placements': 918042219,
                'produits_exceptionnels': 1143385428
            }
        }
    
    def calculate_multi_year_ratios(self):
        """Calcul des ratios techniques et financiers pour toutes les années"""
        self.ratios = {}
        
        for year in self.years:
            bs = self.balance_sheets[year]
            is_stmt = self.income_statements[year]
            
            ratios_year = {}
            
            # Ratios de rentabilité
            ratios_year['roe'] = (is_stmt['resultat_net'] / bs['capitaux_propres']) * 100
            ratios_year['roa'] = (is_stmt['resultat_net'] / bs['actif_total']) * 100
            ratios_year['marge_nette'] = (is_stmt['resultat_net'] / (is_stmt['primes_vie'] + is_stmt['primes_iard'])) * 100
            
            # Ratios de solvabilité
            ratios_year['solvabilite'] = (bs['capitaux_propres'] / bs['actif_total']) * 100
            ratios_year['current_ratio'] = bs['actif_circulant'] / bs['dettes_court_terme']
            ratios_year['couverture_provisions'] = (bs['actif_circulant'] / bs['provisions_techniques']) * 100
            
            # Ratios techniques Vie
            primes_vie = is_stmt['primes_vie']
            if primes_vie > 0:
                ratios_year['ratio_sinistres_vie'] = (is_stmt['sinistres_vie'] / primes_vie) * 100
                ratios_year['ratio_commissions_vie'] = (is_stmt['commissions_vie'] / primes_vie) * 100
                ratios_year['combined_ratio_vie'] = ratios_year['ratio_sinistres_vie'] + ratios_year['ratio_commissions_vie']
            else:
                ratios_year['ratio_sinistres_vie'] = 0
                ratios_year['ratio_commissions_vie'] = 0
                ratios_year['combined_ratio_vie'] = 0
            
            # Ratios techniques IARD
            primes_iard = is_stmt['primes_iard']
            if primes_iard > 0:
                ratios_year['ratio_sinistres_iard'] = (is_stmt['sinistres_iard'] / primes_iard) * 100
                ratios_year['ratio_commissions_iard'] = (is_stmt['commissions_iard'] / primes_iard) * 100
                ratios_year['combined_ratio_iard'] = ratios_year['ratio_sinistres_iard'] + ratios_year['ratio_commissions_iard']
            else:
                ratios_year['ratio_sinistres_iard'] = 0
                ratios_year['ratio_commissions_iard'] = 0
                ratios_year['combined_ratio_iard'] = 0
            
            # Ratio de rendement des placements
            portefeuille_investissement = bs['titres_participations'] + bs['disponibilites']
            if portefeuille_investissement > 0:
                ratios_year['rendement_placements'] = (is_stmt['revenus_placements'] / portefeuille_investissement) * 100
            else:
                ratios_year['rendement_placements'] = 0
            
            # Ratio de croissance des primes
            if year > 2018:
                prev_year = year - 1
                prev_primes_total = (self.income_statements[prev_year]['primes_vie'] + 
                                   self.income_statements[prev_year]['primes_iard'])
                current_primes_total = is_stmt['primes_vie'] + is_stmt['primes_iard']
                ratios_year['croissance_primes'] = ((current_primes_total - prev_primes_total) / prev_primes_total) * 100
            else:
                ratios_year['croissance_primes'] = 0
            
            self.ratios[year] = ratios_year

def format_currency(amount):
    """Formate les montants en millions ou milliards"""
    if abs(amount) >= 1000000000:
        return f"{amount/1000000000:,.2f} Md FCFA"
    elif abs(amount) >= 1000000:
        return f"{amount/1000000:,.2f} M FCFA"
    else:
        return f"{amount:,.0f} FCFA"

def main():
    st.markdown('<h1 class="main-header">📊 Tableau de Bord Analytique SEN-RE 2018-2022</h1>', unsafe_allow_html=True)
    
    # Initialisation de l'analyseur
    analyzer = ReinsuranceMultiYearAnalyzer()
    
    # Sidebar avec informations générales
    st.sidebar.title("ℹ️ Informations Société")
    st.sidebar.markdown("**SOCIETE SENEGALAISE DE REASSURANCES S.A.**")
    st.sidebar.markdown("Capital: 10 000 000 000 FCFA")
    st.sidebar.markdown("Période d'analyse: 2018-2022")
    st.sidebar.markdown("NINEA: 003-6998 003")
    
    st.sidebar.title("🔧 Paramètres d'Analyse")
    analysis_type = st.sidebar.selectbox(
        "Type d'analyse",
        ["Vue d'Ensemble", "Analyse Comparative", "Rentabilité", "Solvabilité", 
         "Performance Technique", "Investissements", "Conformité Normes"]
    )
    
    # Menu principal
    if analysis_type == "Vue d'Ensemble":
        show_overview(analyzer)
    elif analysis_type == "Analyse Comparative":
        show_comparative_analysis(analyzer)
    elif analysis_type == "Rentabilité":
        show_profitability_analysis(analyzer)
    elif analysis_type == "Solvabilité":
        show_solvency_analysis(analyzer)
    elif analysis_type == "Performance Technique":
        show_technical_performance(analyzer)
    elif analysis_type == "Investissements":
        show_investment_analysis(analyzer)
    elif analysis_type == "Conformité Normes":
        show_compliance_analysis(analyzer)

def show_overview(analyzer):
    """Vue d'ensemble des performances 2018-2022"""
    
    st.markdown('<div class="section-header">📈 Évolution des Performances Clés 2018-2022</div>', unsafe_allow_html=True)
    
    # KPIs principaux sur 5 ans
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        resultat_2022 = analyzer.income_statements[2022]['resultat_net']
        resultat_2018 = analyzer.income_statements[2018]['resultat_net']
        evolution = ((resultat_2022 - resultat_2018) / resultat_2018) * 100
        
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.metric(
            label="Résultat Net 2022",
            value=format_currency(resultat_2022),
            delta=f"{evolution:.1f}% vs 2018"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        primes_2022 = analyzer.income_statements[2022]['primes_vie'] + analyzer.income_statements[2022]['primes_iard']
        primes_2018 = analyzer.income_statements[2018]['primes_vie'] + analyzer.income_statements[2018]['primes_iard']
        evolution = ((primes_2022 - primes_2018) / primes_2018) * 100
        
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.metric(
            label="Primes Totales 2022",
            value=format_currency(primes_2022),
            delta=f"{evolution:.1f}% vs 2018"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.metric(
            label="ROE 2022",
            value=f"{analyzer.ratios[2022]['roe']:.1f}%",
            delta=f"{(analyzer.ratios[2022]['roe'] - analyzer.ratios[2018]['roe']):.1f}% vs 2018"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.metric(
            label="Ratio de Solvabilité 2022",
            value=f"{analyzer.ratios[2022]['solvabilite']:.1f}%",
            delta=f"{(analyzer.ratios[2022]['solvabilite'] - analyzer.ratios[2018]['solvabilite']):.1f}% vs 2018"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Graphiques d'évolution
    col1, col2 = st.columns(2)
    
    with col1:
        # Évolution du résultat net
        years = analyzer.years
        resultats_nets = [analyzer.income_statements[y]['resultat_net'] for y in years]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=years, 
            y=resultats_nets,
            mode='lines+markers',
            name='Résultat Net',
            line=dict(color='blue', width=3)
        ))
        
        fig.update_layout(
            title='Évolution du Résultat Net (FCFA)',
            xaxis_title='Année',
            yaxis_title='Résultat Net (FCFA)',
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Évolution des primes par branche
        primes_vie = [analyzer.income_statements[y]['primes_vie'] for y in years]
        primes_iard = [analyzer.income_statements[y]['primes_iard'] for y in years]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Primes Vie', x=years, y=primes_vie, marker_color='lightblue'))
        fig.add_trace(go.Bar(name='Primes IARD', x=years, y=primes_iard, marker_color='blue'))
        
        fig.update_layout(
            title='Évolution des Primes par Branche (FCFA)',
            xaxis_title='Année',
            yaxis_title='Montant des Primes (FCFA)',
            barmode='stack'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Tableau récapitulatif des ratios clés
    st.markdown('<div class="section-header">📊 Tableau des Ratios Clés 2018-2022</div>', unsafe_allow_html=True)
    
    ratios_data = []
    for year in analyzer.years:
        ratios_data.append({
            'Année': year,
            'ROE (%)': f"{analyzer.ratios[year]['roe']:.1f}",
            'ROA (%)': f"{analyzer.ratios[year]['roa']:.1f}",
            'Marge Nette (%)': f"{analyzer.ratios[year]['marge_nette']:.1f}",
            'Solvabilité (%)': f"{analyzer.ratios[year]['solvabilite']:.1f}",
            'Combined Ratio Vie (%)': f"{analyzer.ratios[year]['combined_ratio_vie']:.1f}",
            'Combined Ratio IARD (%)': f"{analyzer.ratios[year]['combined_ratio_iard']:.1f}",
            'Rendement Placements (%)': f"{analyzer.ratios[year]['rendement_placements']:.1f}"
        })
    
    df_ratios = pd.DataFrame(ratios_data)
    st.dataframe(df_ratios, use_container_width=True)

def show_comparative_analysis(analyzer):
    """Analyse comparative détaillée"""
    
    st.markdown('<div class="section-header">📈 Analyse Comparative 2018-2022</div>', unsafe_allow_html=True)
    
    # Sélection des indicateurs à comparer
    indicateurs = st.multiselect(
        "Sélectionnez les indicateurs à comparer:",
        ['Résultat Net', 'Primes Totales', 'ROE', 'ROA', 'Marge Nette', 'Ratio de Solvabilité'],
        default=['Résultat Net', 'ROE', 'Primes Totales']
    )
    
    # Préparation des données
    years = analyzer.years
    data = {'Année': years}
    
    for indicateur in indicateurs:
        if indicateur == 'Résultat Net':
            data[indicateur] = [analyzer.income_statements[y]['resultat_net'] for y in years]
        elif indicateur == 'Primes Totales':
            data[indicateur] = [analyzer.income_statements[y]['primes_vie'] + analyzer.income_statements[y]['primes_iard'] for y in years]
        elif indicateur == 'ROE':
            data[indicateur] = [analyzer.ratios[y]['roe'] for y in years]
        elif indicateur == 'ROA':
            data[indicateur] = [analyzer.ratios[y]['roa'] for y in years]
        elif indicateur == 'Marge Nette':
            data[indicateur] = [analyzer.ratios[y]['marge_nette'] for y in years]
        elif indicateur == 'Ratio de Solvabilité':
            data[indicateur] = [analyzer.ratios[y]['solvabilite'] for y in years]
    
    df_comparatif = pd.DataFrame(data)
    
    # Graphique comparatif
    fig = go.Figure()
    
    for indicateur in indicateurs:
        if indicateur in ['ROE', 'ROA', 'Marge Nette', 'Ratio de Solvabilité']:
            # Ces indicateurs sont en pourcentage
            fig.add_trace(go.Scatter(
                x=years, y=df_comparatif[indicateur],
                mode='lines+markers',
                name=indicateur
            ))
        else:
            # Ces indicateurs sont en valeur absolue
            fig.add_trace(go.Bar(
                x=years, y=df_comparatif[indicateur],
                name=indicateur
            ))
    
    fig.update_layout(
        title='Comparaison des Indicateurs Clés 2018-2022',
        xaxis_title='Année',
        yaxis_title='Valeur',
        barmode='group' if 'Résultat Net' in indicateurs or 'Primes Totales' in indicateurs else 'stack'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tableau de croissance annuelle
    st.markdown('<div class="section-header">📈 Taux de Croissance Annuel</div>', unsafe_allow_html=True)
    
    croissance_data = []
    for i in range(1, len(analyzer.years)):
        year = analyzer.years[i]
        prev_year = analyzer.years[i-1]
        
        # Calcul des taux de croissance
        croissance_resultat = ((analyzer.income_statements[year]['resultat_net'] - analyzer.income_statements[prev_year]['resultat_net']) / 
                             analyzer.income_statements[prev_year]['resultat_net']) * 100
        
        primes_year = analyzer.income_statements[year]['primes_vie'] + analyzer.income_statements[year]['primes_iard']
        primes_prev = analyzer.income_statements[prev_year]['primes_vie'] + analyzer.income_statements[prev_year]['primes_iard']
        croissance_primes = ((primes_year - primes_prev) / primes_prev) * 100
        
        croissance_data.append({
            'Période': f"{prev_year}-{year}",
            'Croissance Résultat Net (%)': f"{croissance_resultat:.1f}",
            'Croissance Primes Totales (%)': f"{croissance_primes:.1f}",
            'Évolution ROE (points)': f"{analyzer.ratios[year]['roe'] - analyzer.ratios[prev_year]['roe']:.1f}",
            'Évolution Solvabilité (points)': f"{analyzer.ratios[year]['solvabilite'] - analyzer.ratios[prev_year]['solvabilite']:.1f}"
        })
    
    df_croissance = pd.DataFrame(croissance_data)
    st.dataframe(df_croissance, use_container_width=True)

def show_profitability_analysis(analyzer):
    """Analyse de rentabilité détaillée"""
    st.markdown('<div class="section-header">💰 Analyse de Rentabilité 2018-2022</div>', unsafe_allow_html=True)
    
    # Graphique d'évolution de la rentabilité
    years = analyzer.years
    roe_values = [analyzer.ratios[y]['roe'] for y in years]
    roa_values = [analyzer.ratios[y]['roa'] for y in years]
    marge_nette_values = [analyzer.ratios[y]['marge_nette'] for y in years]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=roe_values, mode='lines+markers', name='ROE', line=dict(color='blue', width=3)))
    fig.add_trace(go.Scatter(x=years, y=roa_values, mode='lines+markers', name='ROA', line=dict(color='green', width=3)))
    fig.add_trace(go.Scatter(x=years, y=marge_nette_values, mode='lines+markers', name='Marge Nette', line=dict(color='red', width=3)))
    
    fig.update_layout(
        title='Évolution des Ratios de Rentabilité 2018-2022',
        xaxis_title='Année',
        yaxis_title='Pourcentage (%)',
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Analyse de la composition du résultat
    st.markdown('<div class="section-header">🧮 Composition du Résultat Net</div>', unsafe_allow_html=True)
    
    composition_data = []
    for year in analyzer.years:
        is_stmt = analyzer.income_statements[year]
        total_produits = is_stmt['primes_vie'] + is_stmt['primes_iard'] + is_stmt['revenus_placements'] + is_stmt['produits_exceptionnels']
        
        composition_data.append({
            'Année': year,
            'Résultat Net': format_currency(is_stmt['resultat_net']),
            'Marge Technique': f"{(is_stmt['primes_vie'] + is_stmt['primes_iard'] - is_stmt['sinistres_vie'] - is_stmt['sinistres_iard'] - is_stmt['commissions_vie'] - is_stmt['commissions_iard']) / 1000000:.1f} M",
            'Revenus Placements': format_currency(is_stmt['revenus_placements']),
            'Produits Exceptionnels': format_currency(is_stmt['produits_exceptionnels']),
            'Part Technique dans Résultat': f"{((is_stmt['primes_vie'] + is_stmt['primes_iard'] - is_stmt['sinistres_vie'] - is_stmt['sinistres_iard'] - is_stmt['commissions_vie'] - is_stmt['commissions_iard']) / is_stmt['resultat_net'] * 100) if is_stmt['resultat_net'] != 0 else 0:.1f}%"
        })
    
    df_composition = pd.DataFrame(composition_data)
    st.dataframe(df_composition, use_container_width=True)

def show_solvency_analysis(analyzer):
    """Analyse de solvabilité selon les normes internationales"""
    st.markdown('<div class="section-header">🛡️ Analyse de Solvabilité - Normes Internationales</div>', unsafe_allow_html=True)
    
    # Ratios de solvabilité selon Solvabilité II et normes CIMA
    solvency_data = []
    
    for year in analyzer.years:
        ratios = analyzer.ratios[year]
        bs = analyzer.balance_sheets[year]
        
        # Calcul des ratios selon les normes
        ratio_solvabilite_ii = ratios['solvabilite']  # Base simplifiée
        current_ratio = ratios['current_ratio']
        ratio_couverture = ratios['couverture_provisions']
        
        # Évaluation de la conformité
        conforme_solvabilite = ratio_solvabilite_ii >= 15  # Norme sectorielle
        conforme_liquidite = current_ratio >= 1.5
        conforme_couverture = ratio_couverture >= 100
        
        solvency_data.append({
            'Année': year,
            'Ratio Solvabilité II (%)': f"{ratio_solvabilite_ii:.1f}",
            'Current Ratio': f"{current_ratio:.2f}",
            'Couverture Provisions (%)': f"{ratio_couverture:.1f}",
            'Conformité Solvabilité': '✅ Conforme' if conforme_solvabilite else '❌ Non conforme',
            'Conformité Liquidité': '✅ Conforme' if conforme_liquidite else '❌ Non conforme',
            'Conformité Couverture': '✅ Conforme' if conforme_couverture else '❌ Non conforme'
        })
    
    df_solvency = pd.DataFrame(solvency_data)
    st.dataframe(df_solvency, use_container_width=True)
    
    # Graphique radar pour la solvabilité 2022
    st.markdown('<div class="section-header">🎯 Radar de Solvabilité 2022 vs Normes</div>', unsafe_allow_html=True)
    
    categories = ['Solvabilité Base Solo', 'Liquidité', 'Couv. Provisions', 'Rendement', 'Croissance']
    
    fig = go.Figure()
    
    # Valeurs 2022 normalisées
    valeurs_2022 = [
        min(analyzer.ratios[2022]['solvabilite']/20, 1),  # Objectif > 15%
        min(analyzer.ratios[2022]['current_ratio']/2, 1),  # Objectif > 1.5
        min(analyzer.ratios[2022]['couverture_provisions']/150, 1),  # Objectif > 100%
        min(analyzer.ratios[2022]['rendement_placements']/15, 1),  # Objectif > 10%
        min((analyzer.ratios[2022]['croissance_primes'] + 20)/40, 1)  # Objectif croissance positive
    ]
    
    fig.add_trace(go.Scatterpolar(
        r=valeurs_2022,
        theta=categories,
        fill='toself',
        name='SEN-RE 2022'
    ))
    
    # Ligne de référence pour les normes
    normes = [0.75, 0.75, 0.67, 0.67, 0.5]  # Seuils minimum normalisés
    
    fig.add_trace(go.Scatterpolar(
        r=normes,
        theta=categories,
        fill='toself',
        name='Normes Sectorielles',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title='Radar de Solvabilité - Comparaison aux Normes Internationales'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_technical_performance(analyzer):
    """Analyse de la performance technique par branche"""
    st.markdown('<div class="section-header">🎯 Performance Technique par Branche 2018-2022</div>', unsafe_allow_html=True)
    
    # Combined Ratios par année
    years = analyzer.years
    combined_vie = [analyzer.ratios[y]['combined_ratio_vie'] for y in years]
    combined_iard = [analyzer.ratios[y]['combined_ratio_iard'] for y in years]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=combined_vie, mode='lines+markers', name='Combined Ratio Vie', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=years, y=combined_iard, mode='lines+markers', name='Combined Ratio IARD', line=dict(color='red')))
    fig.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="Seuil de Rentabilité (100%)")
    
    fig.update_layout(
        title='Évolution des Combined Ratios par Branche',
        xaxis_title='Année',
        yaxis_title='Combined Ratio (%)',
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Analyse détaillée par branche
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Branche Vie - Ratios Techniques")
        vie_data = []
        for year in analyzer.years:
            vie_data.append({
                'Année': year,
                'Ratio Sinistres (%)': f"{analyzer.ratios[year]['ratio_sinistres_vie']:.1f}",
                'Ratio Commissions (%)': f"{analyzer.ratios[year]['ratio_commissions_vie']:.1f}",
                'Combined Ratio (%)': f"{analyzer.ratios[year]['combined_ratio_vie']:.1f}",
                'Rentabilité': '✅ Rentable' if analyzer.ratios[year]['combined_ratio_vie'] < 100 else '❌ Déficitaire'
            })
        
        df_vie = pd.DataFrame(vie_data)
        st.dataframe(df_vie, use_container_width=True)
    
    with col2:
        st.subheader("🏠 Branche IARD - Ratios Techniques")
        iard_data = []
        for year in analyzer.years:
            iard_data.append({
                'Année': year,
                'Ratio Sinistres (%)': f"{analyzer.ratios[year]['ratio_sinistres_iard']:.1f}",
                'Ratio Commissions (%)': f"{analyzer.ratios[year]['ratio_commissions_iard']:.1f}",
                'Combined Ratio (%)': f"{analyzer.ratios[year]['combined_ratio_iard']:.1f}",
                'Rentabilité': '✅ Rentable' if analyzer.ratios[year]['combined_ratio_iard'] < 100 else '❌ Déficitaire'
            })
        
        df_iard = pd.DataFrame(iard_data)
        st.dataframe(df_iard, use_container_width=True)

def show_investment_analysis(analyzer):
    """Analyse de la performance des investissements"""
    st.markdown('<div class="section-header">📈 Analyse des Investissements 2018-2022</div>', unsafe_allow_html=True)
    
    # Évolution du rendement des placements
    years = analyzer.years
    rendements = [analyzer.ratios[y]['rendement_placements'] for y in years]
    revenus_placements = [analyzer.income_statements[y]['revenus_placements'] for y in years]
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=years, y=rendements, mode='lines+markers', name='Rendement', line=dict(color='green', width=3)))
        fig.add_hline(y=5, line_dash="dash", line_color="red", annotation_text="Seuil Minimum (5%)")
        
        fig.update_layout(
            title='Rendement du Portefeuille d\'Investissements',
            xaxis_title='Année',
            yaxis_title='Rendement (%)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=years, y=revenus_placements, name='Revenus Placements', marker_color='orange'))
        
        fig.update_layout(
            title='Revenus des Placements (FCFA)',
            xaxis_title='Année',
            yaxis_title='Revenus (FCFA)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Composition du portefeuille
    st.markdown('<div class="section-header">🏦 Composition du Portefeuille d\'Investissement</div>', unsafe_allow_html=True)
    
    portefeuille_data = []
    for year in analyzer.years:
        bs = analyzer.balance_sheets[year]
        total_investissements = bs['titres_participations'] + bs['disponibilites'] + bs['creances_reassurance']
        
        if total_investissements > 0:
            portefeuille_data.append({
                'Année': year,
                'Titres & Participations (%)': f"{(bs['titres_participations'] / total_investissements * 100):.1f}",
                'Disponibilités (%)': f"{(bs['disponibilites'] / total_investissements * 100):.1f}",
                'Créances Réassurance (%)': f"{(bs['creances_reassurance'] / total_investissements * 100):.1f}",
                'Rendement (%)': f"{analyzer.ratios[year]['rendement_placements']:.1f}"
            })
    
    df_portefeuille = pd.DataFrame(portefeuille_data)
    st.dataframe(df_portefeuille, use_container_width=True)

def show_compliance_analysis(analyzer):
    """Analyse de conformité aux normes internationales"""
    st.markdown('<div class="section-header">🏛️ Conformité aux Normes Internationales</div>', unsafe_allow_html=True)
    
    # Normes de référence
    normes_internationales = {
        'Solvabilité II (Europe)': {
            'Ratio de Solvabilité': '> 100%',
            'Capital Minimum': '≥ SCR',
            'Pillar 1 - Exigences Quantitatives': 'Conforme',
            'Pillar 2 - Gouv. & Surveillance': 'Partiellement Conforme',
            'Pillar 3 - Transparence': 'Conforme'
        },
        'Normes CIMA (Afrique)': {
            'Ratio de Marge de Solvabilité': '> 15%',
            'Couverture Provisions Techniques': '> 100%',
            'Diversification Actifs': 'Conforme',
            'Reconnaissance Réassurance': 'Conforme',
            'Contrôle Technique': 'Conforme'
        },
        'IFRS 17 (Comptabilité)': {
            'Mesure des Contrats': 'En cours',
            'Reconnaissance des Produits': 'Partiellement Conforme',
            'Informations à Fournir': 'Conforme',
            'Date Application': '2023'
        }
    }
    
    # Évaluation de la conformité SEN-RE
    evaluation_conformite = {
        'Domaine': [
            'Solvabilité - Ratio de Base',
            'Liquidité - Current Ratio',
            'Couverture Provisions Techniques',
            'Diversification du Portefeuille',
            'Transparence Informationnelle',
            'Gouvernance des Risques',
            'Adéquation des Capitaux',
            'Performance Technique'
        ],
        'Norme Référence': [
            'Solvabilité II / CIMA',
            'Normes Sectorielles',
            'Principes Actuariels',
            'Diversification Actifs',
            'IFRS/Transparence',
            'Pillar 2 Solvabilité II',
            'Exigences Réglementaires',
            'Best Practices'
        ],
        'Statut SEN-RE 2022': [
            '✅ Conforme',
            '✅ Conforme',
            '✅ Conforme',
            '⚠️ À Améliorer',
            '✅ Conforme',
            '✅ Conforme',
            '✅ Conforme',
            '✅ Conforme'
        ],
        'Score': [9, 8, 9, 6, 8, 8, 9, 8]
    }
    
    df_conformite = pd.DataFrame(evaluation_conformite)
    st.dataframe(df_conformite, use_container_width=True)
    
    # Score global de conformité
    score_global = sum(evaluation_conformite['Score']) / len(evaluation_conformite['Score'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = score_global,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Score Global de Conformité"},
            gauge = {
                'axis': {'range': [None, 10]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 6], 'color': "red"},
                    {'range': [6, 8], 'color': "yellow"},
                    {'range': [8, 10], 'color': "green"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 8}}
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📋 Recommandations de Conformité")
        recommandations = [
            "**✅ Points Forts:** Excellente solvabilité et couverture des provisions",
            "**✅ Points Forts:** Bonne transparence et gouvernance des risques",
            "**⚠️ À Améliorer:** Diversification du portefeuille d'investissement",
            "**🎯 Priorité:** Préparation à l'IFRS 17 - Mesure des contrats",
            "**🔍 Surveillance:** Maintenir les ratios techniques sous 100%",
            "**📊 Reporting:** Renforcer les indicateurs ESG (Environnementaux, Sociaux, Gouvernance)"
        ]
        
        for rec in recommandations:
            st.markdown(f"- {rec}")

if __name__ == "__main__":
    main()