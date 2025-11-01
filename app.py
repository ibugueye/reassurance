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
    page_title="Analyse Réassurance - SEN-RE",
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
</style>
""", unsafe_allow_html=True)

class ReinsuranceAnalyzer:
    def __init__(self):
        self.load_data()
        self.calculate_ratios()
    
    def load_data(self):
        """Chargement des données financières"""
        # Données du bilan
        self.balance_sheet_2022 = {
            'actif_total': 67736411528,
            'capitaux_propres': 12738350066,
            'provisions_techniques': 35157205175,
            'actif_circulant': 40497600164,
            'disponibilites': 4193211013,
            'creances_reassurance': 18741397700,
            'titres_participations': 5451434971
        }
        
        self.balance_sheet_2021 = {
            'actif_total': 63847390546,
            'capitaux_propres': 11805273655,
            'provisions_techniques': 34158229153,
            'actif_circulant': 39212619761,
            'disponibilites': 8612944474,
            'creances_reassurance': 17378888189,
            'titres_participations': 4084819617
        }
        
        # Données du compte de résultat
        self.income_statement_2022 = {
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
        }
        
        self.income_statement_2021 = {
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
        }
    
    def calculate_ratios(self):
        """Calcul des ratios techniques et financiers"""
        # Ratios de rentabilité
        self.roe_2022 = (self.income_statement_2022['resultat_net'] / self.balance_sheet_2022['capitaux_propres']) * 100
        self.roe_2021 = (self.income_statement_2021['resultat_net'] / self.balance_sheet_2021['capitaux_propres']) * 100
        
        self.roa_2022 = (self.income_statement_2022['resultat_net'] / self.balance_sheet_2022['actif_total']) * 100
        self.roa_2021 = (self.income_statement_2021['resultat_net'] / self.balance_sheet_2021['actif_total']) * 100
        
        # Ratios de solvabilité
        self.solvabilite_2022 = (self.balance_sheet_2022['capitaux_propres'] / self.balance_sheet_2022['actif_total']) * 100
        self.solvabilite_2021 = (self.balance_sheet_2021['capitaux_propres'] / self.balance_sheet_2021['actif_total']) * 100
        
        # Ratios de liquidité
        self.current_ratio_2022 = self.balance_sheet_2022['actif_circulant'] / 16034315935  # Dettes court terme
        self.current_ratio_2021 = self.balance_sheet_2021['actif_circulant'] / 14470724315
        
        # Ratios techniques Vie
        primes_vie_2022 = self.income_statement_2022['primes_vie']
        self.ratio_sinistres_vie_2022 = (self.income_statement_2022['sinistres_vie'] / primes_vie_2022) * 100
        self.ratio_commissions_vie_2022 = (self.income_statement_2022['commissions_vie'] / primes_vie_2022) * 100
        self.combined_ratio_vie_2022 = self.ratio_sinistres_vie_2022 + self.ratio_commissions_vie_2022
        
        primes_vie_2021 = self.income_statement_2021['primes_vie']
        self.ratio_sinistres_vie_2021 = (self.income_statement_2021['sinistres_vie'] / primes_vie_2021) * 100
        self.ratio_commissions_vie_2021 = (self.income_statement_2021['commissions_vie'] / primes_vie_2021) * 100
        self.combined_ratio_vie_2021 = self.ratio_sinistres_vie_2021 + self.ratio_commissions_vie_2021
        
        # Ratios techniques IARD
        primes_iard_2022 = self.income_statement_2022['primes_iard']
        self.ratio_sinistres_iard_2022 = (self.income_statement_2022['sinistres_iard'] / primes_iard_2022) * 100
        self.ratio_commissions_iard_2022 = (self.income_statement_2022['commissions_iard'] / primes_iard_2022) * 100
        self.combined_ratio_iard_2022 = self.ratio_sinistres_iard_2022 + self.ratio_commissions_iard_2022
        
        primes_iard_2021 = self.income_statement_2021['primes_iard']
        self.ratio_sinistres_iard_2021 = (self.income_statement_2021['sinistres_iard'] / primes_iard_2021) * 100
        self.ratio_commissions_iard_2021 = (self.income_statement_2021['commissions_iard'] / primes_iard_2021) * 100
        self.combined_ratio_iard_2021 = self.ratio_sinistres_iard_2021 + self.ratio_commissions_iard_2021
        
        # Ratio de rendement des placements
        self.rendement_placements_2022 = (self.income_statement_2022['revenus_placements'] / 
                                         (self.balance_sheet_2022['titres_participations'] + self.balance_sheet_2022['disponibilites'])) * 100
        self.rendement_placements_2021 = (self.income_statement_2021['revenus_placements'] / 
                                         (self.balance_sheet_2021['titres_participations'] + self.balance_sheet_2021['disponibilites'])) * 100

def format_currency(amount):
    """Formate les montants en millions ou milliards"""
    if abs(amount) >= 1000000000:
        return f"{amount/1000000000:,.2f} Md FCFA"
    elif abs(amount) >= 1000000:
        return f"{amount/1000000:,.2f} M FCFA"
    else:
        return f"{amount:,.0f} FCFA"

def main():
    st.markdown('<h1 class="main-header">📊 Tableau de Bord Analytique - Réassurance SEN-RE</h1>', unsafe_allow_html=True)
    
    # Initialisation de l'analyseur
    analyzer = ReinsuranceAnalyzer()
    
    # Sidebar avec informations générales
    st.sidebar.title("ℹ️ Informations Société")
    st.sidebar.markdown("**SOCIETE SENEGALAISE DE REASSURANCES S.A.**")
    st.sidebar.markdown("Capital: 10 000 000 000 FCFA")
    st.sidebar.markdown("Période d'analyse: 31/12/2022")
    st.sidebar.markdown("NINEA: 003-6998 003")
    
    st.sidebar.title("🔧 Paramètres d'Analyse")
    analysis_type = st.sidebar.selectbox(
        "Type d'analyse",
        ["Analyse Complète", "Rentabilité", "Solvabilité", "Performance Technique", "Investissements"]
    )
    
    # Menu principal
    if analysis_type == "Analyse Complète":
        show_comprehensive_analysis(analyzer)
    elif analysis_type == "Rentabilité":
        show_profitability_analysis(analyzer)
    elif analysis_type == "Solvabilité":
        show_solvency_analysis(analyzer)
    elif analysis_type == "Performance Technique":
        show_technical_performance(analyzer)
    elif analysis_type == "Investissements":
        show_investment_analysis(analyzer)

def show_comprehensive_analysis(analyzer):
    """Affichage de l'analyse complète"""
    
    st.markdown('<div class="section-header">📈 Tableau de Bord Exécutif</div>', unsafe_allow_html=True)
    
    # KPIs principaux
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.metric(
            label="Résultat Net",
            value=format_currency(analyzer.income_statement_2022['resultat_net']),
            delta=f"{((analyzer.income_statement_2022['resultat_net'] - analyzer.income_statement_2021['resultat_net']) / analyzer.income_statement_2021['resultat_net'] * 100):.1f}%"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.metric(
            label="ROE",
            value=f"{analyzer.roe_2022:.1f}%",
            delta=f"{(analyzer.roe_2022 - analyzer.roe_2021):.1f}%"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.metric(
            label="Ratio de Solvabilité",
            value=f"{analyzer.solvabilite_2022:.1f}%",
            delta=f"{(analyzer.solvabilite_2022 - analyzer.solvabilite_2021):.1f}%"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.metric(
            label="Combined Ratio Total",
            value=f"{(analyzer.combined_ratio_vie_2022 + analyzer.combined_ratio_iard_2022)/2:.1f}%",
            delta=f"{((analyzer.combined_ratio_vie_2022 + analyzer.combined_ratio_iard_2022) - (analyzer.combined_ratio_vie_2021 + analyzer.combined_ratio_iard_2021))/2:.1f}%"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Graphiques principaux
    col1, col2 = st.columns(2)
    
    with col1:
        # Évolution des primes par branche
        fig = go.Figure()
        fig.add_trace(go.Bar(name='2021 - Vie', x=['Vie'], y=[analyzer.income_statement_2021['primes_vie']/1000000], marker_color='lightblue'))
        fig.add_trace(go.Bar(name='2022 - Vie', x=['Vie'], y=[analyzer.income_statement_2022['primes_vie']/1000000], marker_color='blue'))
        fig.add_trace(go.Bar(name='2021 - IARD', x=['IARD'], y=[analyzer.income_statement_2021['primes_iard']/1000000], marker_color='lightcoral'))
        fig.add_trace(go.Bar(name='2022 - IARD', x=['IARD'], y=[analyzer.income_statement_2022['primes_iard']/1000000], marker_color='red'))
        
        fig.update_layout(
            title='Évolution des Primes par Branche (en millions FCFA)',
            xaxis_title='Branche',
            yaxis_title='Montant (M FCFA)',
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Structure du bilan
        labels = ['Capitaux Propres', 'Provisions Techniques', 'Dettes Court Terme', 'Autres Passifs']
        values_2022 = [
            analyzer.balance_sheet_2022['capitaux_propres'],
            analyzer.balance_sheet_2022['provisions_techniques'],
            16034315935,  # Dettes court terme
            analyzer.balance_sheet_2022['actif_total'] - analyzer.balance_sheet_2022['capitaux_propres'] - analyzer.balance_sheet_2022['provisions_techniques'] - 16034315935
        ]
        
        fig = go.Figure(data=[go.Pie(labels=labels, values=values_2022, hole=.3)])
        fig.update_layout(title='Structure du Passif 2022')
        st.plotly_chart(fig, use_container_width=True)
    
    # Ratios techniques détaillés
    st.markdown('<div class="section-header">🔧 Ratios Techniques par Branche</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Ratios Vie
        st.subheader("📋 Branche Vie")
        ratios_vie_data = {
            'Ratio': ['Sinistres', 'Commissions', 'Combined Ratio'],
            '2021': [analyzer.ratio_sinistres_vie_2021, analyzer.ratio_commissions_vie_2021, analyzer.combined_ratio_vie_2021],
            '2022': [analyzer.ratio_sinistres_vie_2022, analyzer.ratio_commissions_vie_2022, analyzer.combined_ratio_vie_2022],
            'Évolution': [
                analyzer.ratio_sinistres_vie_2022 - analyzer.ratio_sinistres_vie_2021,
                analyzer.ratio_commissions_vie_2022 - analyzer.ratio_commissions_vie_2021,
                analyzer.combined_ratio_vie_2022 - analyzer.combined_ratio_vie_2021
            ]
        }
        df_vie = pd.DataFrame(ratios_vie_data)
        st.dataframe(df_vie.style.format({
            '2021': '{:.1f}%',
            '2022': '{:.1f}%',
            'Évolution': '{:+.1f}%'
        }), use_container_width=True)
    
    with col2:
        # Ratios IARD
        st.subheader("🏠 Branche IARD")
        ratios_iard_data = {
            'Ratio': ['Sinistres', 'Commissions', 'Combined Ratio'],
            '2021': [analyzer.ratio_sinistres_iard_2021, analyzer.ratio_commissions_iard_2021, analyzer.combined_ratio_iard_2021],
            '2022': [analyzer.ratio_sinistres_iard_2022, analyzer.ratio_commissions_iard_2022, analyzer.combined_ratio_iard_2022],
            'Évolution': [
                analyzer.ratio_sinistres_iard_2022 - analyzer.ratio_sinistres_iard_2021,
                analyzer.ratio_commissions_iard_2022 - analyzer.ratio_commissions_iard_2021,
                analyzer.combined_ratio_iard_2022 - analyzer.combined_ratio_iard_2021
            ]
        }
        df_iard = pd.DataFrame(ratios_iard_data)
        st.dataframe(df_iard.style.format({
            '2021': '{:.1f}%',
            '2022': '{:.1f}%',
            'Évolution': '{:+.1f}%'
        }), use_container_width=True)

def show_profitability_analysis(analyzer):
    """Analyse de rentabilité détaillée"""
    st.markdown('<div class="section-header">💰 Analyse de Rentabilité</div>', unsafe_allow_html=True)
    
    # Graphique d'évolution de la rentabilité
    years = ['2021', '2022']
    roe_values = [analyzer.roe_2021, analyzer.roe_2022]
    roa_values = [analyzer.roa_2021, analyzer.roa_2022]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=roe_values, mode='lines+markers', name='ROE', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=years, y=roa_values, mode='lines+markers', name='ROA', line=dict(color='green')))
    
    fig.update_layout(
        title='Évolution des Ratios de Rentabilité',
        xaxis_title='Année',
        yaxis_title='Pourcentage (%)',
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Tableau détaillé de rentabilité
    profitability_data = {
        'Indicateur': ['Résultat Net', 'ROE', 'ROA', 'Marge Nette'],
        '2021': [
            format_currency(analyzer.income_statement_2021['resultat_net']),
            f"{analyzer.roe_2021:.1f}%",
            f"{analyzer.roa_2021:.1f}%",
            f"{(analyzer.income_statement_2021['resultat_net'] / (analyzer.income_statement_2021['primes_vie'] + analyzer.income_statement_2021['primes_iard']) * 100):.1f}%"
        ],
        '2022': [
            format_currency(analyzer.income_statement_2022['resultat_net']),
            f"{analyzer.roe_2022:.1f}%",
            f"{analyzer.roa_2022:.1f}%",
            f"{(analyzer.income_statement_2022['resultat_net'] / (analyzer.income_statement_2022['primes_vie'] + analyzer.income_statement_2022['primes_iard']) * 100):.1f}%"
        ],
        'Évolution': [
            f"+{((analyzer.income_statement_2022['resultat_net'] - analyzer.income_statement_2021['resultat_net']) / analyzer.income_statement_2021['resultat_net'] * 100):.1f}%",
            f"{analyzer.roe_2022 - analyzer.roe_2021:+.1f}%",
            f"{analyzer.roa_2022 - analyzer.roa_2021:+.1f}%",
            f"+{((analyzer.income_statement_2022['resultat_net'] / (analyzer.income_statement_2022['primes_vie'] + analyzer.income_statement_2022['primes_iard']) * 100) - (analyzer.income_statement_2021['resultat_net'] / (analyzer.income_statement_2021['primes_vie'] + analyzer.income_statement_2021['primes_iard']) * 100)):.1f}%"
        ]
    }
    
    df_profitability = pd.DataFrame(profitability_data)
    st.dataframe(df_profitability, use_container_width=True)

def show_solvency_analysis(analyzer):
    """Analyse de solvabilité détaillée"""
    st.markdown('<div class="section-header">🛡️ Analyse de Solvabilité</div>', unsafe_allow_html=True)
    
    # Ratios de solvabilité selon les normes internationales
    solvency_ratios = {
        'Ratio': [
            'Ratio de Solvabilité (Base Solo)',
            'Current Ratio',
            'Ratio Capitaux Propres/Actif',
            'Couverture Provisions Techniques'
        ],
        'Valeur 2022': [
            f"{analyzer.solvabilite_2022:.1f}%",
            f"{analyzer.current_ratio_2022:.2f}",
            f"{(analyzer.balance_sheet_2022['capitaux_propres'] / analyzer.balance_sheet_2022['actif_total'] * 100):.1f}%",
            f"{(analyzer.balance_sheet_2022['actif_circulant'] / 35157205175 * 100):.1f}%"
        ],
        'Norme Sectorielle': [
            '> 15%',
            '> 1.5',
            '> 15%',
            '> 100%'
        ],
        'Conformité': [
            '✅ Conforme' if analyzer.solvabilite_2022 > 15 else '❌ Non conforme',
            '✅ Conforme' if analyzer.current_ratio_2022 > 1.5 else '❌ Non conforme',
            '✅ Conforme' if (analyzer.balance_sheet_2022['capitaux_propres'] / analyzer.balance_sheet_2022['actif_total'] * 100) > 15 else '❌ Non conforme',
            '✅ Conforme' if (analyzer.balance_sheet_2022['actif_circulant'] / 35157205175 * 100) > 100 else '❌ Non conforme'
        ]
    }
    
    df_solvency = pd.DataFrame(solvency_ratios)
    st.dataframe(df_solvency, use_container_width=True)
    
    # Graphique radar pour la solvabilité
    categories = ['Solvabilité Base Solo', 'Liquidité', 'Couv. Provisions', 'Rendement']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=[analyzer.solvabilite_2022/20, min(analyzer.current_ratio_2022/2, 1), 
           min((analyzer.balance_sheet_2022['actif_circulant'] / 35157205175 * 100)/150, 1),
           min(analyzer.rendement_placements_2022/15, 1)],
        theta=categories,
        fill='toself',
        name='2022'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title='Radar de Solvabilité (Normes Internationales)'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_technical_performance(analyzer):
    """Analyse de la performance technique"""
    st.markdown('<div class="section-header">🎯 Performance Technique</div>', unsafe_allow_html=True)
    
    # Combined Ratio par branche
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Branche Vie', 'Branche IARD'), specs=[[{'type':'domain'}, {'type':'domain'}]])
    
    fig.add_trace(go.Indicator(
        mode = "gauge+number+delta",
        value = analyzer.combined_ratio_vie_2022,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Combined Ratio Vie"},
        delta = {'reference': analyzer.combined_ratio_vie_2021},
        gauge = {
            'axis': {'range': [None, 120]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 80], 'color': "lightgreen"},
                {'range': [80, 100], 'color': "yellow"},
                {'range': [100, 120], 'color': "red"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 100}}
    ), row=1, col=1)
    
    fig.add_trace(go.Indicator(
        mode = "gauge+number+delta",
        value = analyzer.combined_ratio_iard_2022,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Combined Ratio IARD"},
        delta = {'reference': analyzer.combined_ratio_iard_2021},
        gauge = {
            'axis': {'range': [None, 120]},
            'bar': {'color': "darkred"},
            'steps': [
                {'range': [0, 80], 'color': "lightgreen"},
                {'range': [80, 100], 'color': "yellow"},
                {'range': [100, 120], 'color': "red"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 100}}
    ), row=1, col=2)
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Analyse de la sinistralité
    st.subheader("📊 Analyse Détailée de la Sinistralité")
    
    sinistralite_data = {
        'Branche': ['Vie', 'IARD', 'Total'],
        'Sinistres 2021 (M FCFA)': [
            analyzer.income_statement_2021['sinistres_vie']/1000000,
            analyzer.income_statement_2021['sinistres_iard']/1000000,
            (analyzer.income_statement_2021['sinistres_vie'] + analyzer.income_statement_2021['sinistres_iard'])/1000000
        ],
        'Sinistres 2022 (M FCFA)': [
            analyzer.income_statement_2022['sinistres_vie']/1000000,
            analyzer.income_statement_2022['sinistres_iard']/1000000,
            (analyzer.income_statement_2022['sinistres_vie'] + analyzer.income_statement_2022['sinistres_iard'])/1000000
        ],
        'Évolution': [
            f"{((analyzer.income_statement_2022['sinistres_vie'] - analyzer.income_statement_2021['sinistres_vie'])/analyzer.income_statement_2021['sinistres_vie']*100):.1f}%",
            f"{((analyzer.income_statement_2022['sinistres_iard'] - analyzer.income_statement_2021['sinistres_iard'])/analyzer.income_statement_2021['sinistres_iard']*100):.1f}%",
            f"{((analyzer.income_statement_2022['sinistres_vie'] + analyzer.income_statement_2022['sinistres_iard'] - analyzer.income_statement_2021['sinistres_vie'] - analyzer.income_statement_2021['sinistres_iard'])/(analyzer.income_statement_2021['sinistres_vie'] + analyzer.income_statement_2021['sinistres_iard'])*100):.1f}%"
        ]
    }
    
    df_sinistralite = pd.DataFrame(sinistralite_data)
    st.dataframe(df_sinistralite, use_container_width=True)

def show_investment_analysis(analyzer):
    """Analyse des investissements"""
    st.markdown('<div class="section-header">📈 Analyse des Investissements</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Composition du portefeuille
        portefeuille_data = {
            'Type': ['Titres et Participations', 'Disponibilités', 'Créances Réassurance', 'Autres Actifs'],
            'Montant 2022 (Mds FCFA)': [
                analyzer.balance_sheet_2022['titres_participations']/1000000000,
                analyzer.balance_sheet_2022['disponibilites']/1000000000,
                analyzer.balance_sheet_2022['creances_reassurance']/1000000000,
                (analyzer.balance_sheet_2022['actif_total'] - analyzer.balance_sheet_2022['titres_participations'] - analyzer.balance_sheet_2022['disponibilites'] - analyzer.balance_sheet_2022['creances_reassurance'])/1000000000
            ]
        }
        
        df_portefeuille = pd.DataFrame(portefeuille_data)
        fig = px.pie(df_portefeuille, values='Montant 2022 (Mds FCFA)', names='Type', title='Composition du Portefeuille Investissements 2022')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Performance des placements
        performance_data = {
            'Indicateur': ['Rendement Portefeuille', 'Revenus Placements', 'Revenus Participations'],
            '2021': [
                f"{analyzer.rendement_placements_2021:.1f}%",
                format_currency(analyzer.income_statement_2021['revenus_placements']),
                format_currency(289946853)  # Revenus participations 2021
            ],
            '2022': [
                f"{analyzer.rendement_placements_2022:.1f}%",
                format_currency(analyzer.income_statement_2022['revenus_placements']),
                format_currency(289946853)  # Revenus participations 2022
            ],
            'Évolution': [
                f"{analyzer.rendement_placements_2022 - analyzer.rendement_placements_2021:+.1f}%",
                f"+{((analyzer.income_statement_2022['revenus_placements'] - analyzer.income_statement_2021['revenus_placements'])/analyzer.income_statement_2021['revenus_placements']*100):.1f}%",
                "0.0%"  # Stable
            ]
        }
        
        df_performance = pd.DataFrame(performance_data)
        st.dataframe(df_performance, use_container_width=True)
    
    # Recommandations stratégiques
    st.markdown('<div class="section-header">💡 Recommandations Stratégiques</div>', unsafe_allow_html=True)
    
    recommendations = [
        "✅ **Diversification du portefeuille**: Augmenter les investissements en actions à dividendes",
        "✅ **Optimisation de la liquidité**: Réduire les disponibilités excédentaires au profit d'investissements à moyen terme",
        "⚠️ **Gestion du risque de crédit**: Renforcer le suivi des créances de réassurance",
        "🎯 **Stratégie de participation**: Évaluer de nouvelles opportunités de participations stratégiques"
    ]
    
    for rec in recommendations:
        st.markdown(f"- {rec}")

if __name__ == "__main__":
    main()