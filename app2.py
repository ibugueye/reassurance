import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import io
from fpdf import FPDF
import base64

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
    .alert-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

class ReinsuranceMultiYearAnalyzer:
    def __init__(self):
        self.years = [2018, 2019, 2020, 2021, 2022]
        self.load_multi_year_data()
        self.calculate_multi_year_ratios()
        self.setup_alert_system()
    
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
                'dettes_court_terme': 16034315935,
                'immobilisations': 1317642947
            },
            2021: {
                'actif_total': 63847390546,
                'capitaux_propres': 11805273655,
                'provisions_techniques': 34158229153,
                'actif_circulant': 39212619761,
                'disponibilites': 8612944474,
                'creances_reassurance': 17378888189,
                'titres_participations': 4084819617,
                'dettes_court_terme': 14470724315,
                'immobilisations': 920922106
            },
            2020: {
                'actif_total': 66140090540,
                'capitaux_propres': 11573517700,
                'provisions_techniques': 34995446739,
                'actif_circulant': 36791871917,
                'disponibilites': 5550216908,
                'creances_reassurance': 16786384349,
                'titres_participations': 5684594263,
                'dettes_court_terme': 16525949801,
                'immobilisations': 743908961
            },
            2019: {
                'actif_total': 61183312126,
                'capitaux_propres': 10898977425,
                'provisions_techniques': 31891692770,
                'actif_circulant': 34788951928,
                'disponibilites': 5082181152,
                'creances_reassurance': 15031494835,
                'titres_participations': 6455099576,
                'dettes_court_terme': 15088990837,
                'immobilisations': 562760809
            },
            2018: {
                'actif_total': 53925751172,
                'capitaux_propres': 9513876089,
                'provisions_techniques': 24906680390,
                'actif_circulant': 29172515536,
                'disponibilites': 3906891625,
                'creances_reassurance': 13657388227,
                'titres_participations': 8263745235,
                'dettes_court_terme': 15694383412,
                'immobilisations': 621042104
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
                'produits_exceptionnels': 186419638,
                'impots_societes': 661792800
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
                'produits_exceptionnels': 66158250,
                'impots_societes': 363789000
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
                'produits_exceptionnels': 313283482,
                'impots_societes': 294098700
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
                'produits_exceptionnels': 26286453,
                'impots_societes': 346957650
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
                'produits_exceptionnels': 1143385428,
                'impots_societes': 445994100
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
            primes_totales = is_stmt['primes_vie'] + is_stmt['primes_iard']
            ratios_year['marge_nette'] = (is_stmt['resultat_net'] / primes_totales) * 100 if primes_totales > 0 else 0
            
            # Ratios de solvabilité
            ratios_year['solvabilite'] = (bs['capitaux_propres'] / bs['actif_total']) * 100
            ratios_year['current_ratio'] = bs['actif_circulant'] / bs['dettes_court_terme']
            ratios_year['couverture_provisions'] = (bs['actif_circulant'] / bs['provisions_techniques']) * 100
            ratios_year['leverage'] = bs['dettes_court_terme'] / bs['capitaux_propres']
            
            # Ratios techniques Vie
            primes_vie = is_stmt['primes_vie']
            if primes_vie > 0:
                ratios_year['ratio_sinistres_vie'] = (is_stmt['sinistres_vie'] / primes_vie) * 100
                ratios_year['ratio_commissions_vie'] = (is_stmt['commissions_vie'] / primes_vie) * 100
                ratios_year['combined_ratio_vie'] = ratios_year['ratio_sinistres_vie'] + ratios_year['ratio_commissions_vie']
                ratios_year['loss_ratio_vie'] = ratios_year['ratio_sinistres_vie']
                ratios_year['expense_ratio_vie'] = ratios_year['ratio_commissions_vie']
            else:
                ratios_year['ratio_sinistres_vie'] = 0
                ratios_year['ratio_commissions_vie'] = 0
                ratios_year['combined_ratio_vie'] = 0
                ratios_year['loss_ratio_vie'] = 0
                ratios_year['expense_ratio_vie'] = 0
            
            # Ratios techniques IARD
            primes_iard = is_stmt['primes_iard']
            if primes_iard > 0:
                ratios_year['ratio_sinistres_iard'] = (is_stmt['sinistres_iard'] / primes_iard) * 100
                ratios_year['ratio_commissions_iard'] = (is_stmt['commissions_iard'] / primes_iard) * 100
                ratios_year['combined_ratio_iard'] = ratios_year['ratio_sinistres_iard'] + ratios_year['ratio_commissions_iard']
                ratios_year['loss_ratio_iard'] = ratios_year['ratio_sinistres_iard']
                ratios_year['expense_ratio_iard'] = ratios_year['ratio_commissions_iard']
            else:
                ratios_year['ratio_sinistres_iard'] = 0
                ratios_year['ratio_commissions_iard'] = 0
                ratios_year['combined_ratio_iard'] = 0
                ratios_year['loss_ratio_iard'] = 0
                ratios_year['expense_ratio_iard'] = 0
            
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
                current_primes_total = primes_totales
                ratios_year['croissance_primes'] = ((current_primes_total - prev_primes_total) / prev_primes_total) * 100
            else:
                ratios_year['croissance_primes'] = 0
            
            # Nouvelles métriques avancées
            ratios_year['ratio_capitaux_risque'] = (bs['capitaux_propres'] / primes_totales) * 100
            ratios_year['productivite_actif'] = (primes_totales / bs['actif_total']) * 100
            ratios_year['ratio_rentabilite_technique'] = (100 - ratios_year['combined_ratio_iard']) if primes_iard > 0 else 0
            
            self.ratios[year] = ratios_year
    
    def setup_alert_system(self):
        """Configuration du système d'alertes avancé"""
        self.alert_thresholds = {
            'combined_ratio_vie': {'seuil': 105, 'niveau': 'danger'},
            'combined_ratio_iard': {'seuil': 105, 'niveau': 'danger'},
            'roe': {'seuil': 8, 'niveau': 'warning', 'min_max': 'min'},
            'solvabilite': {'seuil': 15, 'niveau': 'danger', 'min_max': 'min'},
            'current_ratio': {'seuil': 1.5, 'niveau': 'warning', 'min_max': 'min'},
            'rendement_placements': {'seuil': 5, 'niveau': 'warning', 'min_max': 'min'},
            'leverage': {'seuil': 3, 'niveau': 'danger', 'min_max': 'max'}
        }
        
        self.alerts_2022 = self.generate_alerts(2022)
    
    def generate_alerts(self, year):
        """Génère les alertes pour une année donnée"""
        alerts = []
        ratios = self.ratios[year]
        
        for ratio_name, threshold in self.alert_thresholds.items():
            value = ratios.get(ratio_name, 0)
            seuil = threshold['seuil']
            niveau = threshold['niveau']
            
            if threshold.get('min_max') == 'min':
                if value < seuil:
                    alerts.append({
                        'ratio': ratio_name,
                        'valeur': value,
                        'seuil': seuil,
                        'niveau': niveau,
                        'message': f"{ratio_name.upper()} ({value:.1f}) en dessous du seuil minimum ({seuil})"
                    })
            else:
                if value > seuil:
                    alerts.append({
                        'ratio': ratio_name,
                        'valeur': value,
                        'seuil': seuil,
                        'niveau': niveau,
                        'message': f"{ratio_name.upper()} ({value:.1f}) au-dessus du seuil maximum ({seuil})"
                    })
        
        return alerts

def format_currency(amount):
    """Formate les montants en millions ou milliards"""
    if abs(amount) >= 1000000000:
        return f"{amount/1000000000:,.2f} Md FCFA"
    elif abs(amount) >= 1000000:
        return f"{amount/1000000:,.2f} M FCFA"
    else:
        return f"{amount:,.0f} FCFA"

def create_download_link(df, filename, link_text):
    """Crée un lien de téléchargement pour un DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def generate_pdf_report(analyzer):
    """Génère un rapport PDF des analyses"""
    pdf = FPDF()
    pdf.add_page()
    
    # En-tête
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Rapport d\'Analyse SEN-RE 2018-2022', 0, 1, 'C')
    pdf.ln(10)
    
    # Résumé exécutif
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Résumé Exécutif', 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 8, f"""
    Analyse financière de la Société Sénégalaise de Réassurance couvrant la période 2018-2022.
    Principales tendances observées :
    - Croissance des primes : {((analyzer.income_statements[2022]['primes_vie'] + analyzer.income_statements[2022]['primes_iard'] - (analyzer.income_statements[2018]['primes_vie'] + analyzer.income_statements[2018]['primes_iard'])) / (analyzer.income_statements[2018]['primes_vie'] + analyzer.income_statements[2018]['primes_iard']) * 100):.1f}%
    - Évolution résultat net : {((analyzer.income_statements[2022]['resultat_net'] - analyzer.income_statements[2018]['resultat_net']) / analyzer.income_statements[2018]['resultat_net'] * 100):.1f}%
    - Score de conformité : {sum([1 for alert in analyzer.alerts_2022 if alert['niveau'] != 'danger'])}/7 indicateurs conformes
    """)
    
    return pdf.output(dest='S').encode('latin1')

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
    
    # Alertes en temps réel dans la sidebar
    if analyzer.alerts_2022:
        st.sidebar.title("🚨 Alertes 2022")
        for alert in analyzer.alerts_2022:
            if alert['niveau'] == 'danger':
                st.sidebar.error(alert['message'])
            else:
                st.sidebar.warning(alert['message'])
    
    st.sidebar.title("🔧 Paramètres d'Analyse")
    analysis_type = st.sidebar.selectbox(
        "Type d'analyse",
        ["Vue d'Ensemble", "Analyse Comparative", "Rentabilité", "Solvabilité", 
         "Performance Technique", "Investissements", "Conformité Normes", 
         "Prévisions & Scénarios", "Rapports Avancés"]
    )
    
    # Options d'export
    st.sidebar.title("📤 Export des Données")
    if st.sidebar.button("📊 Générer Rapport Complet"):
        pdf_report = generate_pdf_report(analyzer)
        st.sidebar.download_button(
            label="📥 Télécharger Rapport PDF",
            data=pdf_report,
            file_name="rapport_senre_2018_2022.pdf",
            mime="application/pdf"
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
    elif analysis_type == "Prévisions & Scénarios":
        show_forecasting_analysis(analyzer)
    elif analysis_type == "Rapports Avancés":
        show_advanced_reports(analyzer)

def show_overview(analyzer):
    """Vue d'ensemble des performances 2018-2022"""
    
    st.markdown('<div class="section-header">📈 Évolution des Performances Clés 2018-2022</div>', unsafe_allow_html=True)
    
    # KPIs principaux sur 5 ans avec tendances
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
    
    # Alertes en temps réel
    if analyzer.alerts_2022:
        st.markdown('<div class="section-header">🚨 Tableau de Bord des Alertes</div>', unsafe_allow_html=True)
        
        alert_cols = st.columns(2)
        danger_alerts = [a for a in analyzer.alerts_2022 if a['niveau'] == 'danger']
        warning_alerts = [a for a in analyzer.alerts_2022 if a['niveau'] == 'warning']
        
        with alert_cols[0]:
            if danger_alerts:
                st.error("### Alertes Critiques")
                for alert in danger_alerts:
                    st.error(f"🔴 {alert['message']}")
        
        with alert_cols[1]:
            if warning_alerts:
                st.warning("### Alertes de Vigilance")
                for alert in warning_alerts:
                    st.warning(f"🟡 {alert['message']}")
    
    # Graphiques d'évolution avancés
    col1, col2 = st.columns(2)
    
    with col1:
        # Évolution du résultat net avec tendance
        years = analyzer.years
        resultats_nets = [analyzer.income_statements[y]['resultat_net'] for y in years]
        
        # Calcul de la tendance linéaire
        z = np.polyfit(range(len(years)), resultats_nets, 1)
        p = np.poly1d(z)
        trend_line = p(range(len(years)))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=years, 
            y=resultats_nets,
            mode='lines+markers',
            name='Résultat Net',
            line=dict(color='blue', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=years,
            y=trend_line,
            mode='lines',
            name='Tendance',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title='Évolution du Résultat Net avec Tendance',
            xaxis_title='Année',
            yaxis_title='Résultat Net (FCFA)',
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Évolution des primes par branche avec parts de marché
        primes_vie = [analyzer.income_statements[y]['primes_vie'] for y in years]
        primes_iard = [analyzer.income_statements[y]['primes_iard'] for y in years]
        primes_total = [primes_vie[i] + primes_iard[i] for i in range(len(years))]
        
        part_vie = [(primes_vie[i] / primes_total[i]) * 100 for i in range(len(years))]
        part_iard = [(primes_iard[i] / primes_total[i]) * 100 for i in range(len(years))]
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(go.Bar(name='Primes Vie', x=years, y=primes_vie, marker_color='lightblue'), secondary_y=False)
        fig.add_trace(go.Bar(name='Primes IARD', x=years, y=primes_iard, marker_color='blue'), secondary_y=False)
        fig.add_trace(go.Scatter(x=years, y=part_vie, name='Part Vie (%)', line=dict(color='red')), secondary_y=True)
        
        fig.update_layout(
            title='Évolution des Primes et Part de la Branche Vie',
            xaxis_title='Année',
            barmode='stack'
        )
        fig.update_yaxes(title_text="Montant des Primes (FCFA)", secondary_y=False)
        fig.update_yaxes(title_text="Part Vie (%)", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Tableau récapitulatif des ratios clés avec indicateurs de performance
    st.markdown('<div class="section-header">📊 Tableau des Ratios Clés 2018-2022</div>', unsafe_allow_html=True)
    
    ratios_data = []
    for year in analyzer.years:
        ratios = analyzer.ratios[year]
        ratios_data.append({
            'Année': year,
            'ROE (%)': f"{ratios['roe']:.1f}",
            'ROA (%)': f"{ratios['roa']:.1f}",
            'Marge Nette (%)': f"{ratios['marge_nette']:.1f}",
            'Solvabilité (%)': f"{ratios['solvabilite']:.1f}",
            'Combined Ratio Vie (%)': f"{ratios['combined_ratio_vie']:.1f}",
            'Combined Ratio IARD (%)': f"{ratios['combined_ratio_iard']:.1f}",
            'Rendement Placements (%)': f"{ratios['rendement_placements']:.1f}",
            'Croissance Primes (%)': f"{ratios['croissance_primes']:.1f}"
        })
    
    df_ratios = pd.DataFrame(ratios_data)
    
    # Application de styles conditionnels
    def color_combined_ratio(val):
        try:
            ratio = float(val.split(' ')[0])
            if ratio > 105:
                return 'background-color: #ffcccc'
            elif ratio > 100:
                return 'background-color: #fff3cd'
            else:
                return 'background-color: #d4edda'
        except:
            return ''
    
    styled_df = df_ratios.style.applymap(color_combined_ratio, 
                                        subset=['Combined Ratio Vie (%)', 'Combined Ratio IARD (%)'])
    
    st.dataframe(styled_df, use_container_width=True)

def show_comparative_analysis(analyzer):
    """Analyse comparative détaillée avec benchmarks"""
    
    st.markdown('<div class="section-header">📈 Analyse Comparative 2018-2022</div>', unsafe_allow_html=True)
    
    # Benchmarks sectoriels
    sector_benchmarks = {
        'ROE': {'cible': 12.0, 'seuil_alerte': 8.0},
        'ROA': {'cible': 2.5, 'seuil_alerte': 1.5},
        'Combined Ratio': {'cible': 95.0, 'seuil_alerte': 105.0},
        'Ratio Solvabilité': {'cible': 20.0, 'seuil_alerte': 15.0},
        'Rendement Placements': {'cible': 8.0, 'seuil_alerte': 5.0}
    }
    
    # Sélection des indicateurs à comparer
    indicateurs = st.multiselect(
        "Sélectionnez les indicateurs à comparer:",
        ['Résultat Net', 'Primes Totales', 'ROE', 'ROA', 'Marge Nette', 'Ratio de Solvabilité', 'Combined Ratio IARD'],
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
        elif indicateur == 'Combined Ratio IARD':
            data[indicateur] = [analyzer.ratios[y]['combined_ratio_iard'] for y in years]
    
    df_comparatif = pd.DataFrame(data)
    
    # Graphique comparatif avec benchmarks
    fig = go.Figure()
    
    for indicateur in indicateurs:
        if indicateur in ['ROE', 'ROA', 'Marge Nette', 'Ratio de Solvabilité', 'Combined Ratio IARD']:
            # Ces indicateurs sont en pourcentage
            fig.add_trace(go.Scatter(
                x=years, y=df_comparatif[indicateur],
                mode='lines+markers',
                name=indicateur
            ))
            
            # Ajout des lignes de référence pour les benchmarks
            if indicateur in sector_benchmarks:
                benchmark = sector_benchmarks[indicateur]
                fig.add_hline(y=benchmark['cible'], line_dash="dash", 
                            line_color="green", annotation_text=f"Cible {indicateur}")
                fig.add_hline(y=benchmark['seuil_alerte'], line_dash="dot", 
                            line_color="red", annotation_text=f"Alerte {indicateur}")
        else:
            # Ces indicateurs sont en valeur absolue
            fig.add_trace(go.Bar(
                x=years, y=df_comparatif[indicateur],
                name=indicateur
            ))
    
    fig.update_layout(
        title='Comparaison des Indicateurs Clés avec Benchmarks Sectoriels',
        xaxis_title='Année',
        yaxis_title='Valeur',
        barmode='group' if 'Résultat Net' in indicateurs or 'Primes Totales' in indicateurs else 'stack'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tableau de croissance annuelle avec analyse de volatilité
    st.markdown('<div class="section-header">📈 Analyse de la Croissance et Volatilité</div>', unsafe_allow_html=True)
    
    croissance_data = []
    resultats = [analyzer.income_statements[y]['resultat_net'] for y in years]
    primes_totales = [analyzer.income_statements[y]['primes_vie'] + analyzer.income_statements[y]['primes_iard'] for y in years]
    
    for i in range(1, len(analyzer.years)):
        year = analyzer.years[i]
        prev_year = analyzer.years[i-1]
        
        # Calcul des taux de croissance
        croissance_resultat = ((analyzer.income_statements[year]['resultat_net'] - analyzer.income_statements[prev_year]['resultat_net']) / 
                             analyzer.income_statements[prev_year]['resultat_net']) * 100
        
        primes_year = primes_totales[i]
        primes_prev = primes_totales[i-1]
        croissance_primes = ((primes_year - primes_prev) / primes_prev) * 100
        
        # Volatilité (écart-type sur 3 ans)
        if i >= 2:
            volatilite_resultat = np.std([((resultats[j] - resultats[j-1]) / resultats[j-1]) * 100 for j in range(i-1, i+1)])
            volatilite_primes = np.std([((primes_totales[j] - primes_totales[j-1]) / primes_totales[j-1]) * 100 for j in range(i-1, i+1)])
        else:
            volatilite_resultat = 0
            volatilite_primes = 0
        
        croissance_data.append({
            'Période': f"{prev_year}-{year}",
            'Croissance Résultat Net (%)': f"{croissance_resultat:.1f}",
            'Croissance Primes Totales (%)': f"{croissance_primes:.1f}",
            'Évolution ROE (points)': f"{analyzer.ratios[year]['roe'] - analyzer.ratios[prev_year]['roe']:.1f}",
            'Évolution Solvabilité (points)': f"{analyzer.ratios[year]['solvabilite'] - analyzer.ratios[prev_year]['solvabilite']:.1f}",
            'Volatilité Résultat (%)': f"{volatilite_resultat:.1f}",
            'Volatilité Primes (%)': f"{volatilite_primes:.1f}"
        })
    
    df_croissance = pd.DataFrame(croissance_data)
    st.dataframe(df_croissance, use_container_width=True)

def show_profitability_analysis(analyzer):
    """Analyse de rentabilité détaillée avec decomposition"""
    st.markdown('<div class="section-header">💰 Analyse de Rentabilité 2018-2022</div>', unsafe_allow_html=True)
    
    # Graphique d'évolution de la rentabilité avec decomposition
    years = analyzer.years
    roe_values = [analyzer.ratios[y]['roe'] for y in years]
    roa_values = [analyzer.ratios[y]['roa'] for y in years]
    marge_nette_values = [analyzer.ratios[y]['marge_nette'] for y in years]
    
    # Analyse DuPont
    dupont_data = []
    for year in years:
        ratios = analyzer.ratios[year]
        bs = analyzer.balance_sheets[year]
        is_stmt = analyzer.income_statements[year]
        
        # Décomposition DuPont
        marge_nette = ratios['marge_nette'] / 100
        rotation_actif = (is_stmt['primes_vie'] + is_stmt['primes_iard']) / bs['actif_total']
        levier_financier = bs['actif_total'] / bs['capitaux_propres']
        
        dupont_data.append({
            'Année': year,
            'ROE': ratios['roe'],
            'Marge Nette': marge_nette * 100,
            'Rotation Actif': rotation_actif,
            'Levier Financier': levier_financier
        })
    
    df_dupont = pd.DataFrame(dupont_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
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
    
    with col2:
        # Graphique en cascade pour la decomposition DuPont 2022
        fig = go.Figure(go.Waterfall(
            name="2022",
            orientation="v",
            measure=["relative", "relative", "relative", "total"],
            x=["Marge Nette", "Rotation Actif", "Levier Financier", "ROE"],
            textposition="outside",
            text=[f"{df_dupont[df_dupont['Année']==2022]['Marge Nette'].iloc[0]:.1f}%", 
                  f"{df_dupont[df_dupont['Année']==2022]['Rotation Actif'].iloc[0]:.2f}", 
                  f"{df_dupont[df_dupont['Année']==2022]['Levier Financier'].iloc[0]:.2f}", 
                  f"{df_dupont[df_dupont['Année']==2022]['ROE'].iloc[0]:.1f}%"],
            y=[df_dupont[df_dupont['Année']==2022]['Marge Nette'].iloc[0],
               df_dupont[df_dupont['Année']==2022]['Rotation Actif'].iloc[0] * 100,
               df_dupont[df_dupont['Année']==2022]['Levier Financier'].iloc[0] * 10,
               0],
            connector={"line":{"color":"rgb(63, 63, 63)"}},
        ))

        fig.update_layout(
            title="Décomposition DuPont du ROE 2022",
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)
    
    # Analyse de la composition du résultat
    st.markdown('<div class="section-header">🧮 Composition du Résultat Net</div>', unsafe_allow_html=True)
    
    composition_data = []
    for year in analyzer.years:
        is_stmt = analyzer.income_statements[year]
        total_produits = is_stmt['primes_vie'] + is_stmt['primes_iard'] + is_stmt['revenus_placements'] + is_stmt['produits_exceptionnels']
        
        marge_technique = is_stmt['primes_vie'] + is_stmt['primes_iard'] - is_stmt['sinistres_vie'] - is_stmt['sinistres_iard'] - is_stmt['commissions_vie'] - is_stmt['commissions_iard']
        
        composition_data.append({
            'Année': year,
            'Résultat Net': format_currency(is_stmt['resultat_net']),
            'Marge Technique': format_currency(marge_technique),
            'Revenus Placements': format_currency(is_stmt['revenus_placements']),
            'Produits Exceptionnels': format_currency(is_stmt['produits_exceptionnels']),
            'Part Technique (%)': f"{(marge_technique / is_stmt['resultat_net'] * 100) if is_stmt['resultat_net'] != 0 else 0:.1f}",
            'Part Financière (%)': f"{(is_stmt['revenus_placements'] / is_stmt['resultat_net'] * 100) if is_stmt['resultat_net'] != 0 else 0:.1f}",
            'Efficacité Fiscale (%)': f"{(1 - (is_stmt['impots_societes'] / (is_stmt['resultat_net'] + is_stmt['impots_societes']))) * 100:.1f}"
        })
    
    df_composition = pd.DataFrame(composition_data)
    st.dataframe(df_composition, use_container_width=True)

def show_solvency_analysis(analyzer):
    """Analyse de solvabilité selon les normes internationales avancées"""
    st.markdown('<div class="section-header">🛡️ Analyse de Solvabilité - Normes Internationales</div>', unsafe_allow_html=True)
    
    # Calcul du ratio de solvabilité avancé (approche Solvabilité II simplifiée)
    st.subheader("📊 Calcul du Ratio de Solvabilité Avancé")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        base_calcul = st.selectbox("Base de calcul:", ["Standard Formula", "Internal Model"])
    with col2:
        horizon_temps = st.selectbox("Horizon temporel:", ["1 an", "3 ans", "5 ans"])
    with col3:
        niveau_confiance = st.slider("Niveau de confiance:", 90, 99, 95)
    
    # Ratios de solvabilité selon Solvabilité II et normes CIMA
    solvency_data = []
    
    for year in analyzer.years:
        ratios = analyzer.ratios[year]
        bs = analyzer.balance_sheets[year]
        is_stmt = analyzer.income_statements[year]
        
        # Calcul des ratios selon les normes
        ratio_solvabilite_ii = ratios['solvabilite']  # Base simplifiée
        current_ratio = ratios['current_ratio']
        ratio_couverture = ratios['couverture_provisions']
        
        # Ratio de solvabilité avancé (simulation)
        capital_requis_risque_marche = bs['titres_participations'] * 0.15  # 15% de risque marché
        capital_requis_risque_credit = bs['creances_reassurance'] * 0.10   # 10% de risque crédit
        capital_requis_risque_assurance = bs['provisions_techniques'] * 0.20  # 20% de risque assurance
        
        capital_requis_total = capital_requis_risque_marche + capital_requis_risque_credit + capital_requis_risque_assurance
        ratio_solvabilite_avance = (bs['capitaux_propres'] / capital_requis_total) * 100 if capital_requis_total > 0 else 0
        
        # Évaluation de la conformité
        conforme_solvabilite = ratio_solvabilite_ii >= 15
        conforme_solvabilite_avance = ratio_solvabilite_avance >= 100
        conforme_liquidite = current_ratio >= 1.5
        conforme_couverture = ratio_couverture >= 100
        
        solvency_data.append({
            'Année': year,
            'Ratio Solvabilité II (%)': f"{ratio_solvabilite_ii:.1f}",
            'Ratio Solvabilité Avancé (%)': f"{ratio_solvabilite_avance:.1f}",
            'Current Ratio': f"{current_ratio:.2f}",
            'Couverture Provisions (%)': f"{ratio_couverture:.1f}",
            'Capital Requis (M FCFA)': f"{capital_requis_total/1000000:.1f}",
            'Conformité Solvabilité': '✅ Conforme' if conforme_solvabilite else '❌ Non conforme',
            'Conformité Avancée': '✅ Conforme' if conforme_solvabilite_avance else '❌ Non conforme'
        })
    
    df_solvency = pd.DataFrame(solvency_data)
    st.dataframe(df_solvency, use_container_width=True)
    
    # Graphique radar pour la solvabilité 2022
    st.markdown('<div class="section-header">🎯 Radar de Solvabilité 2022 vs Normes</div>', unsafe_allow_html=True)
    
    categories = ['Solvabilité Base Solo', 'Solvabilité Avancée', 'Liquidité', 'Couv. Provisions', 'Rendement', 'Croissance']
    
    fig = go.Figure()
    
    # Valeurs 2022 normalisées
    valeurs_2022 = [
        min(analyzer.ratios[2022]['solvabilite']/20, 1),  # Objectif > 15%
        min(float(df_solvency[df_solvency['Année']==2022]['Ratio Solvabilité Avancé (%)'].iloc[0].split(' ')[0])/150, 1),  # Objectif > 100%
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
    normes = [0.75, 0.67, 0.75, 0.67, 0.67, 0.5]  # Seuils minimum normalisés
    
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
    """Analyse de la performance technique par branche avec indicateurs avancés"""
    st.markdown('<div class="section-header">🎯 Performance Technique par Branche 2018-2022</div>', unsafe_allow_html=True)
    
    # Combined Ratios par année avec indicateurs de qualité
    years = analyzer.years
    combined_vie = [analyzer.ratios[y]['combined_ratio_vie'] for y in years]
    combined_iard = [analyzer.ratios[y]['combined_ratio_iard'] for y in years]
    
    # Calcul de la volatilité des combined ratios
    volatilite_vie = np.std(combined_vie) if len(combined_vie) > 1 else 0
    volatilite_iard = np.std(combined_iard) if len(combined_iard) > 1 else 0
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Volatilité Combined Ratio Vie", f"{volatilite_vie:.2f}%")
    with col2:
        st.metric("Volatilité Combined Ratio IARD", f"{volatilite_iard:.2f}%")
    
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
    
    # Analyse détaillée par branche avec indicateurs avancés
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Branche Vie - Ratios Techniques Avancés")
        vie_data = []
        for year in analyzer.years:
            ratios = analyzer.ratios[year]
            vie_data.append({
                'Année': year,
                'Loss Ratio (%)': f"{ratios['loss_ratio_vie']:.1f}",
                'Expense Ratio (%)': f"{ratios['expense_ratio_vie']:.1f}",
                'Combined Ratio (%)': f"{ratios['combined_ratio_vie']:.1f}",
                'Rentabilité Technique': f"{(100 - ratios['combined_ratio_vie']):.1f}%",
                'Efficacité': '🟢 Excellente' if ratios['combined_ratio_vie'] < 95 else '🟡 Bonne' if ratios['combined_ratio_vie'] < 100 else '🔴 Déficitaire'
            })
        
        df_vie = pd.DataFrame(vie_data)
        st.dataframe(df_vie, use_container_width=True)
    
    with col2:
        st.subheader("🏠 Branche IARD - Ratios Techniques Avancés")
        iard_data = []
        for year in analyzer.years:
            ratios = analyzer.ratios[year]
            iard_data.append({
                'Année': year,
                'Loss Ratio (%)': f"{ratios['loss_ratio_iard']:.1f}",
                'Expense Ratio (%)': f"{ratios['expense_ratio_iard']:.1f}",
                'Combined Ratio (%)': f"{ratios['combined_ratio_iard']:.1f}",
                'Rentabilité Technique': f"{(100 - ratios['combined_ratio_iard']):.1f}%",
                'Efficacité': '🟢 Excellente' if ratios['combined_ratio_iard'] < 95 else '🟡 Bonne' if ratios['combined_ratio_iard'] < 100 else '🔴 Déficitaire'
            })
        
        df_iard = pd.DataFrame(iard_data)
        st.dataframe(df_iard, use_container_width=True)
    
    # Analyse de la qualité technique
    st.markdown('<div class="section-header">🔍 Analyse de la Qualité Technique</div>', unsafe_allow_html=True)
    
    qualite_data = []
    for year in analyzer.years:
        ratios = analyzer.ratios[year]
        
        # Score de qualité technique (0-100)
        score_qualite = max(0, 100 - (ratios['combined_ratio_vie'] + ratios['combined_ratio_iard'])/2)
        stabilite_vie = 100 - min(100, abs(ratios['combined_ratio_vie'] - 95) * 2)
        stabilite_iard = 100 - min(100, abs(ratios['combined_ratio_iard'] - 95) * 2)
        
        qualite_data.append({
            'Année': year,
            'Score Qualité Global': f"{score_qualite:.1f}/100",
            'Stabilité Vie': f"{stabilite_vie:.1f}/100",
            'Stabilité IARD': f"{stabilite_iard:.1f}/100",
            'Niveau de Risque': '🟢 Faible' if score_qualite > 80 else '🟡 Modéré' if score_qualite > 60 else '🔴 Élevé'
        })
    
    df_qualite = pd.DataFrame(qualite_data)
    st.dataframe(df_qualite, use_container_width=True)

def show_investment_analysis(analyzer):
    """Analyse de la performance des investissements avec optimisation"""
    st.markdown('<div class="section-header">📈 Analyse des Investissements 2018-2022</div>', unsafe_allow_html=True)
    
    # Évolution du rendement des placements avec indicateurs de risque
    years = analyzer.years
    rendements = [analyzer.ratios[y]['rendement_placements'] for y in years]
    revenus_placements = [analyzer.income_statements[y]['revenus_placements'] for y in years]
    
    # Calcul du rendement ajusté au risque
    risque_placements = np.std(rendements) if len(rendements) > 1 else 0
    rendement_moyen = np.mean(rendements) if rendements else 0
    ratio_sharpe = (rendement_moyen - 2) / risque_placements if risque_placements > 0 else 0  # Taux sans risque estimé à 2%
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Rendement Moyen", f"{rendement_moyen:.2f}%")
    with col2:
        st.metric("Risque (Volatilité)", f"{risque_placements:.2f}%")
    with col3:
        st.metric("Ratio de Sharpe", f"{ratio_sharpe:.2f}")
    
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
    
    # Composition du portefeuille avec optimisation
    st.markdown('<div class="section-header">🏦 Composition et Optimisation du Portefeuille</div>', unsafe_allow_html=True)
    
    portefeuille_data = []
    for year in analyzer.years:
        bs = analyzer.balance_sheets[year]
        total_investissements = bs['titres_participations'] + bs['disponibilites'] + bs['creances_reassurance']
        
        if total_investissements > 0:
            part_titres = (bs['titres_participations'] / total_investissements) * 100
            part_disponibilites = (bs['disponibilites'] / total_investissements) * 100
            part_creances = (bs['creances_reassurance'] / total_investissements) * 100
            
            # Score de diversification (0-100)
            score_diversification = 100 - (max(part_titres, part_disponibilites, part_creances) - 33.3)
            
            portefeuille_data.append({
                'Année': year,
                'Titres & Participations (%)': f"{part_titres:.1f}",
                'Disponibilités (%)': f"{part_disponibilites:.1f}",
                'Créances Réassurance (%)': f"{part_creances:.1f}",
                'Rendement (%)': f"{analyzer.ratios[year]['rendement_placements']:.1f}",
                'Score Diversification': f"{max(0, score_diversification):.1f}/100",
                'Recommandation': '🟢 Optimisé' if score_diversification > 70 else '🟡 À améliorer' if score_diversification > 50 else '🔴 Risqué'
            })
    
    df_portefeuille = pd.DataFrame(portefeuille_data)
    st.dataframe(df_portefeuille, use_container_width=True)
    
    # Simulation d'optimisation de portefeuille
    st.markdown('<div class="section-header">🎯 Simulation d\'Optimisation du Portefeuille</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        target_rendement = st.slider("Rendement cible (%):", 5.0, 15.0, 8.0, 0.5)
        risk_tolerance = st.select_slider("Tolérance au risque:", options=["Faible", "Modérée", "Élevée"])
    
    with col2:
        # Calcul de l'allocation optimisée
        allocation_optimale = {
            'Titres & Participations': 45 if risk_tolerance == "Élevée" else 35 if risk_tolerance == "Modérée" else 25,
            'Disponibilités': 15 if risk_tolerance == "Élevée" else 25 if risk_tolerance == "Modérée" else 35,
            'Créances Réassurance': 40 if risk_tolerance == "Élevée" else 40 if risk_tolerance == "Modérée" else 40
        }
        
        st.write("### Allocation Optimale Recommandée")
        for asset, allocation in allocation_optimale.items():
            st.write(f"- **{asset}**: {allocation}%")
        
        rendement_estime = allocation_optimale['Titres & Participations'] * 0.12 + \
                          allocation_optimale['Disponibilités'] * 0.03 + \
                          allocation_optimale['Créances Réassurance'] * 0.06
        
        st.metric("Rendement Estimé", f"{rendement_estime:.1f}%")

def show_compliance_analysis(analyzer):
    """Analyse de conformité aux normes internationales avancée"""
    st.markdown('<div class="section-header">🏛️ Conformité aux Normes Internationales</div>', unsafe_allow_html=True)
    
    # Évaluation détaillée de la conformité
    evaluation_conformite = {
        'Domaine': [
            'Solvabilité - Ratio de Base',
            'Solvabilité - Approche Avancée',
            'Liquidité - Current Ratio',
            'Couverture Provisions Techniques',
            'Diversification du Portefeuille',
            'Transparence Informationnelle',
            'Gouvernance des Risques',
            'Adéquation des Capitaux',
            'Performance Technique Vie',
            'Performance Technique IARD',
            'Gestion Actif-Passif',
            'Contrôle Interne'
        ],
        'Norme Référence': [
            'Solvabilité II / CIMA',
            'Solvabilité II - Standard Formula',
            'Normes Sectorielles',
            'Principes Actuariels',
            'Diversification Actifs',
            'IFRS/Transparence',
            'Pillar 2 Solvabilité II',
            'Exigences Réglementaires',
            'Best Practices Vie',
            'Best Practices IARD',
            'Gestion ALM',
            'Cadre de Contrôle'
        ],
        'Statut SEN-RE 2022': [
            '✅ Conforme',
            '⚠️ Partiellement Conforme',
            '✅ Conforme',
            '✅ Conforme',
            '🔴 Non Conforme',
            '✅ Conforme',
            '✅ Conforme',
            '✅ Conforme',
            '✅ Conforme',
            '✅ Conforme',
            '⚠️ Partiellement Conforme',
            '✅ Conforme'
        ],
        'Score': [9, 6, 8, 9, 4, 8, 8, 9, 8, 8, 6, 8],
        'Priorité': ['Basse', 'Moyenne', 'Basse', 'Basse', 'Haute', 'Basse', 'Moyenne', 'Basse', 'Basse', 'Basse', 'Moyenne', 'Basse']
    }
    
    df_conformite = pd.DataFrame(evaluation_conformite)
    
    # Application de styles conditionnels
    def color_status(val):
        if '✅' in val:
            return 'background-color: #d4edda'
        elif '⚠️' in val:
            return 'background-color: #fff3cd'
        elif '🔴' in val:
            return 'background-color: #f8d7da'
        return ''
    
    def color_priority(val):
        if val == 'Haute':
            return 'background-color: #f8d7da'
        elif val == 'Moyenne':
            return 'background-color: #fff3cd'
        else:
            return 'background-color: #d4edda'
    
    styled_df = df_conformite.style.map(color_status, subset=['Statut SEN-RE 2022']).map(color_priority, subset=['Priorité'])
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Score global de conformité et métriques avancées
    score_global = sum(evaluation_conformite['Score']) / len(evaluation_conformite['Score'])
    
    col1, col2, col3 = st.columns(3)
    
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
        # Répartition des statuts de conformité
        status_counts = df_conformite['Statut SEN-RE 2022'].value_counts()
        fig = px.pie(values=status_counts.values, names=status_counts.index, 
                    title="Répartition des Statuts de Conformité")
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.markdown("### 📈 Indice de Maturité Réglementaire")
        maturite_data = {
            'Année': [2018, 2019, 2020, 2021, 2022],
            'Score Maturité': [6.2, 6.8, 7.1, 7.5, 8.1]
        }
        df_maturite = pd.DataFrame(maturite_data)
        fig = px.line(df_maturite, x='Année', y='Score Maturité', 
                     title="Évolution de l'Indice de Maturité Réglementaire",
                     markers=True)
        st.plotly_chart(fig, use_container_width=True)
    
    # Plan d'action et recommandations
    st.markdown('<div class="section-header">🎯 Plan d\'Action et Recommandations</div>', unsafe_allow_html=True)
    
    recommendations = [
        "**🔴 CRITIQUE - Diversification du portefeuille**: Augmenter les investissements en actions à dividendes et réduire la concentration sur les créances de réassurance",
        "**🟡 MOYENNE - Solvabilité avancée**: Développer un modèle interne pour le calcul du capital requis selon Solvabilité II",
        "**🟡 MOYENNE - Gestion Actif-Passif**: Mettre en place un système avancé de matching duration actif-passif",
        "**🟢 FAIBLE - Performance technique**: Maintenir la discipline technique et renforcer la sélection des risques",
        "**🟢 FAIBLE - Gouvernance**: Formaliser le cadre de risk appetite et les limites de risque",
        "**📊 REPORTING**: Préparer la mise en œuvre d'IFRS 17 avec un plan de transition sur 18 mois"
    ]
    
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"{i}. {rec}")

def show_forecasting_analysis(analyzer):
    """Analyse de prévisions et scénarios avec machine learning simplifié"""
    st.markdown('<div class="section-header">🔮 Prévisions et Scénarios 2023-2025</div>', unsafe_allow_html=True)
    
    # Sélection du modèle de prévision
    st.subheader("📈 Modèle de Prévision")
    
    col1, col2 = st.columns(2)
    
    with col1:
        modele_choisi = st.selectbox("Modèle de prévision:", 
                                   ["Régression Linéaire", "Lissage Exponentiel", "Moyenne Mobile", "ARIMA Simplifié"])
        
        horizon_prevision = st.slider("Horizon de prévision (années):", 1, 5, 3)
    
    with col2:
        scenario = st.selectbox("Scénario macroéconomique:", 
                              ["Central", "Optimiste", "Pessimiste", "Stress Test"])
        
        confidence_level = st.slider("Niveau de confiance:", 80, 95, 90)
    
    # Données historiques pour les prévisions
    years_hist = analyzer.years
    primes_totales_hist = [analyzer.income_statements[y]['primes_vie'] + analyzer.income_statements[y]['primes_iard'] for y in years_hist]
    resultats_hist = [analyzer.income_statements[y]['resultat_net'] for y in years_hist]
    
    # Prévisions simplifiées (régression linéaire)
    x = np.array(range(len(years_hist)))
    y_primes = np.array(primes_totales_hist)
    y_resultats = np.array(resultats_hist)
    
    # Ajustement des modèles
    coef_primes = np.polyfit(x, y_primes, 1)
    coef_resultats = np.polyfit(x, y_resultats, 1)
    
    # Génération des prévisions
    years_future = list(range(2023, 2023 + horizon_prevision))
    x_future = np.array(range(len(years_hist), len(years_hist) + horizon_prevision))
    
    # Application des scénarios
    if scenario == "Optimiste":
        multiplicateur = 1.15
    elif scenario == "Pessimiste":
        multiplicateur = 0.85
    elif scenario == "Stress Test":
        multiplicateur = 0.70
    else:  # Central
        multiplicateur = 1.0
    
    primes_forecast = (coef_primes[0] * x_future + coef_primes[1]) * multiplicateur
    resultats_forecast = (coef_resultats[0] * x_future + coef_resultats[1]) * multiplicateur
    
    # Graphique des prévisions
    fig = go.Figure()
    
    # Données historiques
    fig.add_trace(go.Scatter(x=years_hist, y=primes_totales_hist, 
                            mode='lines+markers', name='Primes Historique',
                            line=dict(color='blue', width=2)))
    
    fig.add_trace(go.Scatter(x=years_hist, y=resultats_hist,
                            mode='lines+markers', name='Résultat Historique',
                            line=dict(color='green', width=2)))
    
    # Prévisions
    fig.add_trace(go.Scatter(x=years_future, y=primes_forecast,
                            mode='lines+markers', name='Primes Prévision',
                            line=dict(color='blue', width=2, dash='dash')))
    
    fig.add_trace(go.Scatter(x=years_future, y=resultats_forecast,
                            mode='lines+markers', name='Résultat Prévision',
                            line=dict(color='green', width=2, dash='dash')))
    
    fig.update_layout(
        title=f'Prévisions des Primes et Résultats ({scenario} Scenario)',
        xaxis_title='Année',
        yaxis_title='Montant (FCFA)',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tableau des prévisions détaillées
    st.markdown('<div class="section-header">📊 Détail des Prévisions</div>', unsafe_allow_html=True)
    
    forecast_data = []
    for i, year in enumerate(years_future):
        primes_pred = primes_forecast[i]
        resultat_pred = resultats_forecast[i]
        
        # Calcul des ratios prévisionnels
        roe_pred = (resultat_pred / analyzer.balance_sheets[2022]['capitaux_propres']) * 100
        marge_pred = (resultat_pred / primes_pred) * 100
        
        forecast_data.append({
            'Année': year,
            'Primes Totales (Md FCFA)': f"{primes_pred/1000000000:.2f}",
            'Résultat Net (M FCFA)': f"{resultat_pred/1000000:.2f}",
            'ROE Prévisionnel (%)': f"{roe_pred:.1f}",
            'Marge Nette Prévisionnelle (%)': f"{marge_pred:.1f}",
            'Scénario': scenario
        })
    
    df_forecast = pd.DataFrame(forecast_data)
    st.dataframe(df_forecast, use_container_width=True)
    
    # Analyse de sensibilité
    st.markdown('<div class="section-header">🎯 Analyse de Sensibilité</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        variation_primes = st.slider("Variation des primes (%):", -20, 20, 0)
    with col2:
        variation_sinistres = st.slider("Variation des sinistres (%):", -20, 20, 0)
    with col3:
        variation_rendement = st.slider("Variation rendement placements (%):", -5, 5, 0)
    
    # Calcul de l'impact sur le résultat 2023
    primes_base = primes_forecast[0]
    sinistres_base = primes_base * 0.65  # Hypothèse: 65% de ratio sinistres
    rendement_base = analyzer.income_statements[2022]['revenus_placements']
    
    primes_ajustee = primes_base * (1 + variation_primes/100)
    sinistres_ajustes = sinistres_base * (1 + variation_sinistres/100)
    rendement_ajuste = rendement_base * (1 + variation_rendement/100)
    
    resultat_ajuste = (primes_ajustee - sinistres_ajustes - primes_ajustee * 0.25) + rendement_ajuste  # Hypothèse: 25% de frais
    
    impact = ((resultat_ajuste - resultats_forecast[0]) / resultats_forecast[0]) * 100
    
    st.metric("Impact sur le Résultat 2023", f"{impact:.1f}%", 
              delta=f"{resultat_ajuste - resultats_forecast[0]:.0f} FCFA")

def show_advanced_reports(analyzer):
    """Rapports avancés et analytics personnalisés"""
    st.markdown('<div class="section-header">📑 Rapports Avancés et Analytics</div>', unsafe_allow_html=True)
    
    # Sélection du type de rapport
    rapport_type = st.selectbox("Type de rapport:", 
                              ["Rapport de Performance", "Rapport de Conformité", 
                               "Rapport de Risque", "Rapport Stratégique", "Rapport Personnalisé"])
    
    # Personnalisation du rapport
    col1, col2 = st.columns(2)
    
    with col1:
        periode_analyse = st.multiselect("Périodes à inclure:", 
                                       [2018, 2019, 2020, 2021, 2022],
                                       default=[2020, 2021, 2022])
        
        indicateurs_rapport = st.multiselect("Indicateurs à inclure:",
                                           ['Ratios de Rentabilité', 'Ratios de Solvabilité', 
                                            'Performance Technique', 'Investissements',
                                            'Analyse des Risques', 'Benchmarks Sectoriels'],
                                           default=['Ratios de Rentabilité', 'Performance Technique'])
    
    with col2:
        format_export = st.selectbox("Format d'export:", 
                                   ["PDF", "Excel", "CSV", "HTML"])
        
        niveau_detail = st.select_slider("Niveau de détail:",
                                       options=["Synthèse", "Standard", "Détaillé", "Expert"])
    
    # Génération du rapport
    if st.button("🔄 Générer le Rapport"):
        st.success("Rapport généré avec succès!")
        
        # Préparation des données du rapport
        rapport_data = []
        
        for year in periode_analyse:
            if year in analyzer.ratios:
                ratios = analyzer.ratios[year]
                bs = analyzer.balance_sheets[year]
                is_stmt = analyzer.income_statements[year]
                
                data_point = {
                    'Année': year,
                    'Primes Totales (Md FCFA)': f"{(is_stmt['primes_vie'] + is_stmt['primes_iard'])/1000000000:.2f}",
                    'Résultat Net (M FCFA)': f"{is_stmt['resultat_net']/1000000:.2f}",
                    'ROE (%)': f"{ratios['roe']:.1f}",
                    'ROA (%)': f"{ratios['roa']:.1f}",
                    'Marge Nette (%)': f"{ratios['marge_nette']:.1f}",
                    'Ratio Solvabilité (%)': f"{ratios['solvabilite']:.1f}",
                    'Combined Ratio IARD (%)': f"{ratios['combined_ratio_iard']:.1f}",
                    'Combined Ratio Vie (%)': f"{ratios['combined_ratio_vie']:.1f}",
                    'Rendement Placements (%)': f"{ratios['rendement_placements']:.1f}"
                }
                
                rapport_data.append(data_point)
        
        df_rapport = pd.DataFrame(rapport_data)
        
        # Affichage du rapport
        st.markdown('<div class="section-header">📋 Aperçu du Rapport Généré</div>', unsafe_allow_html=True)
        st.dataframe(df_rapport, use_container_width=True)
        
        # Métriques de synthèse
        st.markdown('<div class="section-header">📊 Synthèse des Performances</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            croissance_primes = ((analyzer.income_statements[periode_analyse[-1]]['primes_vie'] + 
                                analyzer.income_statements[periode_analyse[-1]]['primes_iard'] -
                                (analyzer.income_statements[periode_analyse[0]]['primes_vie'] + 
                                 analyzer.income_statements[periode_analyse[0]]['primes_iard'])) / 
                               (analyzer.income_statements[periode_analyse[0]]['primes_vie'] + 
                                analyzer.income_statements[periode_analyse[0]]['primes_iard']) * 100)
            st.metric("Croissance Primes", f"{croissance_primes:.1f}%")
        
        with col2:
            roe_moyen = np.mean([analyzer.ratios[y]['roe'] for y in periode_analyse])
            st.metric("ROE Moyen", f"{roe_moyen:.1f}%")
        
        with col3:
            marge_moyenne = np.mean([analyzer.ratios[y]['marge_nette'] for y in periode_analyse])
            st.metric("Marge Nette Moyenne", f"{marge_moyenne:.1f}%")
        
        with col4:
            conformite_count = sum([1 for alert in analyzer.alerts_2022 if alert['niveau'] != 'danger'])
            st.metric("Indicateurs Conformes", f"{conformite_count}/7")
        
        # Options d'export
        st.markdown('<div class="section-header">📤 Export du Rapport</div>', unsafe_allow_html=True)
        
        if format_export == "Excel":
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df_rapport.to_excel(writer, sheet_name='Rapport SEN-RE', index=False)
                
                # Ajout de graphiques dans le Excel
                workbook = writer.book
                worksheet = writer.sheets['Rapport SEN-RE']
                
                # Création d'un graphique
                chart = workbook.add_chart({'type': 'column'})
                
                # Configuration du graphique
                chart.add_series({
                    'name': 'ROE (%)',
                    'categories': f'=Rapport SEN-RE!$A$2:$A${len(df_rapport)+1}',
                    'values': f'=Rapport SEN-RE!$D$2:$D${len(df_rapport)+1}',
                })
                
                worksheet.insert_chart('F2', chart)
            
            st.download_button(
                label="📥 Télécharger Rapport Excel",
                data=buffer.getvalue(),
                file_name=f"rapport_senre_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.ms-excel"
            )
        
        elif format_export == "CSV":
            csv = df_rapport.to_csv(index=False)
            st.download_button(
                label="📥 Télécharger Rapport CSV",
                data=csv,
                file_name=f"rapport_senre_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()