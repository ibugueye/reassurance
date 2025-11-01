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

# =============================================================================
# PAGE D'ACCUEIL UNIFIÉE
# =============================================================================
if section == "🏠 Accueil & Présentation":
    
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

    # Section Méthodes d'Estimation
    st.markdown("---")
    st.markdown("### 🧮 Méthodes d'Estimation Actuarielles")
    
    col_methods1, col_methods2 = st.columns(2)
    
    with col_methods1:
        st.markdown("""
        <div style='background: #e8f5e8; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;'>
            <h4>📊 Méthode Fréquentiste</h4>
            <p><b>Basée sur l'expérience historique</b></p>
            <p><b>Formule :</b> λ = Σ(sinistres) / Σ(années d'exposition)</p>
            <p><i>Utilise uniquement les données propres de l'assureur</i></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background: #e3f2fd; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;'>
            <h4>🎯 Méthode Bayésienne</h4>
            <p><b>Combinaison expérience propre/collective</b></p>
            <p><b>Formule :</b> P(θ|X) ∝ P(X|θ) × P(θ)</p>
            <p><i>A priori + données = estimation a posteriori</i></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background: #fff3e0; padding: 1.5rem; border-radius: 10px;'>
            <h4>⚖️ Crédibility Theory</h4>
            <p><b>Poids accordé à différentes sources</b></p>
            <p><b>Formule :</b> Z × expérience propre + (1-Z) × expérience collective</p>
            <p><i>Z = facteur de crédibilité (0 ≤ Z ≤ 1)</i></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_methods2:
        st.markdown("""
        <div style='background: #fce4ec; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;'>
            <h4>📈 Modèles de Risque Avancés</h4>
            <ul>
            <li><b>Distribution Log-normale :</b> f(x) = (1/xσ√2π) × exp(-(ln x - μ)²/2σ²)</li>
            <li><b>Distribution Pareto :</b> f(x) = α × θ^α / x^(α+1)</li>
            <li><b>Distribution Gamma :</b> f(x) = x^(k-1) × e^(-x/θ) / (θ^k × Γ(k))</li>
            <li><b>Processus de Poisson :</b> P(N=k) = (λt)^k × e^(-λt) / k!</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background: #e8eaf6; padding: 1.5rem; border-radius: 10px;'>
            <h4>🌪️ Modélisation des Catastrophes Naturelles</h4>
            <ul>
            <li><b>Modèles stochastiques d'événements</b></li>
            <li><b>Analyse de scénarios extrêmes</b></li>
            <li><b>Corrélations géographiques</b></li>
            <li><b>Impact du changement climatique</b></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Section Modèles de Référence et Gestion de Crise
    st.markdown("---")
    st.markdown("### 📊 Modèles de Référence & Gestion de Crise")
    
    col_models1, col_models2 = st.columns(2)
    
    with col_models1:
        st.markdown("""
        <div style='background: #f3e5f5; padding: 1.5rem; border-radius: 10px;'>
            <h4>🏢 Modèles de Référence Internationaux</h4>
            
            <h5>🔬 RMS (Risk Management Solutions)</h5>
            <ul>
            <li>Modélisation probabiliste des catastrophes</li>
            <li>Couverture mondiale tremblements de terre, ouragans</li>
            <li>Évaluation des pertes agrégées</li>
            </ul>
            
            <h5>📊 AIR (Applied Insurance Research)</h5>
            <ul>
            <li>Modèles climatiques avancés</li>
            <li>Analyse de vulnérabilité des constructions</li>
            <li>Scénarios de changement climatique</li>
            </ul>
            
            <h5>🌍 EQECAT</h5>
            <ul>
            <li>Spécialiste modèles sismiques</li>
            <li>Risques tsunami et géotechniques</li>
            <li>Analyse de sol et amplification sismique</li>
            </ul>
            
            <h5>💼 Modèles Propriétaires</h5>
            <ul>
            <li>Développés en interne par les grands réassureurs</li>
            <li>Avantage concurrentiel</li>
            <li>Adaptation aux portefeuilles spécifiques</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col_models2:
        st.markdown("""
        <div style='background: #e8f5e8; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;'>
            <h4>🛡️ Gestion de Crise</h4>
            <ul>
            <li><b>Plans de continuité d'activité</b></li>
            <li><b>Cellules de crise dédiées</b></li>
            <li><b>Communication avec les autorités</b></li>
            <li><b>Gestion des sinistres catastrophiques</b></li>
            <li><b>Respect des délais réglementaires</b></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background: #e3f2fd; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;'>
            <h4>💰 Résultat Technique</h4>
            <p><b>Formule :</b> Résultat Technique = Produits Techniques - Charges Techniques</p>
            <p><b>Détail :</b> (Primes + Sinistres à charge du réassureur) - (Sinistres + Provisions + Frais d'acquisition)</p>
            <p><i>Indicateur clé de performance de l'activité d'assurance</i></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background: #fff3e0; padding: 1.5rem; border-radius: 10px;'>
            <h4>🛡️ Solvabilité II & Simulation SCR</h4>
            <p><b>Capital Requirement (SCR) = Value at Risk 99.5% sur 1 an</b></p>
            <ul>
            <li><b>Module risque de souscription</b></li>
            <li><b>Module risque de marché</b></li>
            <li><b>Module risque de contrepartie</b></li>
            <li><b>Module risque opérationnel</b></li>
            </ul>
            <p><i>Simulations Monte Carlo pour calculer le capital requis</i></p>
        </div>
        """, unsafe_allow_html=True)

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

    # Diagramme Sankey interactif
    st.markdown("---")
    st.markdown("### 🌊 Visualisation des Flux de Réassurance")
    
    col_sankey1, col_sankey2 = st.columns([1, 2])
    
    with col_sankey1:
        st.markdown("""
        <div style='background: #f8f9fa; padding: 1.5rem; border-radius: 15px;'>
            <h4>🎯 Comment lire ce diagramme</h4>
            <p>Visualisez comment le risque circule entre les différents acteurs :</p>
            <ul>
            <li>🟢 <b>Portefeuille</b> : Tous les risques assurés</li>
            <li>🔵 <b>Assureur</b> : Garde une partie du risque</li>
            <li>🟠 <b>Réassureur</b> : Prend le risque excédentaire</li>
            <li>🔴 <b>Sinistres</b> : Les pertes qui surviennent</li>
            </ul>
            <p><i>💡 Plus le flux est épais, plus le montant est important !</i></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Contrôles pour le diagramme Sankey
        sinistres_directs = st.slider("Niveau des sinistres (%)", 40, 80, 60)
        part_reassureur = st.slider("Participation réassureur (%)", 20, 60, 40)

    with col_sankey2:
        # Données pour le diagramme Sankey
        portefeuille_total = 100
        labels = ["Portefeuille", "Assureur", "Réassureur", "Sinistres courants"]
        label_to_idx = {lab: i for i, lab in enumerate(labels)}
        
        sankey_fig = go.Figure(data=[go.Sankey(
            node=dict(
                label=labels,
                pad=18,
                thickness=20,
                line=dict(color="#cfe0ee", width=1),
                color=["#4CAF50", "#2196F3", "#FF9800", "#F44336"]
            ),
            link=dict(
                source=[0, 1, 1],  # Portefeuille, Assureur, Assureur
                target=[1, 3, 2],  # Assureur, Sinistres, Réassureur
                value=[portefeuille_total, sinistres_directs, part_reassureur],
                color="rgba(0,0,0,0.2)"
            )
        )])
        
        sankey_fig.update_layout(
            height=400, 
            title="Flux simplifié du risque entre assureur et réassureur",
            font=dict(size=12)
        )
        
        st.plotly_chart(sankey_fig, use_container_width=True)

    # Roadmap d'apprentissage
    st.markdown("---")
    st.markdown("### 🗺️ Roadmap d'Apprentissage")
    
    roadmap_data = {
        'Phase': ['🎯 Fondamentaux', '⚙️ Techniques', '📊 Analyse', '🚀 Expertise'],
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
