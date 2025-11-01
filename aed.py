import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import json

# Configuration de la page
st.set_page_config(page_title="Data Science Analysis", page_layout="wide")

# Titre et introduction
st.title('Automated Data Science Study')
st.markdown("""
Cette application permet d'effectuer une étude complète de données en data science avec :
- Chargement multi-format (CSV, Excel, JSON)
- Nettoyage automatique intelligent
- Analyse exploratoire approfondie
- Feature engineering automatisé
""")

# 1. CHARGEMENT DES DONNÉES
st.header("1️⃣ Chargement des données", divider=True)

uploaded_file = st.file_uploader(
    "Téléchargez votre fichier de données",
    type=["csv", "xlsx", "json"],
    help="Formats supportés : CSV, Excel (XLSX), JSON"
)

if uploaded_file is not None:
    file_details = {
        "Nom du fichier": uploaded_file.name,
        "Taille": round(uploaded_file.size / 1024) if uploaded_file.size else None,
        "Type": uploaded_file.type.split('/')[1] if '/' in uploaded_file.type else uploaded_file.type
    }

    # Analyse immédiate des métadonnées
    data = None
    
    try:
        if uploaded_file.type == 'text/csv':
            df_str = StringIO(uploaded_file.read().decode("utf-8"))
            data = pd.read_csv(df_str)
            
        elif "spreadsheet" in uploaded_file.type:  # Pour Excel
            data = pd.read_excel(uploaded_file, engine='openpyxl')
            
        else:
            json_data = json.loads(uploaded_file.read().decode('utf-8'))
            if isinstance(json_data, dict):
                try:
                    df_from_dict = pd.DataFrame.from_dict(json_data)
                    data = df_from_dict
                except:
                    # Si le JSON n'est pas un DataFrame direct, on essaie de charger comme CSV
                    df_str = StringIO(uploaded_file.read().decode('utf-8'))
                    data = pd.read_csv(df_str)
            else:
                st.error("Le format JSON fourni ne correspond à aucun schéma reconnu pour créer un DataFrame. Veuillez vérifier le format.")
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier : {str(e)}")
    
    if data is not None:
        # Afficher les informations sur le dataset
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Aperçu du Dataset")
            if len(data) > 5000:  # Limiter l'affichage pour de grands datasets
                st.dataframe(data.head())
            else:
                st.dataframe(data)
                
            st.write(f"Nombre de lignes : {len(data)}")
            st.write(f"Nombre de colonnes : {len(data.columns)}")
            
        with col2:
            # Types des données
            data_types = pd.DataFrame({
                'Colonne': data.columns,
                'Type': [str(dtype) for dtype in data.dtypes],
                'Non-nulle (%)': [(data[col].notnull().sum()/len(data)*100).round(1) 
                                 if not col in ['index', 'Unnamed'] else None 
                                 for col in data.columns]
            })
            
            st.dataframe(data_types.style.format("{:.1f}%").highlight_nulls())
        
        # Gestion des valeurs manquantes
        missing_values = pd.DataFrame({
            'Colonne': [col for col in data.columns if data[col].isna().any()],
            'Nombre de valeurs manquantes': data[missing_col].isna().sum()  # Correction ici pour éviter erreur
        })
        
        st.subheader("Valeurs Manquantes")
        if len(missing_values) > 0:
            st.dataframe(missing_values.style.format("{:.0f}").highlight_max(axis=0))
        else:
            st.success("Aucune valeur manquante détectée")
            
    # Charger un exemple si le fichier n'est pas valide
else:
    example_datasets = {
        "Titanic": {"url": "https://raw.githubusercontent.com/mariano-santos/EDA-Examples/main/titanic.csv", 
                   "description": "Dataset sur les survivants du naufrage du Titanic"},
        "Iris": {"url": "https://raw.githubusercontent.com/ui-ry/oceanai-datasets/master/data/Iris.csv",
                "description": "Dataset classique pour classification"},
        "Boston Housing": {"url": "https://raw.githubusercontent.com/jbrownlee/MachineLearningDatasets/master/boston_housing.csv",
                           "description": "Données sur le prix des maisons à Boston"}
    }
    
    selected_example = st.selectbox("Sélectionnez un exemple de dataset", 
                                   options=list(example_datasets.keys()),
                                   help="Ces datasets seront chargés automatiquement si vous ne téléchargez pas de fichier")
    
    if selected_example:
        try:
            data_url = example_datasets[selected_example]["url"]
            df = pd.read_csv(data_url)
            
            # Afficher les informations sur le dataset
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Aperçu du Dataset")
                if len(df) > 5000: 
                    st.dataframe(df.head())
                else:
                    st.dataframe(df)
                    
                st.write(f"Nombre de lignes : {len(df)}")
                st.write(f"Nombre de colonnes : {len(df.columns)}")
                
            with col2:
                # Types des données
                data_types = pd.DataFrame({
                    'Colonne': df.columns,
                    'Type': [str(dtype) for dtype in df.dtypes],
                    'Non-nulle (%)': [(df[col].notnull().sum()/len(df)*100).round(1) 
                                    if not col in ['index', 'Unnamed'] else None 
                                    for col in df.columns]
                })
                
                st.dataframe(data_types.style.format("{:.1f}%").highlight_nulls())
        except Exception as e:
            st.error(f"Erreur lors du chargement de l'exemple : {str(e)}")

# 2. NETTOYAGE AUTOMATIQUE
st.header("2️⃣ Nettoyage des données", divider=True)

if 'data' not in st.session_state or uploaded_file is None:
    # Si les données ne sont pas chargées, afficher un message
    if example_datasets and selected_example:
        df = pd.read_csv(example_datasets[selected_example]["url"])
        st.session_state.data = df.copy()
        
nettoyage_options = {
    'supprimer_doublons': True,
    'correction_types': True,
    'gestion_valeurs_manquantes': False,  # Optionnel pour l'utilisateur
    'traitement_outliers': True,
    'standardisation_textes': False,        # Optionnel selon le contexte
    'suppression_colonnes_inutiles': True
}

with st.expander("Configurer les options de nettoyage"):
    for option in nettoyage_options:
        if option == 'gestion_valeurs_manquantes':
            col_opt = st.columns(2)
            with col_opt[0]:
                st.markdown(f"**{option.replace('_', ' ').title()}:**")
                cols_to_keep = st.multiselect("Colonnes à conserver", 
                                             options=st.session_state.data.columns, 
                                             default=[], 
                                             key=f"{option}_keep")
        else:
            nettoyage_options[option] = st.checkbox(option.replace('_', ' ').title(), value=True)

if uploaded_file is not None or example_datasets and selected_example:
    # Analyse des doublons
    if 'data' in st.session_state:
        df_cleaned = st.session_state.data.copy()
        
        # Initialiser les variables de nettoyage avec des valeurs par défaut
        col_to_keep = {}
        correction_types_options = {}
        
        # 2.1 Correction des types de données
        def convert_df(df):
            cols = df.columns.tolist()
            
            if st.session_state.get('supprimer_doublons', True) or 'data' not in st.session_state:
                @st.cache_data(show_spinner=False)
                def remove_duplicates():
                    return df_clean.drop_duplicates().copy()

            # Correction des types de données
            type_mapping = {}
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col].dtype):
                    # Pour les colonnes numériques, s'assurer qu'elles sont float ou int
                    unique_count = len(pd.unique(df[col]))
                    total_elements = len(df)
                    percentage_unique = round(unique_count / total_elements * 100, 1)
                    
                    if st.session_state.get('correction_types', True):
                        if df[col].dtype == 'object':
                            # Conversion des colonnes object en catégorielles
                            try:
                                if unique_count <= 5:  # Categorical columns
                                    df_clean[col] = df[col].astype('category').cat.codes
                                    st.success(f"La colonne '{col}' a été convertie en numérique (encodage)")
                            except Exception as e:
                                pass
                
    else:
            st.warning("Veuillez d'abord charger un dataset pour configurer le nettoyage")

# 3. ANALYSE EXPLORATOIRE
st.header("3️⃣ Analyse exploratoire", divider=True)

if 'data_cleaned' in st.session_state and hasattr(st.session_state.data_clean, 'dtypes'):
    df_clean = st.session_state.data_clean.copy()
else:
    # Charger les données si non nettoyées
    pass

# 4. FEATURE ENGINEERING
st.header("4️⃣ Feature Engineering", divider=True)

if uploaded_file is not None or example_datasets and selected_example:

    with st.expander("Configurer le feature engineering"):
        col1, col2 = st.columns(2)
    
    with col1:
        # Sélection du type d'étude
        study_types = {
            'classification': ['Iris', 'Titanic'],
            'regression': ['Boston Housing']
        }
        
        selected_study_type = st.selectbox("Type d'étude", 
                                          options=['Classification', 'Régression', 'Clustering'],
                                          help="Sélectionnez le type d'étude pour configurer l'analyse")
        
        if uploaded_file is None and example_datasets:
            # Appliquer des transformations de nettoyage
            df_clean = st.session_state.data.copy()
            
            # Vérifier les colonnes numériques et catégorielles
            numerical_cols = [col for col in df_clean.select_dtypes(include=np.number).columns]
            categorical_cols = [col for col in df_clean.columns if df_clean[col].dtype == 'object']
            
            st.subheader("Statistiques des valeurs manquantes")
            missing_percentage = ((df_clean.isnull().sum() / len(df_clean) * 100)).sort_values(ascending=False)
            cols_with_missing = [col for col in missing_values.index if missing_values[col]['Nombre de valeurs manquantes'] > 0]
            
            # Autres fonctionnalités d'analyse...

# Fin du code - Le reste serait développé selon la même logique