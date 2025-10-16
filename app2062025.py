import requests
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import pickle
import gzip

# Fichier cloud streamlit

# Mapping des endpoints API
URL_MAPPER = {
    "local": "http://localhost:5000/predict/",
    "hosted": "https://credit-prediction-demo.onrender.com/predict/"
}
api_version = "local"  # Toggle ici si tu veux changer d'environnement

# Chargement de la base client (100 exemples)
client_database = pd.read_csv("prod_client_database_example_100.csv")
client_database = client_database.drop(columns=["TARGET"])
client_ids = client_database["SK_ID_CURR"].to_list()

def request_prediction(data, endpoint_url):

    # Nettoyage des données avant l'appel API
    for key, value in data.items():
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            st.warning(f"Attention : valeur non valide détectée pour {key}, remplacée par 0.0")
            data[key] = 0.0
    
    # ✅ CORRECTION ICI : encapsuler data dans une liste
    data_json = {'inputs': [data]}

    response = requests.post(
        url=endpoint_url,
        headers={"Content-Type": "application/json"},
        json=data_json
    )

    if response.status_code != 200:
        raise Exception(
            f"Request failed with status {response.status_code}, {response.text}"
        )

    return response.json()


# Main Streamlit app
def main():
    st.title("Pret A Depenser - Démo accès API de prédiction")
    st.markdown("Cette application permet de se connecter à une API de prédiction de remboursement de prêt.")

    with st.container(border=True):
        st.subheader("Recherche de client dans base de données interne")
        id_client_choix = st.selectbox("Choisir un client par ID bancaire", tuple(client_ids))
        st.write("ID du client sélectionné :", id_client_choix)

    with gzip.open("shap_values_compressed.pkl.gz", "rb") as f:
        shap_values = pickle.load(f)

    # Charger les valeurs SHAP depuis le fichier pickle
    # with open("shap_values.pkl", "rb") as f:
        # shap_values = pickle.load(f)

    # Liste des features à afficher et leur ordre fixe
    features_to_show = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3','PAYMENT_RATE', 'NAME_FAMILY_STATUS_Married', 'CODE_GENDER', 'AMT_ANNUITY', 'APPROVED_CNT_PAYMENT_MEAN']
    fixed_order = features_to_show  # tu peux les séparer si besoin

    # Sélection du client (id)
    # line_id = int(id_client_choix)
    

    # Exemple : ton DataFrame client_database contient la colonne "SK_ID_CURR"
    client_row2 = client_database[client_database["SK_ID_CURR"] == id_client_choix]

    if client_row2.empty:
        st.error("Client introuvable dans la base.")
    else:
        # Trouver l'index (position) de ce client dans X_test_enc (ou client_database si même ordre)
        try:
            line_id = client_database.index.get_loc(client_row2.index[0])
        except KeyError:
            st.error("⚠️ Impossible de trouver la position du client dans les données d'entraînement.")
            st.stop()

        with st.container():
            st.subheader("Données pour les 8 plus importantes features du client")

            # Vérifier les colonnes disponibles
            available_features = [f for f in features_to_show if f in client_row2.columns]
            missing_features = [f for f in features_to_show if f not in client_row2.columns]

            # Extraire les données existantes
            if not available_features:
                # Si aucune des colonnes n'existe
                st.warning("Aucune donnée disponible pour les colonnes demandées.")
                client_features = pd.DataFrame(
                    {"Valeur": ["Donnée non renseignée"] * len(features_to_show)},
                    index=features_to_show
                )
            else:
                # Créer un DataFrame avec les colonnes présentes
                client_features = client_row2.reindex(columns=features_to_show)
                # Remplacer les NaN ou colonnes manquantes par "Donnée non renseignée"
                client_features = client_features.fillna("Donnée non renseignée")
                client_features = client_features.T.rename(columns={client_row2.index[0]: "Valeur"})

            # Si certaines colonnes sont absentes du DataFrame
            if missing_features:
                for f in missing_features:
                    client_features.loc[f] = "Donnée non renseignée"

            # Affichage dans Streamlit
            st.write("### Valeurs du client sélectionné")
            st.dataframe(client_features)

        with st.container():
            st.subheader("Données SHAP du client")

            # Vérifier la validité de l'index
            if line_id < 0 or line_id >= shap_values.values.shape[0]:
                st.error(f"❌ L'identifiant {line_id} est invalide (max = {shap_values.values.shape[0]-1}).")
            else:
                st.write(f"Client sélectionné : {id_client_choix} (ligne {line_id})")

                # Extraire les valeurs SHAP du client
                shap_client = shap_values[line_id]

                # Créer un mapping feature -> index
                feature_to_index = {name: i for i, name in enumerate(shap_client.feature_names)}

                # Sélectionner uniquement les features présentes ET dans le bon ordre
                ordered_features = [f for f in fixed_order if f in feature_to_index]

                # Construire un nouvel objet SHAP dans l’ordre souhaité
                indices = [feature_to_index[f] for f in ordered_features]

                shap_filtered = shap.Explanation(
                    values=shap_client.values[indices],
                    base_values=shap_client.base_values,
                    data=shap_client.data[indices],
                    feature_names=ordered_features
                )

                # Créer et afficher la figure
                fig, ax = plt.subplots(figsize=(10, 5))
                shap.plots.waterfall(shap_filtered, show=False)
                plt.tight_layout()

                # Afficher dans Streamlit
                st.pyplot(fig)


    with st.container(border=True):
        st.subheader("Requête vers endpoint de prédiction API")  
        st.write(f"Endpoint utilisé : {URL_MAPPER[api_version]}")

        if st.button("Faire requête API pour ce client"):
            # Extraction des données pour ce client
            client_row = client_database[client_database["SK_ID_CURR"] == id_client_choix]

            if client_row.empty:
                st.error("Client introuvable dans la base.")
                return
            
            # top_features = ['EXT_SOURCE_3', 'EXT_SOURCE_2', 'PAYMENT_RATE', 'EXT_SOURCE_1', 'NAME_FAMILY_STATUS_Married', 'CODE_GENDER', 'AMT_ANNUITY', 'APPROVED_CNT_PAYMENT_MEAN']
            data = {
                'EXT_SOURCE_3': client_row['EXT_SOURCE_3'].values[0],
                'EXT_SOURCE_2': client_row['EXT_SOURCE_2'].values[0],
                'EXT_SOURCE_1': client_row['EXT_SOURCE_1'].values[0],
                'AMT_ANNUITY': client_row['AMT_ANNUITY'].values[0],
                
                # PAYMENT_RATE = AMT_ANNUITY / AMT_CREDIT
                'PAYMENT_RATE': client_row['AMT_ANNUITY'].values[0] / client_row['AMT_CREDIT'].values[0],
                
                # Encodage binaire de NAME_FAMILY_STATUS
                'NAME_FAMILY_STATUS_Married': 1 if client_row['NAME_FAMILY_STATUS'].values[0] == 'Married' else 0,
                
                # Encodage binaire de CODE_GENDER
                'CODE_GENDER_F': 1 if client_row['CODE_GENDER'].values[0] == 'F' else 0,
                
                # APPROVED_CNT_PAYMENT_MEAN n'existe pas => valeur fictive (à remplacer si enrichissement disponible)
                'APPROVED_CNT_PAYMENT_MEAN': 0.0  # ou np.nan
            }

            with st.spinner("Envoi de la requête à l'API..."):
                try:
                    pred = request_prediction(data, URL_MAPPER[api_version])
                    st.success("Requête API réussie :white_check_mark:")

                    with st.container(border=True):
                        st.subheader("Prédiction du modèle")
                        st.write("**Classe prédite :**", pred['classe'])
                        st.write("**Probabilité d'échec :**", f"{pred['proba_echec']*100:.2f}%")
                except Exception as e:
                    st.error(f"Erreur lors de la requête API : {e}")

if __name__ == "__main__":
    main()
