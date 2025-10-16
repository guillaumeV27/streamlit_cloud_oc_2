import requests
import streamlit as st
import pandas as pd
import numpy as np
import os

# Mapping des endpoints API
URL_MAPPER = {
    "local": "http://localhost:5000/predict/",
    "hosted": "https://flask-api-predict.onrender.com/predict/"
}

api_version = "hosted"

# Chargement de la base client (50 exemples)
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
