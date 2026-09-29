"""
Application Streamlit — Prédiction de souscription bancaire

Lancement en local :  streamlit run app.py
"""

import pandas as pd
import joblib as jb
import streamlit as st

# Configuration de la page
st.set_page_config(
    page_title="Prédiction de souscription bancaire",
    page_icon="🏦",
    layout="centered",
)

DESCRIPTION = (
    "Ce modèle de machine learning (XGBoost) permet de prédire si un client "
    "va souscrire à un produit bancaire, à partir de ses informations "
    "personnelles et des données de la campagne marketing."
)

# Colonnes attendues, dans l'ordre utilisé à l'entraînement
COLONNES_FEATURES = [
    "age", "job", "marital", "education", "housing", "loan",
    "contact", "month", "day_of_week", "duration", "campaign",
    "pdays", "previous", "poutcome",
]

COLONNES_CATEGORIELLES = [
    "job", "marital", "education", "housing", "loan",
    "contact", "month", "day_of_week", "poutcome",
]


# Chargement des artefacts (mis en cache : chargés une seule fois)
@st.cache_resource
def load_artifacts():
    encoders = jb.load("encoders.joblib")            # dict des LabelEncoder par colonne
    target_encoder = jb.load("target_encoder.joblib")  # LabelEncoder de la cible y
    scaler = jb.load("scaler.joblib")                 # normaliseur
    xgb = jb.load("xgb_model.joblib")                 # modèle
    return encoders, target_encoder, scaler, xgb


encoders, target_encoder, scaler, xgb = load_artifacts()

# Reconstruire les libellés de la cible dans l'ordre des classes encodées (ex. [0, 1] -> ['no', 'yes'])
class_names = list(target_encoder.classes_)


# Fonction de prédiction simple
def pred_func(valeurs: dict):
    entree = pd.DataFrame([valeurs], columns=COLONNES_FEATURES)

    for col in COLONNES_CATEGORIELLES:
        entree[col] = encoders[col].transform(entree[col])

    x_new = scaler.transform(entree)
    y_pred = xgb.predict(x_new)

    return class_names[y_pred[0]]


# Fonction de prédiction multiple à partir d'un CSV
def pred_func_csv(file):
    df = pd.read_csv(file)
    predictions = []
    for _, row in df[COLONNES_FEATURES].iterrows():
        y_pred = pred_func(row.to_dict())
        predictions.append(y_pred)
    df["prediction"] = predictions
    return df


# Interface
st.title("🏦 Prédiction de souscription bancaire")

onglet1, onglet2 = st.tabs(["Prédiction simple", "Prédiction multiple"])

# ----------------------------- Onglet 1 -------------------------------
with onglet1:
    st.subheader("Prédire la souscription d'un client à partir d'une entrée")
    st.write(DESCRIPTION)

    with st.form("formulaire_simple"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Âge", min_value=18, max_value=100, value=35, step=1)
            job = st.selectbox("Job", options=sorted(encoders["job"].classes_))
            marital = st.selectbox("Situation familiale", options=sorted(encoders["marital"].classes_))
            education = st.selectbox("Éducation", options=sorted(encoders["education"].classes_))
            housing = st.selectbox("Prêt immobilier (housing)", options=sorted(encoders["housing"].classes_))
            loan = st.selectbox("Prêt personnel (loan)", options=sorted(encoders["loan"].classes_))
            contact = st.selectbox("Moyen de contact", options=sorted(encoders["contact"].classes_))

        with col2:
            month = st.selectbox("Mois du dernier contact", options=sorted(encoders["month"].classes_))
            day_of_week = st.selectbox("Jour de la semaine", options=sorted(encoders["day_of_week"].classes_))
            duration = st.number_input("Durée du dernier appel (secondes)", min_value=0, value=180, step=10)
            campaign = st.number_input("Nombre de contacts (campagne)", min_value=1, value=1, step=1)
            pdays = st.number_input("Jours depuis le dernier contact (999 = jamais)", min_value=0, value=999, step=1)
            previous = st.number_input("Nombre de contacts précédents", min_value=0, value=0, step=1)
            poutcome = st.selectbox("Résultat de la campagne précédente", options=sorted(encoders["poutcome"].classes_))

        soumettre = st.form_submit_button("Prédire", type="primary")

    if soumettre:
        try:
            valeurs = {
                "age": age, "job": job, "marital": marital, "education": education,
                "housing": housing, "loan": loan, "contact": contact, "month": month,
                "day_of_week": day_of_week, "duration": duration, "campaign": campaign,
                "pdays": pdays, "previous": previous, "poutcome": poutcome,
            }
            resultat = pred_func(valeurs)
            if resultat.lower() == "yes":
                st.success(f"**Prédiction :** le client va probablement souscrire ({resultat}).")
            else:
                st.warning(f"**Prédiction :** le client ne va probablement pas souscrire ({resultat}).")
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")

# ----------------------------- Onglet 2 -------------------------------
with onglet2:
    st.subheader("Prédire la souscription de plusieurs clients à partir d'un fichier CSV")
    st.write(DESCRIPTION)
    st.caption(
        "Le fichier CSV doit contenir, avec ces noms de colonnes exacts : "
        + ", ".join(COLONNES_FEATURES)
    )

    fichier = st.file_uploader("Importer un fichier CSV", type=["csv"])

    if fichier is not None:
        try:
            with st.spinner("Prédictions en cours…"):
                df_resultat = pred_func_csv(fichier)

            st.success(f"{len(df_resultat)} prédiction(s) effectuée(s).")
            st.dataframe(df_resultat, use_container_width=True)

            st.download_button(
                label="⬇️ Télécharger le fichier CSV",
                data=df_resultat.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
                type="primary",
            )
        except Exception as e:
            st.error(f"Erreur lors du traitement du fichier : {e}")
