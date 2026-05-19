# api/main.py
# SenSante API - Assistant pre-diagnostic medical
# Lab 3 & 5 - Integration de Modeles IA - ESP/UCAD

import os
import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from groq import Groq

# --- 1. CONFIGURATION & ENVIRONNEMENT ---
load_dotenv()
#os.environ["GROQ_API_KEY"] = "" : cela permet d'ecraser de force la cle

# --- 2. INITIALISATION DE L'APPLICATION FASTAPI ---
app = FastAPI(
    title="SenSante API",
    description="Assistant pre-diagnostic medical pour le Senegal",
    version="0.2.0"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Autorise toutes les origines en développement
    allow_credentials=True,
    allow_methods=["*"],      # Autorise toutes les méthodes (POST, GET, etc.)
    allow_headers=["*"],      # Autorise tous les headers
)

# --- 3. INITIALISATION DU CLIENT GROQ ---
groq_client = None
groq_api_key = os.getenv("GROQ_API_KEY")

if groq_api_key:
    groq_client = Groq(api_key=groq_api_key)
    print("Client Groq initialise avec succes.")
else:
    print("ATTENTION : GROQ_API_KEY non trouvee. /explain sera desactive.")

# --- 4. CHARGEMENT DES MODELES ML ---
model = None
le_sexe = None
le_region = None
feature_cols = None

try:
    print("Chargement des modeles ML...")
    model = joblib.load("models/model.pkl")
    le_sexe = joblib.load("models/encoder_sexe.pkl")
    le_region = joblib.load("models/encoder_region.pkl")
    feature_cols = joblib.load("models/feature_cols.pkl")
    print(f"Modele charge avec succes. Classes : {list(model.classes_)}")
except Exception as e:
    print(f"Erreur critique lors du chargement des modeles : {e}")

# --- 5. SCHÉMAS PYDANTIC ---

# Pour /predict
class PatientInput(BaseModel):
    age: int = Field(..., ge=0, le=120)
    sexe: str = Field(...)
    temperature: float = Field(..., ge=35.0, le=42.0)
    tension_sys: int = Field(..., ge=60, le=250)
    toux: bool = Field(...)
    fatigue: bool = Field(...)
    maux_tete: bool = Field(...)
    region: str = Field(...)

class DiagnosticOutput(BaseModel):
    diagnostic: str
    probabilite: float
    confiance: str
    message: str

# Pour /explain
class ExplainInput(BaseModel):
    diagnostic: str = Field(..., description="Diagnostic predit par le modele")
    probabilite: float = Field(..., description="Probabilite du diagnostic")
    age: int = Field(...)
    sexe: str = Field(...)
    temperature: float = Field(...)
    region: str = Field(...)

class ExplainOutput(BaseModel):
    explication: str = Field(..., description="Explication en francais")
    modele_llm: str = Field(default="llama-3.1-8b-instant", description="Modele LLM utilise")

# --- 6. PROMPTS ---
SYSTEM_PROMPT = """Tu es un assistant medical senegalais.
Tu recois un diagnostic et des donnees patient.
Explique le resultat en francais simple,
comme un medecin parlerait a son patient.
Sois rassurant mais recommande toujours
une consultation medicale.
Maximum 3 phrases.
Ne fais JAMAIS de diagnostic toi-meme.
Tu expliques uniquement le diagnostic fourni."""

# --- 7. ROUTES API ---

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "SenSante API is running"}

@app.get("/model-info")
def model_info():
    if not model:
        return {"status": "error", "message": "Modele non charge"}
    return {
        "type": type(model).__name__,
        "nombre_arbres": getattr(model, 'n_estimators', "N/A"),
        "classes": list(model.classes_),
        "nombre_features": model.n_features_in_
    }

@app.post("/predict", response_model=DiagnosticOutput)
def predict(patient: PatientInput):
    if not model:
        return DiagnosticOutput(
            diagnostic="erreur", probabilite=0.0, confiance="aucune",
            message="Le modele de prediction est indisponible sur le serveur."
        )
    try:
        # Encodage du sexe
        try:
            sexe_enc = le_sexe.transform([patient.sexe])[0]
        except ValueError:
            return DiagnosticOutput(
                diagnostic="erreur", probabilite=0.0, confiance="aucune",
                message=f"Sexe invalide : {patient.sexe}")

        # Encodage de la région
        try:
            region_enc = le_region.transform([patient.region])[0]
        except ValueError:
            return DiagnosticOutput(
                diagnostic="erreur", probabilite=0.0, confiance="aucune",
                message=f"Region inconnue : {patient.region}")

        # Préparation des features
        features = np.array([[
            patient.age, sexe_enc, patient.temperature,
            patient.tension_sys, int(patient.toux),
            int(patient.fatigue), int(patient.maux_tete),
            region_enc
        ]])

        # Prediction
        diagnostic = model.predict(features)[0]
        proba_max = float(model.predict_proba(features)[0].max())
        
        # Calcul de la confiance
        confiance = ("haute" if proba_max >= 0.7 
                     else "moyenne" if proba_max >= 0.4 
                     else "faible")

        messages = {
            "palu": "Suspicion de paludisme. Consultez rapidement.",
            "grippe": "Suspicion de grippe. Repos et hydratation.",
            "typh": "Suspicion de typhoide. Consultation necessaire.",
            "sain": "Pas de pathologie detectee."
        }

        return DiagnosticOutput(
            diagnostic=str(diagnostic),
            probabilite=round(proba_max, 2),
            confiance=confiance,
            message=messages.get(diagnostic, "Consultez un medecin.")
        )
    
    except Exception as e:
        return DiagnosticOutput(
            diagnostic="erreur", probabilite=0.0, confiance="aucune",
            message=f"Erreur interne : {str(e)}"
        )

@app.post("/explain", response_model=ExplainOutput)
def explain(data: ExplainInput):
    """Expliquer un diagnostic en francais avec un LLM."""
    if not groq_client:
        return ExplainOutput(
            explication="Service d'explication indisponible. Clé API non configurée.",
            modele_llm="aucun"
        )

    # Construire le user prompt
    user_prompt = (
        f"Patient : {data.sexe}, {data.age} ans, "
        f"region {data.region}\n"
        f"Temperature : {data.temperature} C\n"
        f"Diagnostic du modele : {data.diagnostic} "
        f"(probabilite {data.probabilite:.0%})\n"
        f"Explique ce resultat au patient."
    )

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=200,
            temperature=0.3
        )
        explication = response.choices[0].message.content

    except Exception as e:
        explication = f"Erreur lors de l'appel au LLM : {str(e)}"

    return ExplainOutput(explication=explication)