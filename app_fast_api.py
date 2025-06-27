import logging
import time
import sys
import os
import io
import zipfile
from pathlib import Path
import shutil
from typing import List
import subprocess
import tempfile
import re

import requests
from fastapi import FastAPI, Request, HTTPException, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# --- Configuration et Initialisation ---

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(
    title="LaTeX Generator API with Gemini",
    description="Une API pour générer, améliorer, compiler et exporter des documents LaTeX en utilisant l'API Google Gemini.",
    version="2.1.0"
)

UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

# --- Exceptions personnalisées ---

class LatexGeneratorError(Exception): pass
class APIError(LatexGeneratorError): pass
class CompilationError(Exception): pass

# --- Classe "Moteur" pour l'API Gemini ---

class LatexGenerator:
    API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    
    # --- PROMPTS SPÉCIALISÉS ---

    # Prompt pour un document complet (CORRIGÉ)
    DOCUMENT_PROMPT_BASE = """
Tu es un assistant expert en LaTeX. Ton unique objectif est de générer du code LaTeX propre, complet et compilable.
Règles strictes :
- Réponds UNIQUEMENT avec le code LaTeX brut.
- N'ajoute AUCUNE explication, commentaire ou texte en dehors du code LaTeX lui-même.
- Le code doit être un document complet, commençant par \\documentclass et se terminant par \\end{{document}}.
- Adapte la langue du document (package babel) et la classe du document selon la demande.
- Inclus systématiquement les packages essentiels : amsmath pour les maths, graphicx pour les images, [utf8]{{inputenc}}, et [T1]{{fontenc}}.

Génère un document LaTeX complet de type "{doc_type}" en langue "{lang}" sur papier "{doc_format}".
Le document doit correspondre à la description suivante : "{description}"
"""

    # Prompt pour un bloc de code (ex: un tableau, une section)
    BLOCK_PROMPT = """
Tu es un assistant expert en LaTeX. Génère UNIQUEMENT un bloc de code LaTeX correspondant à la description suivante.
Règles strictes:
- Ne génère QUE le code LaTeX demandé.
- N'inclus PAS de préambule, pas de \\documentclass, pas de \\begin{{document}} ou \\end{{document}}.
- Ne fournis aucune explication.
Description : "{description}"
"""

    # Prompt pour une formule mathématique
    FORMULA_PROMPT = """
Tu es un assistant expert en LaTeX. Transforme la description suivante en une formule mathématique LaTeX.
Règles strictes:
- Réponds UNIQUEMENT avec le code de la formule elle-même.
- N'ajoute PAS de délimiteurs comme $$...$$, \\[...\\] ou $...$. Le contexte sera ajouté par l'utilisateur.
- Ne fournis aucune explication.
Description de la formule : "{description}"
"""

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("La clé API Gemini est requise.")
        self.api_key = api_key
        self.session = requests.Session()

    def _clean_response(self, text: str) -> str:
        match = re.search(r'```(?:latex)?\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()

    def _query_api(self, prompt: str) -> str:
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 4096,
            }
        }
        params = {'key': self.api_key}
        
        try:
            response = self.session.post(self.API_URL, params=params, json=payload, timeout=90)
            response.raise_for_status()
            data = response.json()

            if not data.get('candidates'):
                raise APIError(f"Réponse de l'API invalide ou bloquée. Réponse reçue : {data}")

            generated_text = data['candidates'][0]['content']['parts'][0]['text']
            return self._clean_response(generated_text)

        except requests.exceptions.RequestException as e:
            logging.error(f"Erreur de communication avec l'API Gemini: {e}")
            raise APIError(f"Erreur de connexion à l'API Gemini: {e}") from e

    def generate_document(self, description: str, doc_type: str, lang: str, doc_format: str) -> str:
        if not description or not description.strip():
            return "Veuillez entrer une description non vide."
        
        full_prompt = self.DOCUMENT_PROMPT_BASE.format(
            description=description, doc_type=doc_type, lang=lang, doc_format=doc_format
        )
        return self._query_api(full_prompt)

    def generate_block(self, description: str) -> str:
        if not description or not description.strip():
            return "Veuillez entrer une description non vide."
        full_prompt = self.BLOCK_PROMPT.format(description=description)
        return self._query_api(full_prompt)

    def generate_formula(self, description: str) -> str:
        if not description or not description.strip():
            return "Veuillez entrer une description non vide."
        full_prompt = self.FORMULA_PROMPT.format(description=description)
        return self._query_api(full_prompt)

# --- Instanciation du générateur ---
try:
    
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        # Remplacez par votre clé si elle n'est pas dans .env pour le test
        # gemini_api_key = "VOTRE_CLE_API_GEMINI_ICI"
        raise RuntimeError("La variable d'environnement GEMINI_API_KEY n'est pas définie.")
    
    latex_generator = LatexGenerator(api_key=gemini_api_key)
    logging.info("Générateur LaTeX avec Gemini initialisé.")
except Exception as e:
    logging.critical(f"ERREUR FATALE à l'initialisation: {e}")
    sys.exit(1)

# --- Modèles de données Pydantic (modifié) ---

class GenerationRequest(BaseModel):
    description: str
    generation_type: str  # 'document', 'block', ou 'formula'
    doc_type: str | None = None
    lang: str | None = None
    doc_format: str | None = None

class ExportRequest(BaseModel):
    latex_code: str
    filenames: List[str]

class PdfRequest(BaseModel):
    latex_code: str

# --- Endpoints de l'API (endpoint /generate modifié) ---

@app.get("/", response_class=HTMLResponse)
async def get_main_page():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="index.html non trouvé.")

@app.post("/generate")
async def generate_latex_code(req: GenerationRequest):
    try:
        generated_code = ""
        if req.generation_type == "document":
            if not all([req.doc_type, req.lang, req.doc_format]):
                 raise HTTPException(status_code=400, detail="Pour un document, doc_type, lang, et doc_format sont requis.")
            generated_code = latex_generator.generate_document(
                description=req.description, doc_type=req.doc_type, lang=req.lang, doc_format=req.doc_format
            )
        elif req.generation_type == "block":
            generated_code = latex_generator.generate_block(description=req.description)
        elif req.generation_type == "formula":
            generated_code = latex_generator.generate_formula(description=req.description)
        else:
            raise HTTPException(status_code=400, detail="Type de génération invalide.")

        return {"latex_code": generated_code}
        
    except APIError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logging.error(f"Erreur inattendue /generate: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Erreur interne du serveur.")

# ... (le reste du fichier app.py reste inchangé : /upload-figure, /export-zip, /download-pdf)
@app.post("/upload-figure")
async def upload_figure(file: UploadFile = File(...)):
    # Sanitize filename pour éviter les traversées de répertoire
    filename = Path(file.filename).name
    file_path = UPLOADS_DIR / filename
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"message": f"Fichier '{filename}' uploadé.", "filename": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload du fichier échoué: {e}")

@app.post("/export-zip")
async def export_zip(req: ExportRequest):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        zip_file.writestr("main.tex", req.latex_code.encode('utf-8'))
        for filename in req.filenames:
            # S'assurer que les noms de fichiers sont sûrs
            safe_filename = Path(filename).name
            file_path = UPLOADS_DIR / safe_filename
            if file_path.is_file():
                zip_file.write(file_path, arcname=safe_filename)

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=latex_project.zip"}
    )

@app.post("/download-pdf")
async def download_pdf(req: PdfRequest):
    # L'utilisation d'un répertoire temporaire est une bonne pratique pour l'isolation et le nettoyage.
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        tex_file_path = temp_dir / "document.tex"
        pdf_file_path = temp_dir / "document.pdf"
        log_file_path = temp_dir / "document.log"

        # Écrire le code LaTeX reçu dans un fichier .tex
        tex_file_path.write_text(req.latex_code, encoding='utf-8')

        # Commande pour compiler le document LaTeX avec pdflatex
        command = [
            "pdflatex",
            "-interaction=nonstopmode",  # Empêche pdflatex de s'arrêter en cas d'erreur
            f"-output-directory={temp_dir}",
            str(tex_file_path)
        ]

        try:
            logging.info(f"Début de la compilation LaTeX dans {temp_dir}...")
            # Première passe de compilation
            subprocess.run(command, check=False, capture_output=True, timeout=30)
            # Seconde passe (importante pour les références, tables des matières, etc.)
            process = subprocess.run(command, check=False, capture_output=True, text=True, encoding='utf-8', errors='ignore', timeout=30)

            # Vérifier si le fichier PDF a bien été créé
            if not pdf_file_path.exists():
                logging.error("Échec de la compilation LaTeX : le fichier PDF n'a pas été généré.")
                log_content = log_file_path.read_text() if log_file_path.exists() else "Fichier de log non trouvé."
                # Lève une exception interne avec le log de compilation
                raise CompilationError(log_content)

            logging.info("Compilation réussie. Envoi du fichier PDF.")
            # FileResponse streame le fichier depuis le disque.
            # FastAPI attend que le streaming soit terminé avant que le contexte 'with' ne se ferme
            # et ne supprime le répertoire temporaire. C'est donc sûr et correct.
            return FileResponse(
                path=pdf_file_path,
                media_type='application/pdf',
                filename='document.pdf'
            )

        except FileNotFoundError:
            logging.critical("La commande 'pdflatex' n'a pas été trouvée sur le serveur.")
            raise HTTPException(
                status_code=501, # 501 Not Implemented
                detail="Erreur serveur : l'outil 'pdflatex' n'est pas installé ou n'est pas dans le PATH du système."
            )
        except subprocess.TimeoutExpired:
            logging.error("La compilation LaTeX a dépassé le temps imparti (timeout).")
            raise HTTPException(
                status_code=400,
                detail={"message": "La compilation a pris trop de temps. Vérifiez les boucles infinies ou les erreurs complexes dans le code.", "log": "Timeout expired."}
            )
        except CompilationError as e:
            # Si la compilation a échoué, on retourne une erreur 400 (Bad Request)
            # avec le log pour que l'utilisateur puisse déboguer son code LaTeX.
            raise HTTPException(
                status_code=400,
                detail={"message": "Erreur de compilation LaTeX. Le code fourni n'est pas valide. Consultez le log pour plus de détails.", "log": str(e)}
            )
        except Exception as e:
            logging.error(f"Erreur inattendue lors de la compilation PDF : {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Erreur interne inattendue lors de la génération du PDF.")