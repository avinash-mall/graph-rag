from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import blingfire
from langdetect import detect
from flair.models import SequenceTagger
from flair.data import Sentence
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(
    title="Flair API",
    version="1.0.0",
    description="API for performing Flair NER and text chunking using Blingfire."
)

# Cache for loaded models
model_cache = {}

DEFAULT_MODEL_MAP = {
    'en': 'ner-large',
    'de': 'de-ner-large',
    'es': 'es-ner-large',
    'nl': 'nl-ner-large',
    'fr': 'fr-ner',
    'da': 'da-ner',
    'ar': 'ar-ner',
    'uk': 'ner-ukrainian'
}

class TextRequest(BaseModel):
    text: str

class ChunkRequest(BaseModel):
    text: str
    max_chunk_size: int = 512

@app.post("/extract_flair_entities")
def extract_flair_entities_endpoint(req: TextRequest):
    text = req.text
    try:
        lang = detect(text).lower()
    except Exception as e:
        lang = 'en'
    if '-' in lang:
        lang = lang.split('-')[0]
    model_name = DEFAULT_MODEL_MAP.get(lang, 'ner-large')
    if model_name in model_cache:
        tagger = model_cache[model_name]
    else:
        try:
            tagger = SequenceTagger.load(model_name)
            model_cache[model_name] = tagger
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model {model_name}: {e}")
    sentence_strs = [s.strip() for s in blingfire.text_to_sentences(text).splitlines() if s.strip()]
    sentences = [Sentence(s) for s in sentence_strs]
    try:
        tagger.predict(sentences, mini_batch_size=32)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NER prediction error: {e}")
    entities = {}
    for sentence in sentences:
        for label in sentence.get_labels():
            key = (label.data_point.text, label.value)
            entities[key] = {"name": label.data_point.text, "label": label.value}
    return {"entities": list(entities.values())}

@app.post("/chunk_text")
def chunk_text_endpoint(req: ChunkRequest):
    text = req.text
    max_chunk_size = req.max_chunk_size
    sentences = [s.strip() for s in blingfire.text_to_sentences(text).splitlines() if s.strip()]
    chunks = []
    current_chunk = ""
    for s in sentences:
        if current_chunk and (len(current_chunk) + len(s) + 1) > max_chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = s
        else:
            current_chunk = s if not current_chunk else current_chunk + " " + s
    if current_chunk:
        chunks.append(current_chunk.strip())
    return {"chunks": chunks}

if __name__ == "__main__":
    api_host = os.getenv("FLAIR_API_HOST", "0.0.0.0")
    api_port = int(os.getenv("FLAIR_API_PORT", "8001"))
    import uvicorn
    uvicorn.run(app, host=api_host, port=api_port)
