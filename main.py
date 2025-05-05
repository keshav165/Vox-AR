from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import shutil
import os
import uuid
import config
from text_to_3d_generator import generate_3d_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
RESULTS_DIR = config.OUTPUT_DIR
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

@app.post("/api/transcribe/")
async def transcribe_audio_api(file: UploadFile = File(...)):
    # Save uploaded file
    file_id = str(uuid.uuid4())
    temp_path = os.path.join(UPLOAD_DIR, f"{file_id}.wav")
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    from voice_to_3d import transcribe_audio
    text = transcribe_audio(temp_path)
    os.remove(temp_path)
    return {"text": text}

class PromptRequest(BaseModel):
    prompt: str

@app.post("/api/generate_3d/")
async def generate_3d_api(request: PromptRequest):
    try:
        prompt = request.prompt
        gif_path, ply_path, obj_path = generate_3d_model(prompt, RESULTS_DIR)
        return {
            "gif": gif_path,
            "ply": ply_path,
            "obj": obj_path
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/download/")
def download_file(path: str):
    # Sanitize path to use forward slashes
    safe_path = path.replace("\\", "/")
    if os.path.exists(safe_path):
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Methods": "*"
        }
        return FileResponse(safe_path, headers=headers)
    return JSONResponse(status_code=404, content={"error": "File not found"})

@app.get("/api/models/")
def list_models():
    model_dir = RESULTS_DIR
    files = os.listdir(model_dir)
    models = {}
    for f in files:
        if not (f.endswith('.gif') or f.endswith('.obj') or f.endswith('.ply')):
            continue
        base = f.rsplit('.', 1)[0]
        if base not in models:
            models[base] = {}
        ext = f.rsplit('.', 1)[-1]
        models[base][ext] = os.path.join(model_dir, f)
    # Format as list of dicts with keys gif, obj, ply
    result = []
    for base, d in models.items():
        result.append({
            'base': base,
            'gif': d.get('gif'),
            'obj': d.get('obj'),
            'ply': d.get('ply'),
        })
    return result
