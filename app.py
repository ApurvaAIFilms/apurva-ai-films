import os, tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import cv2
import requests

app = FastAPI(title="Apurva AI Films")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.get("/api/health")
def health():
    return {"status": "ok", "project": "Apurva AI Films"}

@app.get("/api/youtube")
def youtube_info(url: str):
    # YouTube oEmbed provides public video metadata only, not video frames.
    try:
        r = requests.get("https://www.youtube.com/oembed",
                         params={"url": url, "format": "json"}, timeout=15)
        r.raise_for_status()
        data = r.json()
        return {"title": data.get("title"), "author": data.get("author_name"),
                "thumbnail": data.get("thumbnail_url"),
                "note": "This reads public metadata only. It does not download or extract YouTube video frames."}
    except Exception:
        raise HTTPException(400, "Could not read YouTube video metadata. Check the URL.")

@app.post("/api/frames")
async def extract_frames(video: UploadFile = File(...), count: int = 12):
    if count not in (8, 12, 16, 24):
        raise HTTPException(400, "Choose 8, 12, 16, or 24 frames.")
    suffix = os.path.splitext(video.filename or "")[1] or ".mp4"
    temp_path = None
    try:
        content = await video.read()
        if len(content) > 250 * 1024 * 1024:
            raise HTTPException(413, "Video must be under 250 MB.")
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(content)
            temp_path = f.name
        cap = cv2.VideoCapture(temp_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise HTTPException(400, "Could not read this video.")
        frames = []
        for i in range(count):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * max(total - 1, 0) / max(count - 1, 1)))
            ok, frame = cap.read()
            if ok:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if ok:
                    import base64
                    frames.append("data:image/jpeg;base64," + base64.b64encode(encoded).decode())
        cap.release()
        return {"frames": frames, "count": len(frames)}
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/api/analyze")
async def analyze(payload: dict):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(503, "AI analysis is not configured yet. Add an OpenAI API key in the hosting service's environment settings.")
    # Text-only starter analysis. API use is billed separately from ChatGPT.
    prompt = payload.get("transcript", "")
    if not prompt.strip():
        raise HTTPException(400, "Paste a transcript or story description first.")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-5.5"),
            input="Analyse this reference material for an original film concept. Do not reproduce protected scenes or dialogue. Identify broad themes, genre, character dynamics, and suggest a distinct new story.\n\n" + prompt
        )
        return {"analysis": response.output_text}
    except Exception as e:
        raise HTTPException(502, "AI analysis failed. Check the API key, model setting, and API billing.")
