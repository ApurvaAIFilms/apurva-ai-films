import os
import tempfile
import base64
from pathlib import Path

import cv2
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

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
    # oEmbed provides public metadata only; it does not grant access to video frames.
    try:
        r = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": url, "format": "json"},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        return {
            "title": data.get("title"),
            "author": data.get("author_name"),
            "thumbnail": data.get("thumbnail_url"),
            "note": "YouTube metadata only. Upload an authorised video file to analyse its visual frames.",
        }
    except Exception:
        raise HTTPException(400, "Could not read YouTube video metadata. Check the URL.")


@app.post("/api/frames")
async def extract_frames(video: UploadFile = File(...), count: int = 12):
    if count not in (8, 12, 16, 24):
        raise HTTPException(400, "Choose 8, 12, 16, or 24 frames.")
    suffix = Path(video.filename or "video.mp4").suffix or ".mp4"
    temp_path = None
    cap = None
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
            position = int(i * max(total - 1, 0) / max(count - 1, 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, position)
            ok, frame = cap.read()
            if ok:
                # Resize to reduce mobile upload and vision API costs.
                h, w = frame.shape[:2]
                max_side = 768
                scale = min(1.0, max_side / max(h, w))
                if scale < 1.0:
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
                if ok:
                    frames.append("data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii"))
        if not frames:
            raise HTTPException(400, "No frames could be extracted from this video.")
        return {"frames": frames, "count": len(frames), "note": "Frames are held in this browser session; analyse them before refreshing the page."}
    finally:
        if cap is not None:
            cap.release()
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/api/analyze")
async def analyze(payload: dict):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(503, "AI analysis is not configured. Add OPENAI_API_KEY in Render Environment settings.")

    mode = payload.get("mode", "original")
    instructions = (payload.get("instructions") or "").strip()
    original_prompt = (payload.get("story_prompt") or "").strip()
    frames = payload.get("frames") or []
    reference_title = (payload.get("reference_title") or "").strip()

    if mode == "original" and not original_prompt:
        raise HTTPException(400, "Please describe the original movie you want to create.")
    if mode == "reference" and not instructions and not frames:
        raise HTTPException(400, "Add creative instructions or extract frames from an authorised reference video first.")
    if len(frames) > 12:
        frames = frames[:12]

    if mode == "original":
        task = (
            "Create a completely ORIGINAL feature-film concept. Do not use or assume any reference film. "
            "Develop the user's idea into: title, genre, logline, premise, detailed synopsis, main characters "
            "(including appearance/personality/motivation), three-act structure, 12-20 scene outline, "
            "key emotional and suspense turns, and a short sample dialogue scene. Keep continuity and make "
            "the story suitable for later screenplay and video-production stages. Be clear that this is a "
            "development package, not a rendered movie.\n\nUSER'S STORY IDEA:\n" + original_prompt
        )
    elif mode == "reference":
        task = (
            "Develop a DISTINCT new film inspired only by broad visual cues or themes in the supplied "
            "authorised reference frames and the user's instructions. Do not recreate the film, copy scenes, "
            "or reproduce dialogue. If frames are supplied, describe only what is visibly supported; do not "
            "claim to know the full plot from a few samples. Return: visible reference observations, what to "
            "retain/change, a new title and logline, synopsis, characters, three-act structure, 12-20 scene "
            "outline, and sample dialogue. Clearly identify uncertainty where the frames do not establish facts.\n\n"
            f"REFERENCE TITLE (metadata only): {reference_title or 'Not provided'}\n"
            f"USER'S CREATIVE INSTRUCTIONS:\n{instructions or 'Use the reference frames only as broad inspiration and invent a distinct story.'}"
        )
    else:
        raise HTTPException(400, "Choose Reference-Based or Original Story mode.")

    content = [{"type": "input_text", "text": task}]
    for frame in frames:
        if isinstance(frame, str) and frame.startswith("data:image/"):
            content.append({"type": "input_image", "image_url": frame, "detail": "low"})

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            input=[{"role": "user", "content": content}],
        )
        return {"analysis": response.output_text, "mode": mode, "frames_used": len(content) - 1}
    except Exception as e:
        # Keep provider details and secrets out of the response shown to the browser.
        print(f"OpenAI analysis error: {type(e).__name__}: {e}")
        raise HTTPException(502, "AI analysis failed. Check API billing, key permissions, and the configured model in Render.")
