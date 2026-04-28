import os
import base64
import json
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

app = Flask(__name__)
CORS(app)

client = Groq(api_key=st.secrets["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"])
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
TEXT_MODEL   = "llama3-8b-8192"

history = []


def encode_image(file) -> tuple[str, str]:
    data = file.read()
    b64 = base64.standard_b64encode(data).decode("utf-8")
    mime = file.mimetype or "image/jpeg"
    return b64, mime


def build_analysis_prompt(tone: str) -> str:
    tone_map = {
        "emotional": "deeply emotional and heartfelt",
        "funny":     "humorous and witty",
        "dramatic":  "suspenseful and cinematic",
        "formal":    "professional and journalistic",
        "kids":      "fun and simple, suitable for children",
        "detective": "narrated by a hard-boiled detective",
    }
    story_tone = tone_map.get(tone, "emotional")
    return f"""You are an advanced vision AI. Analyze this image thoroughly.
Respond ONLY with a valid JSON object — no markdown, no code fences, no extra text.

Use exactly this structure:
{{
  "caption": "One concise factual sentence describing the image.",
  "summary": "A 3-5 line descriptive paragraph about the scene.",
  "objects": ["object1", "object2", "object3", "object4", "object5"],
  "emotion": "Overall emotional tone in 1-3 words.",
  "scene_type": "One of: classroom, office, street, park, event, indoor, outdoor, crowd, nature, other",
  "story": "A 4-6 sentence creative story inspired by this scene. Tone: {story_tone}.",
  "confidence": 85,
  "image_quality": "One of: clear, blurry, dark, bright, low-light, partially-visible",
  "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
  "mood_score": 65
}}

Rules:
- confidence: integer 0-100
- mood_score: integer 0-100 (0=very negative, 50=neutral, 100=very positive)
- Return ONLY raw JSON. Nothing else."""


def safe_json(text: str) -> dict:
    text = text.strip()
    if "```" in text:
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    tone = request.form.get("tone", "emotional")
    b64, mime = encode_image(file)
    try:
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": build_analysis_prompt(tone)}
            ]}],
            max_tokens=1500,
        )
        raw = response.choices[0].message.content
        result = safe_json(raw)
        history.insert(0, {
            "thumbnail": f"data:{mime};base64,{b64}",
            "caption": result.get("caption", ""),
            "emotion": result.get("emotion", ""),
            "mood_score": result.get("mood_score", 50),
            "scene_type": result.get("scene_type", ""),
            "tone": tone,
        })
        if len(history) > 6:
            history.pop()
        return jsonify(result)
    except json.JSONDecodeError as e:
        return jsonify({"error": f"JSON parse error: {str(e)}", "raw": raw}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/followup", methods=["POST"])
def followup():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    file = request.files["image"]
    question = request.form.get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400
    b64, mime = encode_image(file)
    try:
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": f"Look at this image and answer accurately: {question}"}
            ]}],
            max_tokens=500,
        )
        return jsonify({"answer": response.choices[0].message.content.strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/compare", methods=["POST"])
def compare():
    if "image1" not in request.files or "image2" not in request.files:
        return jsonify({"error": "Both image1 and image2 are required"}), 400
    try:
        b64_1, mime1 = encode_image(request.files["image1"])
        b64_2, mime2 = encode_image(request.files["image2"])
        prompt = """Compare these two images. Respond ONLY with valid JSON, no markdown:
{
  "brightness_diff": "which image is brighter",
  "crowd_level": {"image1": "empty/sparse/moderate/crowded", "image2": "empty/sparse/moderate/crowded"},
  "mood": {"image1": "positive/neutral/negative", "image2": "positive/neutral/negative"},
  "scene_type": {"image1": "scene", "image2": "scene"},
  "key_differences": ["diff1", "diff2", "diff3"],
  "similarity_score": 40,
  "summary": "2-3 sentence comparison."
}"""
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime1};base64,{b64_1}"}},
                {"type": "image_url", "image_url": {"url": f"data:{mime2};base64,{b64_2}"}},
                {"type": "text", "text": prompt}
            ]}],
            max_tokens=800,
        )
        raw = response.choices[0].message.content
        result = safe_json(raw)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/history", methods=["GET"])
def get_history():
    return jsonify(history)


if __name__ == "__main__":
    print("=" * 50)
    print("  TechMesh'26 Vision AI — Groq Edition")
    print("  Open http://localhost:5000 in browser")
    print("=" * 50)
    app.run(debug=True, port=5000)
