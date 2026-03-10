from flask import Flask, request, jsonify
from flask_cors import CORS
import yt_dlp
import whisper
import anthropic
import os
import uuid
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY")


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/factcheck", methods=["POST"])
def factcheck():
    data = request.json
    url = data.get("url", "").strip()
    context = data.get("context", "").strip()

    if not url:
        return jsonify({"error": "No URL provided."}), 400
    if "instagram.com" not in url:
        return jsonify({"error": "Please provide a valid Instagram URL."}), 400
    if not ANTHROPIC_KEY:
        return jsonify({"error": "Server API key not configured. Contact the admin."}), 500

    # ── STEP A: Download audio from reel ──
    tmp_id = str(uuid.uuid4())
    tmp_path = f"/tmp/{tmp_id}"
    audio_file = f"{tmp_path}.mp3"

    transcript = ""
    download_error = None

    try:
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": f"{tmp_path}.%(ext)s",
            "quiet": True,
            "no_warnings": True,
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "64",
            }],
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

    except Exception as e:
        download_error = str(e)

    # ── STEP B: Transcribe with Whisper ──
    if not download_error and os.path.exists(audio_file):
        try:
            model = whisper.load_model("base")
            result = model.transcribe(audio_file)
            transcript = result.get("text", "").strip()
        except Exception as e:
            transcript = ""
        finally:
            # ── STEP C: Delete audio immediately ──
            try:
                os.remove(audio_file)
            except Exception:
                pass
    else:
        transcript = ""

    # ── STEP D: Build prompt ──
    if transcript:
        content_block = f"TRANSCRIPT OF REEL AUDIO:\n{transcript}"
    elif context:
        content_block = f"USER-PROVIDED CONTEXT (reel could not be downloaded):\n{context}"
    else:
        content_block = "No transcript or context available. Analyse based on the URL topic only."

    prompt = f"""You are a professional fact-checker and media analyst.

A user submitted this Instagram reel for fact-checking.

REEL URL: {url}
{content_block}
{f"ADDITIONAL USER NOTES: {context}" if transcript and context else ""}

YOUR TASK:
1. Identify the topic and all specific claims made
2. For EACH claim give a clear verdict: TRUE / FALSE / MISLEADING / UNVERIFIABLE
3. Provide brief evidence and reasoning for each verdict
4. List reliable sources you know of
5. Give an OVERALL VERDICT with a one-line bottom line

Use this exact format:

━━━━━━━━━━━━━━━━━━━━
📌 TOPIC
[What this reel is about in 1-2 sentences]

━━━━━━━━━━━━━━━━━━━━
🔎 CLAIMS ANALYSIS

Claim 1: [exact claim]
Verdict: TRUE / FALSE / MISLEADING / UNVERIFIABLE
Evidence: [clear explanation with reasoning]

Claim 2: [exact claim]
Verdict: TRUE / FALSE / MISLEADING / UNVERIFIABLE
Evidence: [clear explanation with reasoning]

[continue for all claims]

━━━━━━━━━━━━━━━━━━━━
📚 SOURCES
[Reliable sources relevant to these claims]

━━━━━━━━━━━━━━━━━━━━
⚖️ OVERALL VERDICT: TRUE / FALSE / MISLEADING / UNVERIFIABLE
Bottom line: [One clear sentence summarising the finding]

Be impartial, concise, and evidence-based."""

    # ── STEP E: Send to Claude ──
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        result_text = message.content[0].text.strip()
    except anthropic.AuthenticationError:
        return jsonify({"error": "Invalid Anthropic API key on server."}), 500
    except Exception as e:
        return jsonify({"error": f"Claude API error: {str(e)}"}), 500

    return jsonify({
        "transcript": transcript,
        "download_error": download_error,
        "result": result_text
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)