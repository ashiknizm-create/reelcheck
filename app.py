from flask import Flask, request, jsonify
from flask_cors import CORS
import yt_dlp
import whisper
import anthropic
import os
import uuid
import requests
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY")


@app.route("/")
def index():
    return app.send_static_file("index.html")


def get_stream_url(instagram_url):
    """Use Cobalt.tools API to get direct stream URL from Instagram reel"""
    try:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        payload = {
            "url": instagram_url,
            "aFormat": "mp3",
            "isAudioOnly": True
        }
        response = requests.post(
            "https://api.cobalt.tools/api/json",
            headers=headers,
            json=payload,
            timeout=30
        )
        data = response.json()

        if data.get("status") == "stream":
            return data.get("url"), None
        elif data.get("status") == "redirect":
            return data.get("url"), None
        elif data.get("status") == "picker":
            items = data.get("picker", [])
            if items:
                return items[0].get("url"), None
        elif data.get("status") == "error":
            return None, data.get("text", "Cobalt error")
        else:
            return None, f"Unexpected response: {data.get('status')}"

    except requests.exceptions.Timeout:
        return None, "Cobalt.tools request timed out"
    except Exception as e:
        return None, str(e)


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
        return jsonify({"error": "Server API key not configured."}), 500

    # ── STEP 1: Get stream URL from Cobalt.tools ──
    stream_url, cobalt_error = get_stream_url(url)

    transcript = ""
    download_error = None
    tmp_id = str(uuid.uuid4())
    audio_file = f"/tmp/{tmp_id}.mp3"

    if stream_url:
        # ── STEP 2: Stream and extract audio ──
        try:
            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": f"/tmp/{tmp_id}.%(ext)s",
                "quiet": True,
                "no_warnings": True,
                "postprocessors": [{
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "64",
                }],
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([stream_url])

        except Exception as e:
            download_error = str(e)

        # ── STEP 3: Transcribe with Whisper ──
        if not download_error and os.path.exists(audio_file):
            try:
                model = whisper.load_model("base")
                result = model.transcribe(audio_file)
                transcript = result.get("text", "").strip()
            except Exception as e:
                transcript = ""
            finally:
                # ── STEP 4: Delete audio immediately ──
                try:
                    os.remove(audio_file)
                except Exception:
                    pass
    else:
        download_error = cobalt_error or "Could not get stream URL from Cobalt.tools"

    # ── STEP 5: Build Claude prompt ──
    if transcript:
        content_block = f"TRANSCRIPT OF REEL AUDIO:\n{transcript}"
    elif context:
        content_block = f"USER-PROVIDED CONTEXT (reel could not be streamed):\n{context}"
    else:
        content_block = "No transcript available. Analyse based on the Instagram URL topic only."

    prompt = f"""You are a professional fact-checker and media analyst.

A user submitted this Instagram reel for fact-checking.

REEL URL: {url}
{content_block}
{f"ADDITIONAL USER NOTES: {context}" if transcript and context else ""}

YOUR TASK:
1. Identify the topic and all specific claims made
2. For EACH claim give a clear verdict: TRUE / FALSE / MISLEADING / UNVERIFIABLE
3. Provide brief evidence and reasoning for each verdict
4. List reliable sources
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

━━━━━━━━━━━━━━━━━━━━
📚 SOURCES
[Reliable sources relevant to these claims]

━━━━━━━━━━━━━━━━━━━━
⚖️ OVERALL VERDICT: TRUE / FALSE / MISLEADING / UNVERIFIABLE
Bottom line: [One clear sentence]

Be impartial, concise, and evidence-based."""

    # ── STEP 6: Send to Claude ──
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
        "stream_url_found": bool(stream_url),
        "result": result_text
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)