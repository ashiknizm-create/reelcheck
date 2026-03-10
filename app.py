from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
import os
import uuid
import requests
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

GROQ_KEY = os.environ.get("GROQ_API_KEY")
IG_USERNAME = os.environ.get("INSTAGRAM_USERNAME")
IG_PASSWORD = os.environ.get("INSTAGRAM_PASSWORD")

# Global Instagram client — login once, reuse
ig_client = None


def get_ig_client():
    """Login to Instagram with bot account — reuse session if already logged in"""
    global ig_client
    if ig_client:
        return ig_client, None
    try:
        from instagrapi import Client
        cl = Client()
        cl.login(IG_USERNAME, IG_PASSWORD)
        ig_client = cl
        return ig_client, None
    except Exception as e:
        return None, str(e)


def extract_shortcode(instagram_url):
    """Extract reel shortcode from Instagram URL"""
    patterns = [
        r"instagram\.com/reel/([A-Za-z0-9_-]+)",
        r"instagram\.com/p/([A-Za-z0-9_-]+)",
        r"instagram\.com/reels/([A-Za-z0-9_-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, instagram_url)
        if match:
            return match.group(1)
    return None


def download_reel_audio(instagram_url, output_path):
    """Download reel audio using Instagram bot account"""

    # ── Method 1: Instagrapi (bot account login) ──
    try:
        client, error = get_ig_client()
        if client:
            shortcode = extract_shortcode(instagram_url)
            if shortcode:
                # Get media info
                media_pk = client.media_pk_from_code(shortcode)
                media_info = client.media_info(media_pk)

                # Get video URL
                video_url = None
                if hasattr(media_info, 'video_url') and media_info.video_url:
                    video_url = str(media_info.video_url)
                elif hasattr(media_info, 'resources') and media_info.resources:
                    for resource in media_info.resources:
                        if hasattr(resource, 'video_url') and resource.video_url:
                            video_url = str(resource.video_url)
                            break

                if video_url:
                    # Download audio using yt-dlp from direct video URL
                    import yt_dlp
                    tmp_base = output_path.replace(".mp3", "")
                    ydl_opts = {
                        "format": "bestaudio/best",
                        "outtmpl": f"{tmp_base}.%(ext)s",
                        "quiet": True,
                        "no_warnings": True,
                        "postprocessors": [{
                            "key": "FFmpegExtractAudio",
                            "preferredcodec": "mp3",
                            "preferredquality": "64",
                        }],
                    }
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([video_url])

                    if os.path.exists(output_path):
                        return True, None

                    # Alternative: direct download with requests
                    response = requests.get(
                        video_url,
                        headers={"User-Agent": "Mozilla/5.0"},
                        timeout=60,
                        stream=True
                    )
                    if response.status_code == 200:
                        tmp_video = output_path.replace(".mp3", ".mp4")
                        with open(tmp_video, "wb") as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)

                        # Convert to mp3 using ffmpeg
                        os.system(f'ffmpeg -i "{tmp_video}" -q:a 0 -map a "{output_path}" -y -loglevel quiet')
                        try:
                            os.remove(tmp_video)
                        except Exception:
                            pass

                        if os.path.exists(output_path):
                            return True, None

    except Exception as e:
        # Reset client on error so it tries to login again next time
        global ig_client
        ig_client = None
        pass

    # ── Method 2: yt-dlp direct (fallback) ──
    try:
        import yt_dlp
        tmp_base = output_path.replace(".mp3", "")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": f"{tmp_base}.%(ext)s",
            "quiet": True,
            "no_warnings": True,
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "64",
            }],
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([instagram_url])

        if os.path.exists(output_path):
            return True, None

    except Exception as e:
        pass

    return False, "Could not access reel. Instagram may have flagged the bot account. Please add context manually."


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
    if not GROQ_KEY:
        return jsonify({"error": "Groq API key not configured on server."}), 500
    if not IG_USERNAME or not IG_PASSWORD:
        return jsonify({"error": "Instagram bot credentials not configured on server."}), 500

    transcript = ""
    download_error = None
    tmp_id = str(uuid.uuid4())
    audio_file = f"/tmp/{tmp_id}.mp3"

    # ── STEP 1: Download reel audio via bot account ──
    success, error = download_reel_audio(url, audio_file)

    if success and os.path.exists(audio_file):
        # ── STEP 2: Transcribe with Whisper ──
        try:
            model = whisper.load_model("base")
            result = model.transcribe(audio_file)
            transcript = result.get("text", "").strip()
        except Exception as e:
            transcript = ""
            download_error = f"Transcription failed: {str(e)}"
        finally:
            # ── STEP 3: Delete audio immediately ──
            try:
                os.remove(audio_file)
            except Exception:
                pass
    else:
        download_error = error

    # ── STEP 4: Build Claude prompt ──
    if transcript:
        content_block = f"TRANSCRIPT OF REEL AUDIO:\n{transcript}"
    elif context:
        content_block = f"USER-PROVIDED CONTEXT:\n{context}"
    else:
        content_block = "No transcript or context provided. Analyse the URL topic only."

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

    # ── STEP 5: Send to Groq ──
    try:
        client = Groq(api_key=GROQ_KEY)
        message = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        result_text = message.choices[0].message.content.strip()
    except Exception as e:
        error_msg = str(e)
        if "invalid_api_key" in error_msg.lower() or "authentication" in error_msg.lower():
            return jsonify({"error": "Invalid Groq API key. Check console.groq.com"}), 500
        return jsonify({"error": f"Groq API error: {error_msg}"}), 500

    return jsonify({
        "transcript": transcript,
        "download_error": download_error,
        "result": result_text
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)