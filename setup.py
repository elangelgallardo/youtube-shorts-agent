"""
One-time setup script. Run this before starting the agent for the first time.

    python3 setup.py

It will:
  1. Verify Python + FFmpeg are available
  2. Load .env and check required env vars
  3. Locate or create YouTube OAuth client secrets
  4. Complete YouTube OAuth flow
  5. Verify Gemini API (text + image generation)
  6. Verify Google Cloud TTS
  7. Create all required directories
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Load .env before anything else so env vars are available for all checks
from dotenv import load_dotenv
load_dotenv()


def check(condition: bool, ok_msg: str, fail_msg: str) -> bool:
    if condition:
        print(f"  ✓ {ok_msg}")
        return True
    else:
        print(f"  ✗ {fail_msg}")
        return False


def main() -> None:
    print("\n=== YouTube Shorts Agent — Setup Verification ===\n")
    all_ok = True

    # ── 1. Python version ────────────────────────────────────────────────
    print("[1] Python version")
    ok = check(
        sys.version_info >= (3, 11),
        f"Python {sys.version.split()[0]}",
        "Python 3.11+ required",
    )
    all_ok = all_ok and ok

    # ── 2. FFmpeg ────────────────────────────────────────────────────────
    print("\n[2] FFmpeg (required by MoviePy for encoding)")
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        ok = check(result.returncode == 0, "FFmpeg found", "FFmpeg not found — run: apt install ffmpeg")
    except FileNotFoundError:
        ok = check(False, "", "FFmpeg not found — run: apt install ffmpeg")
    all_ok = all_ok and ok

    # ── 3. Environment variables ─────────────────────────────────────────
    print("\n[3] Environment variables (.env)")
    ok_env = check(os.path.exists(".env"), ".env file found", ".env not found — copy .env.example to .env and fill in your values")
    all_ok = all_ok and ok_env

    # Only GOOGLE_API_KEY and YOUTUBE_CHANNEL_ID are strictly required now
    # (GCP_PROJECT_ID was needed for Vertex AI which we no longer use)
    required_vars = {
        "GOOGLE_API_KEY": "Get from https://aistudio.google.com/app/apikey",
        "YOUTUBE_CHANNEL_ID": "Your channel ID starting with UC...",
    }
    for var, hint in required_vars.items():
        ok = check(bool(os.environ.get(var)), f"{var} is set", f"{var} is NOT set — {hint}")
        all_ok = all_ok and ok

    # ── 4. YouTube OAuth client secrets ──────────────────────────────────
    print("\n[4] YouTube OAuth credentials")
    import config
    secrets_path = Path(config.YOUTUBE_CLIENT_SECRETS)  # credentials/client_secrets.json

    if not secrets_path.exists():
        # Google downloads the file as client_secret_XXXX.apps.googleusercontent.com.json
        # Try to find and rename it automatically
        candidates = list(Path("credentials").glob("client_secret*.json"))
        if candidates:
            shutil.copy(candidates[0], secrets_path)
            print(f"  ✓ Found and copied {candidates[0].name} → client_secrets.json")
        else:
            print("  ✗ credentials/client_secrets.json not found")
            print()
            print("     How to get it:")
            print("       1. Go to https://console.cloud.google.com")
            print("       2. APIs & Services → Enabled APIs → enable both:")
            print("            - YouTube Data API v3")
            print("            - YouTube Analytics API")
            print("       3. APIs & Services → Credentials")
            print("       4. Click '+ CREATE CREDENTIALS' → 'OAuth client ID'")
            print("       5. Application type: *** Desktop app ***  ← IMPORTANT")
            print("          (NOT 'Web application' — that causes redirect_uri_mismatch)")
            print("       6. Click CREATE, then the download ↓ button")
            print("          File will be named like:")
            print("          client_secret_XXXX.apps.googleusercontent.com.json")
            print("       7. Place it in the credentials/ folder")
            print("          (it will be auto-renamed to client_secrets.json)")
            print()
            # Offer to create from raw client_id + client_secret
            answer = input("     Do you have a client_id and client_secret to enter manually? [y/N] ").strip().lower()
            if answer == "y":
                client_id = input("     Client ID: ").strip()
                client_secret = input("     Client Secret: ").strip()
                if client_id and client_secret:
                    _write_client_secrets(secrets_path, client_id, client_secret)
                    print(f"  ✓ Created {secrets_path}")
                else:
                    print("  ✗ Skipped — no credentials entered")
                    all_ok = False
            else:
                all_ok = False

    if secrets_path.exists():
        # Warn if the file is a Web Application client (causes redirect_uri_mismatch)
        try:
            secrets_data = json.load(open(secrets_path))
            client_type = next(iter(secrets_data), None)
            if client_type == "web":
                check(False, "",
                      "client_secrets.json is a 'Web application' client — this WILL cause "
                      "'redirect_uri_mismatch'.\n"
                      "     Re-create it as a 'Desktop app' in Google Cloud Console.")
                all_ok = False
            else:
                check(True, f"client_secrets.json found (type: {client_type})", "")
        except Exception:
            check(True, f"client_secrets.json found at {secrets_path}", "")
    else:
        all_ok = False

    # ── 5. YouTube OAuth flow ────────────────────────────────────────────
    print("\n[5] YouTube OAuth token")
    if secrets_path.exists():
        try:
            from auth import get_youtube_credentials
            creds = get_youtube_credentials()
            ok = check(creds.valid, "YouTube OAuth token valid", "YouTube OAuth failed")
            all_ok = all_ok and ok
        except Exception as exc:
            ok = check(False, "", f"YouTube OAuth failed: {exc}")
            all_ok = False
    else:
        print("  — Skipped (no client_secrets.json)")

    # ── 6. Gemini API — text ─────────────────────────────────────────────
    print("\n[6] Gemini API (text)")
    try:
        from google import genai as google_genai
        from google.genai import types as genai_types
        client = google_genai.Client(api_key=os.environ.get("GOOGLE_API_KEY", ""))
        resp = client.models.generate_content(
            model=config.GEMINI_FLASH_MODEL,
            contents="Reply with just the word: ready",
        )
        ok = check(
            "ready" in resp.text.lower(),
            f"Gemini API responds ({config.GEMINI_FLASH_MODEL})",
            f"Gemini API error: {resp.text[:100]}",
        )
    except Exception as exc:
        ok = check(False, "", f"Gemini API failed: {exc}")
    all_ok = all_ok and ok

    # ── 7. Gemini image generation ───────────────────────────────────────
    print("\n[7] Gemini image generation ({})".format(config.IMAGEN_MODEL))
    try:
        from google import genai as genai_new
        from google.genai import types

        client = genai_new.Client(api_key=config.GOOGLE_API_KEY)
        response = client.models.generate_content(
            model=config.IMAGEN_MODEL,
            contents="A simple blue circle on a white background, minimalist",
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
            ),
        )
        has_image = any(
            part.inline_data is not None
            for part in response.candidates[0].content.parts
        )
        ok = check(has_image, f"Image generation works ({config.IMAGEN_MODEL})", "No image returned")
    except Exception as exc:
        ok = check(False, "", f"Image generation failed: {exc}")
    all_ok = all_ok and ok

    # ── 8. Google Cloud TTS ──────────────────────────────────────────────
    print("\n[8] Google Cloud Text-to-Speech")
    try:
        from google.api_core.client_options import ClientOptions
        from google.cloud import texttospeech
        tts_client = texttospeech.TextToSpeechClient(
            client_options=ClientOptions(api_key=os.environ.get("GOOGLE_TTS_API_KEY", os.environ.get("GOOGLE_API_KEY", "")))
        )
        voices = tts_client.list_voices(language_code="en-US")
        ok = check(len(voices.voices) > 0, "Cloud TTS accessible via API key", "Cloud TTS returned no voices")
    except Exception as exc:
        ok = check(False, "", f"Cloud TTS failed: {exc}")
        print("     Make sure GOOGLE_API_KEY is set and Cloud Text-to-Speech API is enabled")
        print("     Enable it at: https://console.cloud.google.com/apis/library/texttospeech.googleapis.com")
    all_ok = all_ok and ok

    # ── 9. Directories ───────────────────────────────────────────────────
    print("\n[9] Creating directories")
    for d in [config.WORKSPACE_DIR, config.LOGS_DIR, config.CREDENTIALS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        check(True, str(d), "")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    if all_ok:
        print("✓ All checks passed! Run the agent with:")
        print("    source .venv/bin/activate")
        print("    python3 main.py --now --dry-run   # test one video (private)")
        print("    python3 main.py                   # start 24h scheduler")
    else:
        print("✗ Some checks failed. Fix the issues above, then re-run:")
        print("    python3 setup.py")
    print()


def _write_client_secrets(path: Path, client_id: str, client_secret: str) -> None:
    """Create a client_secrets.json from raw credentials."""
    data = {
        "installed": {
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uris": ["http://localhost", "urn:ietf:wg:oauth:2.0:oob"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
