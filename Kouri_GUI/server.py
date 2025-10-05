import os, re, json, subprocess, shlex
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import requests
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Kouri_backend import KouriBackend

MODEL_NAME = os.environ.get("KOURI_MODEL", "gemma3:4b")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
USE_CLI = os.environ.get("KOURI_USE_CLI", "0") == "1"
OLLAMA_BIN = os.environ.get("OLLAMA_BIN", "ollama")
TIMEOUT_SEC = 180
API_TOKEN = os.environ.get("KOURI_API_TOKEN")

ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)

app = Flask(__name__)
CORS(app)

def run_kouri_api(prompt: str) -> str:
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": True}
    url = f"{OLLAMA_HOST}/api/generate"
    try:
        r = requests.post(url, json=payload, stream=True, timeout=TIMEOUT_SEC)
    except requests.RequestException as e:
        return f"Error: cannot reach Ollama API: {e}"
    if r.status_code == 404:
        return f"Error: model '{MODEL_NAME}' not found (404). Try: ollama pull {MODEL_NAME}"
    if r.status_code != 200:
        return f"Error: HTTP {r.status_code}: {r.text[:400]}"
    chunks = []
    for line in r.iter_lines():
        if not line:
            continue
        try:
            data = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError:
            continue
        if "response" in data:
            chunks.append(data["response"])
        if data.get("done"):
            break
    out = "".join(chunks).strip()
    return strip_ansi(out) or "(no output)"

def run_kouri_cli(prompt: str) -> str:
    cmd = f"{shlex.quote(OLLAMA_BIN)} run {shlex.quote(MODEL_NAME)}"
    try:
        proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = proc.communicate(prompt, timeout=TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        proc.kill()
        return "Error: model inference timed out."
    except FileNotFoundError:
        return f"Error: ollama binary '{OLLAMA_BIN}' not found."
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"
    if proc.returncode != 0:
        err = (stderr or "").strip()
        if "pull model manifest" in err.lower():
            return f"Error: model '{MODEL_NAME}' missing. Run: ollama pull {MODEL_NAME}"
        return f"Error: {err or 'unknown failure'}"
    return strip_ansi(stdout.strip()) or "(no output)"
    
backend = KouriBackend(model_name=MODEL_NAME)

def run_kouri(prompt: str) -> str:
    return backend.route_task(prompt)

@app.route("/")
def root():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    index_path = os.path.join(project_root, "index.html")
    if os.path.exists(index_path):
        return send_file(index_path)
    return jsonify({"message": "Kouri backend running", "hint": "index.html not found at root."})

@app.route("/chat", methods=["POST"])
def chat():
    if API_TOKEN and request.headers.get("X-API-Key") != API_TOKEN:
        return jsonify({"response": "Unauthorized"}), 401
    data = request.get_json(silent=True) or {}
    msg = (data.get("message") or "").strip()
    if not msg:
        return jsonify({"response": "Please provide a message."}), 400
    resp = run_kouri(msg)
    return jsonify({"response": resp})

@app.route("/debug/last_prompt", methods=["GET"])
def debug_last_prompt():
    if API_TOKEN and request.headers.get("X-API-Key") != API_TOKEN:
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify({"last_prompt": backend.get_last_prompt()})

@app.route("/debug/reset_greeting", methods=["POST"])
def reset_greeting():
    if API_TOKEN and request.headers.get("X-API-Key") != API_TOKEN:
        return jsonify({"error": "Unauthorized"}), 401
    mem_path = backend.memory_file
    try:
        with open(mem_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        data = {"history": []}
    data.pop("greeted", None)
    data.pop("greeting_sanitized", None)
    with open(mem_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    return jsonify({"status": "ok", "message": "Greeting flags cleared."})

@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok", "model": MODEL_NAME, "mode": "cli" if USE_CLI else "api"}

remote_sessions = {}

def build_remote_context(session_id):
    hist = remote_sessions.get(session_id, [])
    lines = []
    for role, txt in hist[-8:]:
        lines.append(f"{'User' if role=='user' else 'Kouri'}: {txt}")
    return "\n".join(lines)

@app.route("/remote/chat", methods=["POST"])
def remote_chat():
    if API_TOKEN and request.headers.get("X-API-Key") != API_TOKEN:
        return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json(silent=True) or {}
    msg = (data.get("message") or "").strip()
    if not msg:
        return jsonify({"error": "Empty message"}), 400
    session_id = data.get("session_id") or "default"
    ctx = build_remote_context(session_id)
    final_input = f"{ctx}\nUser: {msg}" if ctx else msg
    resp = run_kouri(final_input)
    remote_sessions.setdefault(session_id, []).append(("user", msg))
    remote_sessions[session_id].append(("assistant", resp))
    remote_sessions[session_id] = remote_sessions[session_id][-20:]
    return jsonify({"response": resp, "session_id": session_id})

@app.route("/remote/sessions", methods=["GET"])
def list_remote_sessions():
    return jsonify({"sessions": [{"id": sid, "turns": len(hist)} for sid, hist in remote_sessions.items()]})

@app.route("/remote/session/reset", methods=["POST"])
def reset_remote_session():
    if API_TOKEN and request.headers.get("X-API-Key") != API_TOKEN:
        return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json(silent=True) or {}
    sid = data.get("session_id")
    if not sid:
        return jsonify({"error": "session_id required"}), 400
    remote_sessions.pop(sid, None)
    return jsonify({"status": "cleared", "session_id": sid})

if __name__ == "__main__":
    port = int(os.environ.get("FLASK_PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)