
import os
import json
from functools import wraps
from flask import (
    Flask, render_template, request, redirect, url_for,
    session, flash, jsonify, send_from_directory
)
from flask_socketio import SocketIO, join_room
from werkzeug.security import generate_password_hash, check_password_hash

from dataset_generator import (
    make_pid,
    generate_text_dataset,
    generate_image_dataset,
    get_history,
    create_zip,
    HISTORY_DIR
)

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "supersecret")
socketio = SocketIO(app, async_mode="threading")
port = int(os.environ.get("PORT", 8080))
USERS_FILE = "users.json"

def load_users():
    if os.path.exists(USERS_FILE):
        return json.load(open(USERS_FILE))
    return {}

def save_users(u):
    json.dump(u, open(USERS_FILE, "w"), indent=2)

def login_required(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        if "username" not in session:
            if request.is_json or request.headers.get("X-Requested-With")== "XMLHttpRequest":
                return jsonify({"error": "authentication required"}), 401
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapped

# ---------- Auth routes ----------
@app.route("/register", methods=["GET","POST"])
def register():
    if request.method=="POST":
        u,p = request.form["username"], request.form["password"]
        users = load_users()
        if u in users:
            flash("Exists","error")
        else:
            users[u] = generate_password_hash(p)
            save_users(users)
            flash("Registered!","success")
            return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method=="POST":
        u,p = request.form["username"], request.form["password"]
        users = load_users()
        if u in users and check_password_hash(users[u],p):
            session["username"]=u
            os.makedirs(os.path.join(HISTORY_DIR,u), exist_ok=True)
            return redirect(url_for("dashboard"))
        flash("Invalid","error")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ---------- Dashboard ----------
@app.route("/")
@login_required
def dashboard():
    return render_template("index.html")

# ---------- Generate endpoint ----------
@app.route("/generate", methods=["POST"])
@login_required
def generate():
    data = request.get_json()
    cats    = data.get("categories", [])
    uploads = data.get("images", [])
    reset   = data.get("reset", False)

    user = session["username"]
    pid  = make_pid()
    base = os.path.join(HISTORY_DIR, user, pid)
    
    # start background task
    socketio.start_background_task(
        _bg_generate, user, pid, cats, uploads, reset, base
    )
    return jsonify({"status":"started","pid":pid})

def _bg_generate(user, pid, cats, uploads, reset, base):
    room = pid
    def progress(cat, f, t):
        pct = int(100*f/t) if t else 100
        socketio.emit("progress", {
            "pid": pid, "category": cat,
            "fetched": f, "total": t, "percent": pct
        }, room=room)

    # 1. Crawl text categories
    text_res = generate_text_dataset(cats, reset, base, pid, progress_callback=progress)

    # 2. Handle uploads
    up_count = None
    if uploads:
        total_up = sum(c["qty"] for c in cats if c["name"]=="uploaded") or len(uploads)
        up_count = generate_image_dataset(uploads, total_up, base, progress_callback=progress)

    # 3. Record history
    entry = {"pid":pid, "categories":cats, "uploaded":up_count, "results":text_res}
    get_history(user, entry)

    # 4. Notify completion
    socketio.emit("completed", {"pid":pid, "entry":entry}, room=room)

# ---------- History & Preview ----------
# Returns the HTML page
@app.route("/history_page")
@login_required
def history_page():
    return render_template("history.html")

# Returns the JSON history for frontend
@app.route("/history")
@login_required
def history_json():
    return jsonify(get_history(session["username"]))

@app.route("/image_list/<pid>/<category>")
@login_required
def image_list(pid, category):
    user = session["username"]
    d = os.path.join(HISTORY_DIR, user, pid, "images", category)
    if os.path.isdir(d):
        return jsonify(sorted(os.listdir(d)))
    return jsonify([])

# ---------- Serve images ----------
@app.route("/data/<u>/<pid>/<cat>/<fn>")
@login_required
def serve(u, pid, cat, fn):
    path = os.path.join(HISTORY_DIR, u, pid, "images", cat)
    return send_from_directory(path, fn)

# ---------- Download ZIP ----------
@app.route("/download/<pid>")
@login_required
def download(pid):
    user = session["username"]
    zp = create_zip(user, pid)
    return send_from_directory(os.path.dirname(zp), os.path.basename(zp), as_attachment=True)

# ---------- Socket.IO join ----------
@socketio.on("join")
def on_join(data):
    pid = data.get("pid")
    if pid:
        join_room(pid)

if __name__=="__main__":
    os.makedirs(HISTORY_DIR, exist_ok=True)
    socketio.run(app, host="0.0.0.0", port=port)
