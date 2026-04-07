import os
import json
import re
import datetime
import base64
import secrets
import hashlib
import requests
import logging
from functools import wraps
from time import time

from flask import Flask, request, jsonify, session, redirect
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI
from pymongo import MongoClient
from bson import ObjectId
from bson.errors import InvalidId
from urllib.parse import urlencode
import chromadb

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-me-before-production")
CORS(app, origins=[
    os.getenv("FRONTEND_URL", "http://localhost:3000"),
], supports_credentials=True)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

mongo_client = MongoClient(os.getenv("MONGO_URI"))
db = mongo_client["FitAuth"]
collection = db["Details"]

chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("health_docs")

FITBIT_CLIENT_ID = os.getenv("FITBIT_CLIENT_ID")
FITBIT_CLIENT_SECRET = os.getenv("CLIENT_SECRET")
FITBIT_REDIRECT_URI = os.getenv("REDIRECT_URI", "http://localhost:5000/callback")
NWS_USER_AGENT = os.getenv("NWS_USER_AGENT", "(LifestyleAPI, contact@example.com)")

MAX_HISTORY_MESSAGES = 40
MAX_MESSAGE_LENGTH = 2000
CITY_PATTERN = re.compile(r"^[a-zA-Z\s\-'.]+$")

rate_limit_store = {}
RATE_LIMIT_WINDOW = 60
RATE_LIMIT_MAX = 20


def rate_limit(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        ip = request.remote_addr or "unknown"
        now = time()
        if ip in rate_limit_store:
            timestamps = [t for t in rate_limit_store[ip] if now - t < RATE_LIMIT_WINDOW]
            rate_limit_store[ip] = timestamps
        else:
            rate_limit_store[ip] = []

        if len(rate_limit_store[ip]) >= RATE_LIMIT_MAX:
            return jsonify({"error": "Too many requests. Please wait a moment."}), 429

        rate_limit_store[ip].append(now)
        return f(*args, **kwargs)
    return decorated


def sanitize_city(city: str) -> str | None:
    city = city.strip()[:100]
    if not city or not CITY_PATTERN.match(city):
        return None
    return city


def sanitize_history(history: list) -> list:
    clean = []
    for msg in history:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content", "")
        if role not in ("user", "assistant"):
            continue
        if not isinstance(content, str):
            continue
        clean.append({"role": role, "content": content[:MAX_MESSAGE_LENGTH]})
    return clean


def retrieve_context(query: str, n_results: int = 5) -> str:
    count = chroma_collection.count()
    if count == 0:
        return "No literature has been ingested yet."

    response = openai_client.embeddings.create(
        input=[query],
        model="text-embedding-3-small",
    )
    query_embedding = response.data[0].embedding

    results = chroma_collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, count),
    )

    chunks = results["documents"][0]
    return "\n\n---\n\n".join(chunks)


def _geocode_city(city: str) -> tuple[float, float] | None:
    try:
        res = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=5,
        )
        res.raise_for_status()
        data = res.json()
        results = data.get("results")
        if results:
            return results[0]["latitude"], results[0]["longitude"]
        return None
    except Exception as e:
        logger.warning("Geocoding failed for '%s': %s", city, e)
        return None


def get_weather(city: str = "Columbus") -> str:
    coords = _geocode_city(city)
    if coords is None:
        return f"Weather data unavailable (could not locate '{city}')"

    lat, lon = coords
    nws_headers = {"User-Agent": NWS_USER_AGENT, "Accept": "application/geo+json"}

    try:
        points_res = requests.get(
            f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}",
            headers=nws_headers,
            timeout=5,
        )
        points_res.raise_for_status()
        forecast_url = points_res.json()["properties"]["forecast"]
    except Exception as e:
        logger.warning("NWS points lookup failed: %s", e)
        return "Weather data unavailable (NWS only covers US locations)"

    try:
        forecast_res = requests.get(forecast_url, headers=nws_headers, timeout=5)
        forecast_res.raise_for_status()
        periods = forecast_res.json()["properties"]["periods"]
        current = periods[0]

        temp = current["temperature"]
        unit = current["temperatureUnit"]
        desc = current["shortForecast"]
        wind_speed = current["windSpeed"]
        wind_dir = current["windDirection"]

        return f"{city}: {temp}°{unit}, {desc}, wind {wind_dir} {wind_speed}"
    except Exception as e:
        logger.warning("NWS forecast fetch failed: %s", e)
        return "Weather data unavailable"


def _pkce_code_verifier() -> str:
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("utf-8").rstrip("=")


def _pkce_code_challenge(code_verifier: str) -> str:
    digest = hashlib.sha256(code_verifier.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=")


def _basic_auth_header() -> str:
    return base64.b64encode(
        f"{FITBIT_CLIENT_ID}:{FITBIT_CLIENT_SECRET}".encode()
    ).decode()


def _fitbit_configured() -> bool:
    return bool(FITBIT_CLIENT_ID and FITBIT_CLIENT_SECRET)


def save_tokens(access_token: str, refresh_token: str, user_id: str | None = None) -> str | None:
    token_data = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "client_id": FITBIT_CLIENT_ID,
        "updated_at": datetime.datetime.now(datetime.timezone.utc),
    }
    try:
        if user_id:
            result = collection.update_one(
                {"_id": ObjectId(user_id)},
                {"$set": token_data},
            )
            if result.matched_count > 0:
                return user_id
            logger.warning("save_tokens: no document matched _id=%s", user_id)
            return None

        token_data["created_at"] = datetime.datetime.now(datetime.timezone.utc)
        result = collection.insert_one(token_data)
        return str(result.inserted_id)
    except InvalidId:
        logger.error("save_tokens: invalid ObjectId '%s'", user_id)
        return None
    except Exception as e:
        logger.error("Error saving tokens: %s", e)
        return None


def load_tokens(user_id: str | None = None):
    try:
        if user_id:
            document = collection.find_one({"_id": ObjectId(user_id)})
        else:
            document = collection.find_one(sort=[("updated_at", -1)])

        if document:
            return (
                document.get("access_token"),
                document.get("refresh_token"),
                str(document["_id"]),
            )
        return None, None, None
    except InvalidId:
        logger.error("load_tokens: invalid ObjectId '%s'", user_id)
        return None, None, None
    except Exception as e:
        logger.error("Error loading tokens: %s", e)
        return None, None, None


def refresh_access_token(refresh_token: str, user_doc_id: str | None = None):
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": FITBIT_CLIENT_ID,
    }
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Basic {_basic_auth_header()}",
    }

    res = requests.post("https://api.fitbit.com/oauth2/token", data=data, headers=headers)
    if res.status_code == 200:
        token_data = res.json()
        if user_doc_id:
            save_tokens(token_data["access_token"], token_data["refresh_token"], user_doc_id)
        return token_data

    logger.error("Token refresh failed: %s %s", res.status_code, res.text)
    return None


def fetch_fitbit_summary(access_token: str) -> dict | None:
    today = datetime.date.today().isoformat()
    base = "https://api.fitbit.com/1/user/-"
    headers = {"Authorization": f"Bearer {access_token}"}

    summary = {}

    try:
        r = requests.get(f"{base}/activities/date/{today}.json", headers=headers, timeout=10)
        if r.status_code == 200:
            act = r.json().get("summary", {})
            summary["activity"] = {
                "steps": act.get("steps"),
                "calories_out": act.get("caloriesOut"),
                "active_minutes": act.get("fairlyActiveMinutes", 0) + act.get("veryActiveMinutes", 0),
                "distance_km": None,
            }
            distances = act.get("distances", [])
            for d in distances:
                if d.get("activity") == "total":
                    summary["activity"]["distance_km"] = d.get("distance")
    except Exception as e:
        logger.warning("Fitbit activity fetch failed: %s", e)

    try:
        r = requests.get(f"{base}/sleep/date/{today}.json", headers=headers, timeout=10)
        if r.status_code == 200:
            sleep_data = r.json().get("summary", {})
            summary["sleep"] = {
                "total_minutes_asleep": sleep_data.get("totalMinutesAsleep"),
                "total_time_in_bed": sleep_data.get("totalTimeInBed"),
            }
    except Exception as e:
        logger.warning("Fitbit sleep fetch failed: %s", e)

    try:
        r = requests.get(f"{base}/activities/heart/date/{today}/1d.json", headers=headers, timeout=10)
        if r.status_code == 200:
            hr_data = r.json().get("activities-heart", [])
            if hr_data:
                val = hr_data[0].get("value", {})
                summary["heart_rate"] = {
                    "resting_heart_rate": val.get("restingHeartRate"),
                }
    except Exception as e:
        logger.warning("Fitbit heart-rate fetch failed: %s", e)

    return summary if summary else None


@app.route("/authorize")
def authorize():
    if not _fitbit_configured():
        return jsonify({"error": "Fitbit integration is not configured"}), 503

    code_verifier = _pkce_code_verifier()
    code_challenge = _pkce_code_challenge(code_verifier)
    session["code_verifier"] = code_verifier

    params = {
        "response_type": "code",
        "client_id": FITBIT_CLIENT_ID,
        "redirect_uri": FITBIT_REDIRECT_URI,
        "scope": "activity heartrate sleep profile",
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    return redirect(f"https://www.fitbit.com/oauth2/authorize?{urlencode(params)}")


@app.route("/callback")
def callback():
    if not _fitbit_configured():
        return jsonify({"error": "Fitbit integration is not configured"}), 503

    code = request.args.get("code")
    if not code:
        return jsonify({"error": "Missing authorization code"}), 400

    headers = {
        "Authorization": f"Basic {_basic_auth_header()}",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = {
        "client_id": FITBIT_CLIENT_ID,
        "grant_type": "authorization_code",
        "redirect_uri": FITBIT_REDIRECT_URI,
        "code": code,
        "code_verifier": session.get("code_verifier"),
    }

    res = requests.post("https://api.fitbit.com/oauth2/token", headers=headers, data=data)
    if res.status_code == 200:
        tokens = res.json()
        user_doc_id = save_tokens(tokens["access_token"], tokens["refresh_token"])
        session["user_doc_id"] = user_doc_id
        frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
        return redirect(f"{frontend_url}?fitbit=connected")

    logger.error("Fitbit callback error: %s %s", res.status_code, res.text)
    return jsonify({"error": "Fitbit authorization failed"}), 400


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/endpoint", methods=["POST"])
@rate_limit
def chatbot():
    body = request.get_json()
    if not body:
        return jsonify({"error": "Invalid JSON body"}), 400

    user_message = body.get("message", "")
    if not isinstance(user_message, str):
        return jsonify({"error": "Invalid message"}), 400
    user_message = user_message.strip()[:MAX_MESSAGE_LENGTH]

    raw_history = body.get("history", [])
    if not isinstance(raw_history, list):
        raw_history = []
    history = sanitize_history(raw_history)

    raw_city = body.get("city", "Columbus")
    if not isinstance(raw_city, str):
        raw_city = "Columbus"
    city = sanitize_city(raw_city) or "Columbus"

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        context = retrieve_context(user_message)
    except Exception as e:
        logger.warning("RAG retrieval failed: %s", e)
        context = "No literature context available."

    weather = get_weather(city)

    fitbit_section = ""
    user_doc_id = session.get("user_doc_id")
    if user_doc_id and _fitbit_configured():
        access_token, refresh_token, doc_id = load_tokens(user_doc_id)
        if access_token:
            fitbit_data = fetch_fitbit_summary(access_token)
            if fitbit_data is None and refresh_token:
                new_tokens = refresh_access_token(refresh_token, doc_id)
                if new_tokens:
                    fitbit_data = fetch_fitbit_summary(new_tokens["access_token"])
            if fitbit_data:
                fitbit_section = f"""
USER'S FITBIT DATA (today):
{json.dumps(fitbit_data, indent=2)}
Use this data to personalize recommendations. Reference their actual steps,
sleep, and heart-rate when relevant."""

    system_prompt = f"""You are a supportive health coach chatbot. You help users with:
- Exercise and physical activity recommendations
- SMART goal setting (Specific, Measurable, Achievable, Relevant, Time-bound)
- Motivational Interviewing: ask open-ended questions, reflect what you hear,
  affirm effort, never lecture or push. Let the user arrive at their own conclusions.

STRICT RULE: Only answer based on the CONTEXT below and the user's Fitbit data
(if available). If a question cannot be answered from the context, say:
"I don't have information on that in my knowledge base."
Do not use any outside knowledge.

CURRENT WEATHER:
{weather}
Use this when recommending exercise. If it's raining or below 45°F, suggest indoor alternatives.
{fitbit_section}

CONTEXT FROM HEALTH LITERATURE:
{context}

When helping with SMART goals:
- If a goal is vague, ask clarifying questions one at a time.
- Once you have enough info, reflect the goal back in SMART format.

Keep responses warm, concise, and encouraging."""

    messages = [{"role": "system", "content": system_prompt}]
    truncated_history = history[-MAX_HISTORY_MESSAGES:]
    messages.extend(truncated_history)
    messages.append({"role": "user", "content": user_message})

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=600,
            temperature=0.4,
        )
        reply = response.choices[0].message.content
    except Exception as e:
        logger.error("OpenAI call failed: %s", e)
        return jsonify({"error": "AI call failed"}), 500

    updated_history = truncated_history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": reply},
    ]

    return jsonify({"reply": reply, "history": updated_history})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.getenv("FLASK_DEBUG", "0") == "1")
