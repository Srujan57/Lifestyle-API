#!/bin/sh

# ---------------------------------------------------------------------------
# Populate chroma_db/ on first run.
#
# Preferred path: pull a pre-built chroma_db/ from a private GitHub repo
# (CHROMA_ASSETS_REPO, e.g. "yourname/lifestyle-api-chroma-assets") using a
# read-only fine-grained PAT (GH_ASSET_TOKEN) set as a platform secret env
# var. This keeps the copyrighted source PDFs / their embedded text out of
# the public Lifestyle-API repo (same reason documents/ is gitignored there),
# while avoiding re-embedding every PDF via the OpenAI API on every cold
# start on hosts with no persistent disk (e.g. Render's free tier).
#
# Fallback: if those env vars aren't set, or the pull fails, run ingest.py
# to build chroma_db/ from documents/ locally, same as before. On hosts with
# a real persistent volume (Railway, a VM), this only runs once — subsequent
# restarts see chroma_db/ already populated and skip straight to the app.
#
# Locally: delete chroma_db/ and restart to force a re-ingest after adding
# new documents.
# ---------------------------------------------------------------------------
CHROMA_DIR="$(dirname "$0")/chroma_db"

populate_from_assets_repo() {
    [ -n "$CHROMA_ASSETS_REPO" ] && [ -n "$GH_ASSET_TOKEN" ] || return 1
    echo "Fetching pre-built chroma_db/ from ${CHROMA_ASSETS_REPO}..."
    rm -rf /tmp/chroma-assets
    git clone --depth 1 \
        "https://x-access-token:${GH_ASSET_TOKEN}@github.com/${CHROMA_ASSETS_REPO}.git" \
        /tmp/chroma-assets || return 1
    [ -d /tmp/chroma-assets/chroma_db ] || return 1
    mkdir -p "$CHROMA_DIR"
    cp -r /tmp/chroma-assets/chroma_db/. "$CHROMA_DIR"/
    rm -rf /tmp/chroma-assets
    [ -n "$(ls -A "$CHROMA_DIR" 2>/dev/null)" ]
}

if [ "${FORCE_REINGEST:-0}" = "1" ]; then
    echo "FORCE_REINGEST=1 — running document ingestion..."
    python ingest.py
    echo "Ingestion complete."
elif [ ! -d "$CHROMA_DIR" ] || [ -z "$(ls -A "$CHROMA_DIR" 2>/dev/null)" ]; then
    if populate_from_assets_repo; then
        echo "chroma_db/ populated from assets repo."
    else
        echo "chroma_db/ not found or empty and no assets repo available — running document ingestion..."
        python ingest.py
        echo "Ingestion complete."
    fi
else
    echo "chroma_db/ already populated — skipping ingestion."
fi

echo "Starting Flask app..."
# --limit-request-field_size / --limit-request-fields raised above gunicorn's
# defaults (8190 bytes / 100 fields). A large Cookie header (e.g. an
# accumulated session cookie, or a stale cookie left over from another local
# app that used this same port) can exceed the default limit; gunicorn then
# drops the connection without sending a real HTTP response, which the
# browser reports as a bare "TypeError: Failed to fetch" instead of a
# diagnosable error. Raising the limit avoids that failure mode for
# legitimately larger (but not abusive) header sets.
exec gunicorn --bind 0.0.0.0:"${PORT:-5000}" --workers 1 --timeout 120 \
    --limit-request-field_size 16380 --limit-request-fields 200 app:app
