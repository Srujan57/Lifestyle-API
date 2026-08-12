FROM python:3.12-slim

WORKDIR /app

# git is used by entrypoint.sh to pull a pre-built chroma_db/ from a private
# assets repo on hosts with no persistent disk (e.g. Render's free tier),
# instead of re-embedding all PDFs via the OpenAI API on every cold start.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000

EXPOSE 5000

CMD ["sh", "entrypoint.sh"]
