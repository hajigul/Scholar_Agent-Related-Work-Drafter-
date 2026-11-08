import os

MODEL = os.environ.get("SCHOLAR_MODEL", "llama3.2:3b")
PORT = int(os.environ.get("SCHOLAR_PORT", "8000"))
KEEP_ALIVE = "30m"        # keep the model in memory between questions
CACHE_TTL = 6 * 3600      # seconds to reuse search results
CONTACT_EMAIL = os.environ.get("SCHOLAR_EMAIL", "")  # optional: OpenAlex serves you faster