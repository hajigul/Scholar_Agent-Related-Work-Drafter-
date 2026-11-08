import argparse
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import ollama

import config
from related_work import MODEL, run_draft

HERE = Path(__file__).parent


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, (HERE / "rw.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/models":
            try:
                models = [m.model for m in ollama.list().models]
            except Exception:
                models = []
            self._send(200, json.dumps({"models": models, "default": MODEL}).encode(), "application/json")
        else:
            self._send(404, b"Not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/draft":
            self._send(404, b"Not found", "text/plain")
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except Exception:
            self._send(400, b"Bad request", "text/plain")
            return
        text = str(body.get("text", ""))[:3000].strip()
        if not text:
            self._send(400, b"Empty text", "text/plain")
            return
        model = str(body.get("model") or MODEL)
        papers = body.get("papers")
        if isinstance(papers, list):
            need = ("title", "published", "summary", "link")
            papers = [p for p in papers if isinstance(p, dict) and all(k in p for k in need)] or None

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            for ev in run_draft(text, model, papers):
                self.wfile.write((json.dumps(ev) + "\n").encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", nargs="?", type=int, default=8001)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"Related-work drafter running at {url}  (Ctrl+C to stop)")
    threading.Thread(target=lambda: ollama.generate(model=MODEL, prompt="", keep_alive=config.KEEP_ALIVE),
                     daemon=True).start()
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()