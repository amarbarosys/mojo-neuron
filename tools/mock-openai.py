"""mock-openai.py PORT LOG: a minimal OpenAI chat server for gates: replies "reply N" (N = user turns so far), usage grows with the history,
appends every request body to LOG, and answers a request whose last user text is FAIL with a body that has no choices."""
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
port, log = int(sys.argv[1]), sys.argv[2]
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, o):
        b = json.dumps(o).encode(); self.send_response(200); self.send_header("content-type", "application/json"); self.send_header("content-length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        self._send({"data": [{"id": "mock-model"}]} if self.path == "/v1/models" else {"ok": True})
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"]))); open(log, "a").write(json.dumps(body) + "\n")
        msgs = body["messages"]; n = sum(m["role"] == "user" for m in msgs)
        if msgs[-1]["content"] == "FAIL": return self._send({"error": "boom"})
        self._send({"choices": [{"message": {"role": "assistant", "content": "reply %d" % n}}], "usage": {"prompt_tokens": 10 * len(msgs), "completion_tokens": 2}})
HTTPServer(("127.0.0.1", port), H).serve_forever()
