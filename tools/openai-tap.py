"""openai-tap.py PORT UPSTREAM_URL LOG: a pass-through proxy for OpenAI chat servers (streaming included). Appends every POST body as one JSON line to LOG.
The history in a request holds the assistant's earlier tool_calls, so LOG shows what an agent CLI asked its tools to do, in one format for every CLI."""
import http.client, json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
port, up, log = int(sys.argv[1]), urlparse(sys.argv[2]), sys.argv[3]
class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"  # the body ends when the connection closes, so SSE streams pass through unchanged
    def log_message(self, *a): pass
    def _fwd(self):
        n = int(self.headers.get("content-length") or 0); body = self.rfile.read(n) if n else b""
        if self.command == "POST":
            try: open(log, "a").write(json.dumps(json.loads(body)) + "\n")
            except Exception: pass
        c = http.client.HTTPConnection(up.hostname, up.port, timeout=900)
        c.request(self.command, self.path, body, {k: v for k, v in self.headers.items() if k.lower() not in ("host", "connection")})
        r = c.getresponse(); self.send_response(r.status)
        for k, v in r.getheaders():
            if k.lower() not in ("transfer-encoding", "connection", "content-length"): self.send_header(k, v)
        self.end_headers()
        try:
            while True:
                b = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
                if not b: break
                self.wfile.write(b); self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # the client gave up (a CLI timeout or abort): nothing to forward
    do_GET = do_POST = _fwd
ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
