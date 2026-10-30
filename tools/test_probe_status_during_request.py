import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from probe_status_during_request import observe


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        time.sleep(0.08)
        body = json.dumps({"usage": {"completion_tokens": 8}}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        body = b'{"status":"ok"}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class StatusProbeTests(unittest.TestCase):
    def test_overlapping_status_requests(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            result = observe(
                f"http://127.0.0.1:{server.server_port}",
                "fake",
                "text",
                interval=0.01,
                status_timeout=1,
                request_timeout=1,
            )
            self.assertTrue(result["completion"]["ok"])
            self.assertGreaterEqual(len(result["status_samples"]), 2)
            self.assertTrue(all(sample["ok"] for sample in result["status_samples"]))
        finally:
            server.shutdown()
            server.server_close()
            worker.join()

    def test_invalid_timing_rejected(self):
        with self.assertRaises(ValueError):
            observe("http://127.0.0.1:1", "fake", "text", interval=0)


if __name__ == "__main__":
    unittest.main()
