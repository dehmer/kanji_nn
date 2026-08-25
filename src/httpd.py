#!/usr/bin/env python3

import sys
import os
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from functools import partial
import numpy as np

from kanji_nn.data import Character
from kanji_nn.pipelines import post_pipeline


class JSONRequestHandler(BaseHTTPRequestHandler):
    def __init__(self, dataset, *args, **kwargs):
        self.dataset = dataset
        self.pipeline = post_pipeline(dataset)
        super().__init__(*args, **kwargs)

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        content_type = self.headers.get("Content-Type", "")

        if content_length == 0:
            self.send_error_response(400, "Empty body")
            return

        if "application/json" not in content_type:
            self.send_error_response(415, "Unsupported Media Type. Use application/json")
            return

        raw_data = self.rfile.read(content_length)
        json_data = json.loads(raw_data.decode("utf-8"))
        literal = json_data["literal"]
        code_point = f"U+{format(ord(literal), "X")}"

        timestamp = np.array(json_data["timestamp"], dtype=np.float32)      # 0
        dx = np.array(json_data["dx"], dtype=np.float32)                    # 1
        dy = np.array(json_data["dy"], dtype=np.float32)                    # 2
        pressure = np.array(json_data["pressure"], dtype=np.float32)        # 3
        down = np.array(json_data["down"], dtype=np.float32)                # 4

        raw = np.column_stack([timestamp, dx, dy, pressure, down])
        filename = f"data/dataset/{dataset}/npy-raw/{code_point}.npy"
        np.save(filename, raw)

        character = Character.of_npy(self.dataset, filename)
        strokes = character.strokes()
        [self.pipeline(stroke) for stroke in strokes]

        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()


    def send_error_response(self, status_code, message):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        response = json.dumps({"error": message}).encode("utf-8")
        self.wfile.write(response)


def run(dataset):
    server_address = ("0.0.0.0", 3000)
    handler_factory = partial(JSONRequestHandler, dataset)

    httpd = HTTPServer(server_address, handler_factory)
    print("httpd up on port 8080...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nserver shut down.")
        httpd.server_close()


if __name__ == "__main__":
    if (len(sys.argv)) < 2:
        raise Exception("no output dir given")

    dataset = sys.argv[1]

    dirs = [
        f"data/dataset/{dataset}",
        f"data/dataset/{dataset}/npy-raw",
        f"data/dataset/{dataset}/png-raw"
    ]

    for dir in dirs:
        if os.path.exists(dir): continue
        os.mkdir(dir)

    run(dataset)
