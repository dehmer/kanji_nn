#!/usr/bin/env python3

import sys
import os
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from functools import partial
import numpy as np
import kanji_nn.plot.character as character

class JSONRequestHandler(BaseHTTPRequestHandler):
    def __init__(self, output_dir, *args, **kwargs):
        self.output_dir = f"data/{output_dir}"
        super().__init__(*args, **kwargs)

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        content_type = self.headers.get('Content-Type', '')

        if content_length == 0:
            self.send_error_response(400, "Empty body")
            return

        if 'application/json' not in content_type:
            self.send_error_response(415, "Unsupported Media Type. Use application/json")
            return

        raw_data = self.rfile.read(content_length)
        try:
            try:
                json_data = json.loads(raw_data.decode('utf-8'))
                literal = json_data['literal']
                code_point = f"U+{format(ord(literal), 'X')}"

                timestamp = np.array(json_data['timestamp'], dtype=np.float32)      # 0
                dx = np.array(json_data['dx'], dtype=np.float32)                    # 1
                dy = np.array(json_data['dy'], dtype=np.float32)                    # 2
                pressure = np.array(json_data['pressure'], dtype=np.float32)        # 3
                orientation = np.array(json_data['orientation'], dtype=np.float32)  # 4
                tilt = np.array(json_data['tilt'], dtype=np.float32)                # 5
                down = np.array(json_data['down'], dtype=np.float32)                # 6

                raw = np.vstack([timestamp, dx, dy, pressure, orientation, tilt, down]).T

                dirs = [
                    f'data/{output_dir}',
                    f'data/{output_dir}/npy.7-raw',
                    f'data/{output_dir}/png-raw'
                ]

                for dir in dirs:
                    if os.path.exists(dir): continue
                    os.mkdir(dir)

                np.save(f'data/{output_dir}/npy.7-raw/{code_point}.npy', raw)
                character.save(f'data/{output_dir}/png-raw/{code_point}', raw)
                # character.show(raw)

            except (ValueError, TypeError) as error:
                msg = "error handling request"
                print(msg, error)
                self.send_error_response(400, msg)
                return

            self.send_response(200)
            self.send_header('Content-Length', '0')
            self.end_headers()

        except json.JSONDecodeError:
            self.send_error_response(400, "Invalid JSON format")

    def send_error_response(self, status_code, message):
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        response = json.dumps({"error": message}).encode('utf-8')
        self.wfile.write(response)

def run(output_dir):
    server_address = ('0.0.0.0', 3000)
    handler_factory = partial(JSONRequestHandler, output_dir)

    httpd = HTTPServer(server_address, handler_factory)
    print("httpd up on port 8080...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nserver shut down.")
        httpd.server_close()

if __name__ == '__main__':
    if (len(sys.argv)) < 2:
        raise Exception('no output dir given')

    output_dir = sys.argv[1]
    run(output_dir)
