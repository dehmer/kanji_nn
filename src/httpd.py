#!/usr/bin/env python3

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np
import kanji_nn.plot as plot

class JSONRequestHandler(BaseHTTPRequestHandler):
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
            json_data = json.loads(raw_data.decode('utf-8'))

            # Validierung aller erforderlichen Attribute:
            required_keys = ['literal', 'xs', 'ys', 'fs', 'ts']
            if not all(key in json_data for key in required_keys):
                self.send_error_response(400, "Missing required attributes. Needed: xs, ys, fs, ts")
                return
            try:
                literal = json_data['literal']
                xs_array = np.array(json_data['xs'], dtype=np.float32)
                ys_array = np.array(json_data['ys'], dtype=np.float32)
                fs_array = np.array(json_data['fs'], dtype=np.float32) # status feature
                ts_array = np.array(json_data['ts'], dtype=np.float32)
                ps_array = np.array(json_data['ps'], dtype=np.float32)

                code_point = f"U+{format(ord(literal), 'X')}"
                strokes = np.vstack([xs_array, ys_array, fs_array]).T
                plot.save(f'images/{code_point}.png', strokes, (6, 6))
                # plot.show(strokes, (10, 10))

                blob = np.vstack([ts_array, xs_array, ys_array, ps_array, fs_array]).T
                np.save(f'strokes/{code_point}.npy', blob)

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

def run():
    server_address = ('0.0.0.0', 3000)
    httpd = HTTPServer(server_address, JSONRequestHandler)
    print("Server läuft auf allen Schnittstellen (Port 8080)...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer wird beendet.")
        httpd.server_close()

if __name__ == '__main__':
    run()
