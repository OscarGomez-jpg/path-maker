from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import base64
import numpy as np
from processor import DepthProcessor
import socket
import os

"""Path-Maker Server with Auto-IP Detection.

This module provides a Flask-SocketIO server that auto-detects its local IP
address to facilitate connections over mobile hotspots or different Wi-Fi networks.
"""

app = Flask(__name__)
app.config["SECRET_KEY"] = "path-maker-secret!"
socketio = SocketIO(app, cors_allowed_origins="*", max_decode_packets=1000000)

processor = DepthProcessor()


def get_local_ip():
    """Detects the current local IP address of the machine.

    Returns:
        str: The local IP address (e.g., '192.168.43.10').
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Connect to an external IP (doesn't send data) to find local interface
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("image")
def handle_image(data):
    try:
        header, encoded = data.split(",", 1)
        nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return

        params = {"min_area": 1000}
        points_data, path_points, debug_img = processor.process_frame(frame, params)

        h, w = frame.shape[:2]
        d_res = cv2.resize(debug_img, (w, h))
        cv2.imshow("Path-Maker Server (Navigator Mode)", d_res)
        cv2.waitKey(1)

        emit("response", {"points": points_data, "path": path_points})

    except Exception as e:
        print(f"Server Error: {e}")


if __name__ == "__main__":
    # 1. AUTO-DETECT CURRENT IP
    current_ip = get_local_ip()

    print("\n" + "=" * 50)
    print("PATH-MAKER SERVER ACTIVE")
    print("Network Mode: Local Network / Hotspot")
    print(f"Target URL: https://{current_ip}:5000")
    print("=" * 50 + "\n")

    # 2. GENERATE DYNAMIC SSL CERTIFICATE FOR THIS IP
    from OpenSSL import crypto
    import tempfile

    key = crypto.PKey()
    key.generate_key(crypto.TYPE_RSA, 2048)
    cert = crypto.X509()
    # Use the detected IP as the Common Name (CN)
    cert.get_subject().CN = current_ip
    cert.set_serial_number(1000)
    cert.gmtime_adj_notBefore(0)
    cert.gmtime_adj_notAfter(31536000)
    cert.set_issuer(cert.get_subject())
    cert.set_pubkey(key)
    cert.sign(key, "sha256")

    with (
        tempfile.NamedTemporaryFile(delete=False) as cert_file,
        tempfile.NamedTemporaryFile(delete=False) as key_file,
    ):
        cert_file.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
        key_file.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, key))
        cert_path = cert_file.name
        key_path = key_file.name

    socketio.run(
        app,
        host="0.0.0.0",
        port=5000,
        debug=False,
        certfile=cert_path,
        keyfile=key_path,
    )
