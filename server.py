from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import base64
import numpy as np
from processor import DepthProcessor

"""Path-Maker Server.

This module provides a Flask-SocketIO server that receives images from a 
remote client (mobile phone), processes them using the Robot Navigator 
(FastSAM), and sends back object vertices and a safe navigation path.
"""

app = Flask(__name__)
app.config["SECRET_KEY"] = "path-maker-secret!"
# Increase max packet size to support high-frequency image streaming
socketio = SocketIO(app, cors_allowed_origins="*", max_decode_packets=1000000)

# Initialize the FastSAM processor
processor = DepthProcessor()


@app.route("/")
def index():
    """Renders the main mobile interface."""
    return render_template("index.html")


@socketio.on("image")
def handle_image(data):
    """Processes incoming camera frames from the client.

    Decodes the base64 image, runs navigation logic, displays a monitoring
    dashboard on the PC, and emits the calculated path and obstacles back
    to the mobile device.

    Args:
        data: Base64 encoded JPEG image string.
    """
    try:
        # Decode base64 image from client
        header, encoded = data.split(",", 1)
        nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return

        params = {"min_area": 1000}

        # Returns: detected object points, safe path nodes, and debug image
        points_data, path_points, debug_img = processor.process_frame(frame, params)

        # PC Dashboard Visualization
        h, w = frame.shape[:2]
        d_res = cv2.resize(debug_img, (w, h))
        cv2.imshow("Path-Maker Server Monitoring (FastSAM)", d_res)
        cv2.waitKey(1)

        # Emit results back to mobile client
        emit("response", {"points": points_data, "path": path_points})

    except Exception as e:
        print(f"Server Error: {e}")


if __name__ == "__main__":
    """Initializes the server with an ad-hoc SSL certificate for mobile camera access."""
    print("Path-Maker Server starting in HTTPS mode.")
    print("Connect your device to: https://192.168.1.11:5000")

    # MANUAL SSL GENERATION FOR GEVENT
    # Mobile browsers require HTTPS to enable the camera API (navigator.mediaDevices)
    from OpenSSL import crypto
    import tempfile

    key = crypto.PKey()
    key.generate_key(crypto.TYPE_RSA, 2048)
    cert = crypto.X509()
    cert.get_subject().CN = "192.168.1.11"
    cert.set_serial_number(1000)
    cert.gmtime_adj_notBefore(0)
    cert.gmtime_adj_notAfter(31536000)
    cert.set_issuer(cert.get_subject())
    cert.set_pubkey(key)
    cert.sign(key, "sha256")

    # Create temporary files for the SSL credentials
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
