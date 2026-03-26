from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import base64
import numpy as np
from processor import DepthProcessor
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'path-maker-secret!'
socketio = SocketIO(app, cors_allowed_origins="*", max_decode_packets=1000000)

# Inicializar el procesador de FastSAM
processor = DepthProcessor()

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('image')
def handle_image(data):
    try:
        header, encoded = data.split(",", 1)
        nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None: return

        # Obtenemos puntos de objetos, el camino y la imagen de debug
        points_data, path_points, debug_img = processor.process_frame(frame, params)
        
        # Visualización en PC
        h, w = frame.shape[:2]
        d_res = cv2.resize(debug_img, (w, h))
        cv2.imshow("Path-Maker Server View (Robot Navigator)", d_res)
        cv2.waitKey(1)

        # Enviar vértices y CAMINO al teléfono
        emit('response', { 
            'points': points_data,
            'path': path_points
        })
        
    except Exception as e:
        print(f"Error en el servidor: {e}")

if __name__ == "__main__":
    print("Servidor FastSAM iniciado en HTTPS.")
    print("Conecta tu teléfono a: https://192.168.1.11:5000")
    
    # --- GENERAR SSL MANUAL PARA GEVENT ---
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
    cert.sign(key, 'sha256')

    with tempfile.NamedTemporaryFile(delete=False) as cert_file, \
         tempfile.NamedTemporaryFile(delete=False) as key_file:
        cert_file.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
        key_file.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, key))
        cert_path = cert_file.name
        key_path = key_file.name

    socketio.run(app, host="0.0.0.0", port=5000, debug=False, certfile=cert_path, keyfile=key_path)
