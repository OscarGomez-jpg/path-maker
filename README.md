# Path-Maker v1.1 🤖🚀

Path-Maker is a real-time autonomous navigation prototype that converts a mobile phone's camera feed into a structured 3D-aware navigation map. It identifies objects, detects the floor, and calculates a safe path for a robot to follow, all using a high-performance Client-Server architecture.

## 🌟 Key Features

*   **Semantic Segmentation:** Uses **FastSAM-s** (Fast Segment Anything) to identify object boundaries with pixel-level precision.
*   **Automatic Floor Detection:** Heuristically identifies the walkable ground plane from the robot's starting position.
*   **Safety Erosion:** Applies mathematical morphological operations to create a safety buffer around obstacles, ensuring the path never clips an edge.
*   **Real-time Pathfinding:** A custom scan-line centerline algorithm that finds the safest gap through complex terrain at 5 FPS.
*   **Mobile AR Viewer:** A web-based client that renders the navigation path and object contours directly over the live camera feed using HTTPS/WebSockets.
*   **Zero-Config Networking:** Auto-detects local IP addresses and generates dynamic SSL certificates for immediate use over Wi-Fi or Mobile Hotspots.

## 🏗️ Architecture

```text
[ Mobile Phone ] <--- (WebSockets/HTTPS) ---> [ PC Server (AI Hub) ]
  - Camera Capture                              - Pre-processing (CLAHE/Gamma)
  - 5 FPS Streaming                             - FastSAM Inference
  - AR Path Rendering                           - Safety Erosion & Pathfinding
                                                - PC Dashboard Monitoring
```

## 🛠️ Tech Stack

*   **AI Backend:** PyTorch, Ultralytics (FastSAM).
*   **Computer Vision:** OpenCV (Contour analysis, Morphological operations).
*   **Server:** Flask, Flask-SocketIO, Gevent, PyOpenSSL.
*   **Frontend:** HTML5, CSS3, JavaScript (Socket.io).

## 🚀 Getting Started

### 1. Requirements
Ensure you have Python 3.8+ installed. You will need a PC with a decent CPU (or NVIDIA GPU for faster inference).

```bash
pip install flask flask-socketio eventlet ultralytics opencv-python numpy pyOpenSSL
```

### 2. Run the Server
The server will automatically detect your local IP and generate a temporary SSL certificate (required for mobile camera access).

```bash
python server.py
```

### 3. Connect your Phone
1.  Check the terminal output for the **Target URL** (e.g., `https://192.168.1.11:5000`).
2.  Open the URL in your mobile browser.
3.  **Accept the SSL Warning:** Since the certificate is self-signed, you must click "Advanced" and "Proceed to site (unsafe)."
4.  Click **START NAVIGATOR**.

## 🧠 Technical Deep Dive

### Mathematical Erosion
To account for the physical width of a robot, we don't just find any path. We apply a 15x15 kernel erosion to the floor mask. This "eats away" the edges of the walkable area, creating a mathematical buffer zone that keeps the calculated path at a safe distance from all detected obstacles.

### Scan-Line Centerline Pathfinding
The algorithm performs horizontal scans of the "Safe Floor" at multiple depth levels. For each scan, it identifies the largest continuous walkable segment and targets its geometric center. These centers are then connected to form a smooth, stable navigation guide.

## 📝 License
MIT License - Created for educational and prototyping purposes in autonomous robotics.
