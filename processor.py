import cv2
import numpy as np
from ultralytics import FastSAM

class DepthProcessor:
    def __init__(self):
        # Cargar FastSAM para segmentación rápida
        self.model = FastSAM("FastSAM-s.pt")
        self.device = "cpu"
        print(f"Robot Navigator (FastSAM) cargado en {self.device}")

    def apply_preprocessing(self, frame, gamma=1.2, clahe_clip=2.0):
        # Mejorar visibilidad para segmentación
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        img = cv2.LUT(frame, table)
        
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8,8))
        cl = clahe.apply(l)
        img = cv2.merge((cl,a,b))
        img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
        return img

    def process_frame(self, frame, params):
        min_area = params.get('min_area', 1000)
        h, w = frame.shape[:2]
        
        # 1. Inferencia de FastSAM
        results = self.model(frame, device=self.device, retina_masks=True, imgsz=320, conf=0.4, iou=0.9)[0]
        
        points_data = []
        path_points = []
        floor_mask = np.zeros((h, w), dtype=np.uint8)
        
        if results.masks is not None:
            # Identificar el Suelo (Heurística: Segmento en la parte inferior central)
            start_point = (w // 2, h - 20)
            masks_data = results.masks.data.cpu().numpy() # Formato (N, H, W)
            
            # Buscar la máscara que contiene el punto de inicio del robot
            floor_idx = -1
            for i, mask in enumerate(masks_data):
                # Redimensionar máscara al tamaño original si es necesario
                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                
                if mask[start_point[1], start_point[0]] > 0:
                    floor_idx = i
                    floor_mask = (mask * 255).astype(np.uint8)
                    break
            
            # Procesar el resto como obstáculos y enviar sus contornos
            for i, mask_pts in enumerate(results.masks.xy):
                pts = np.array(mask_pts, dtype=np.int32)
                if cv2.contourArea(pts) < min_area: continue
                
                # Si no es el suelo, enviamos sus puntos
                if i != floor_idx:
                    points_data.append({
                        'type': 'obstacle',
                        'points': pts[::3].tolist(), # Muestrear para velocidad
                        'color': [0, 0, 255] # Rojo en el envío
                    })
                else:
                    # Enviar el contorno del suelo (opcional, para visualización)
                    points_data.append({
                        'type': 'floor',
                        'points': pts[::4].tolist(),
                        'color': [0, 255, 0] # Verde
                    })

            # 2. CALCULAR CAMINO (PATHFINDING)
            if floor_idx != -1:
                # Erosionar el suelo para "alejar" el camino de las paredes/objetos
                kernel = np.ones((15, 15), np.uint8)
                safe_floor = cv2.erode(floor_mask, kernel, iterations=1)
                
                # Escanear de abajo hacia arriba en pasos
                for y in range(h - 20, h // 3, -40):
                    row = safe_floor[y, :]
                    # Encontrar el segmento blanco más largo en esta fila
                    on_segments = np.where(row > 0)[0]
                    if len(on_segments) > 0:
                        # Buscamos grupos continuos de pixeles blancos
                        diff = np.diff(on_segments)
                        splits = np.where(diff > 1)[0] + 1
                        segments = np.split(on_segments, splits)
                        
                        # El camino va por el centro del segmento más grande
                        best_seg = max(segments, key=len)
                        target_x = int(np.mean(best_seg))
                        path_points.append([target_x, y])

        # Imagen de Debug para el PC
        debug_img = results.plot()
        if len(path_points) > 1:
            pts_array = np.array(path_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(debug_img, [pts_array], False, (255, 255, 0), 5) # Cyan path
            for p in path_points:
                cv2.circle(debug_img, tuple(p), 8, (255, 255, 255), -1)

        return points_data, path_points, debug_img
