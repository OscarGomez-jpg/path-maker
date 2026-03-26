import cv2
import torch
import numpy as np

# 1. Modelo optimizado (Small)
model_type = "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device).eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# 2. Captura de video
cap = cv2.VideoCapture(0)

main_window = "Path-Maker v1.1"
cv2.namedWindow(main_window)


def nothing(x):
    pass


first_frame = True
smoothed_depth = None
tracked_objects = []


def apply_preprocessing(frame, gamma=1.0, clahe_clip=2.0):
    # Sin padding, directo sobre el frame
    invGamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    img = cv2.LUT(frame, table)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    img = cv2.merge((cl, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    return img


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    if first_frame:
        cv2.createTrackbar("Simplicidad", main_window, 4, 25, nothing)
        cv2.createTrackbar("Area Min", main_window, 1500, 8000, nothing)
        cv2.createTrackbar("Capas", main_window, 8, 20, nothing)
        cv2.createTrackbar("Nitidez Prof.", main_window, 3, 10, nothing)
        cv2.createTrackbar("Est. Puntos", main_window, 60, 95, nothing)
        cv2.createTrackbar("Est. Mapa", main_window, 85, 95, nothing)
        first_frame = False

    # Leer Controles
    simplicity = cv2.getTrackbarPos("Simplicidad", main_window) / 100.0
    min_area = cv2.getTrackbarPos("Area Min", main_window)
    num_layers = cv2.getTrackbarPos("Capas", main_window)
    if num_layers < 1:
        num_layers = 1
    depth_sharp = cv2.getTrackbarPos("Nitidez Prof.", main_window)
    point_stab = cv2.getTrackbarPos("Est. Puntos", main_window) / 100.0
    map_stab = cv2.getTrackbarPos("Est. Mapa", main_window) / 100.0

    # 3. Pre-procesamiento de Imagen
    processed_frame = apply_preprocessing(frame, gamma=1.3, clahe_clip=3.0)

    # 4. Predicción de profundidad
    img_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        # Interpolamos al tamaño ORIGINAL del frame
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    # Aumentar contraste local de la profundidad
    if depth_sharp > 0:
        kernel = np.array([[-1, -1, -1], [-1, 9 + depth_sharp, -1], [-1, -1, -1]])
        depth_map = cv2.filter2D(depth_map, -1, kernel)

    depth_map_norm = cv2.normalize(
        depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )

    # Estabilización temporal del mapa
    if smoothed_depth is None:
        smoothed_depth = depth_map_norm.astype(float)
    else:
        cv2.accumulateWeighted(depth_map_norm, smoothed_depth, map_stab)
    depth_final = cv2.convertScaleAbs(smoothed_depth)
    depth_final = cv2.bilateralFilter(depth_final, 9, 75, 75)

    depth_color = cv2.applyColorMap(depth_final, cv2.COLORMAP_MAGMA)

    # 5. Procesamiento por CAPAS
    points_mask = np.zeros_like(frame)
    current_objects = []
    layer_size = 255 // num_layers

    for i in range(num_layers):
        lower = i * layer_size
        upper = (i + 1) * layer_size
        if lower < 60:
            continue

        layer_mask = cv2.inRange(depth_final, lower, upper)
        layer_mask = cv2.morphologyEx(
            layer_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)
        )
        contours, _ = cv2.findContours(
            layer_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        color = (0, 255 - (i * (200 // num_layers)), 50 + (i * (200 // num_layers)))
        for cnt in contours:
            # Filtro de Aspect Ratio para evitar olas horizontales
            x, y, w_cnt, h_cnt = cv2.boundingRect(cnt)
            if float(w_cnt) / h_cnt > 4.0 or cv2.contourArea(cnt) < min_area:
                continue

            approx = cv2.approxPolyDP(cnt, simplicity * cv2.arcLength(cnt, True), True)

            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                for old_obj in tracked_objects:
                    dist = np.sqrt(
                        (cx - old_obj["cx"]) ** 2 + (cy - old_obj["cy"]) ** 2
                    )
                    if dist < 60 and len(approx) == len(old_obj["points"]):
                        approx = (
                            approx.astype(float) * (1 - point_stab)
                            + old_obj["points"].astype(float) * point_stab
                        ).astype(np.int32)
                        break
                current_objects.append({"cx": cx, "cy": cy, "points": approx})

            cv2.drawContours(points_mask, [approx], 0, color, 2)
            for p in approx:
                cv2.circle(points_mask, tuple(p[0]), 5, color, -1)

    tracked_objects = current_objects

    # --- UI & COMPOSICIÓN ---
    h, w = frame.shape[:2]
    f_res = cv2.resize(frame, (w // 2, h // 2))
    d_res = cv2.resize(depth_color, (w // 2, h // 2))
    p_res = cv2.resize(processed_frame, (w // 2, h // 2))
    v_res = cv2.resize(points_mask, (w // 2, h // 2))

    font = cv2.FONT_HERSHEY_SIMPLEX

    def add_label(img, text, pos=(15, 25)):
        cv2.putText(img, text, pos, font, 0.6, (0, 0, 0), 3)
        cv2.putText(img, text, pos, font, 0.6, (255, 255, 255), 1)

    add_label(f_res, "1. Original")
    add_label(d_res, "2. Profundidad (Filtrada)")
    add_label(p_res, "3. IA Input (Pre-Proc)")
    add_label(v_res, "4. Objetos Estructurales")

    top_row = np.hstack((f_res, d_res))
    bot_row = np.hstack((p_res, v_res))
    canvas = np.vstack((top_row, bot_row))

    cv2.putText(
        canvas,
        f"Objetos: {len(tracked_objects)}",
        (w - 150, h - 20),
        font,
        0.5,
        (0, 255, 0),
        1,
    )
    cv2.imshow(main_window, canvas)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
