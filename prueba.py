from pathlib import Path
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt


# =========================
# VISUALIZACION
# =========================

def ver(img, titulo="", cmap=None, tam=(6, 6)):
    plt.figure(figsize=tam)
    if len(img.shape) == 2:
        plt.imshow(img, cmap=cmap or "gray")
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(titulo)
    plt.axis("off")
    plt.show()


# =========================
# KERNELS Y PALETA
# =========================

k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

paleta = {
    "BLACK":       (70, 70, 70),
    "YELLOW":      (0, 255, 255),
    "GRAY":        (170, 170, 170),
    "BROWN":       (42, 42, 165),
    "GREEN":       (0, 255, 0),
    "LIGHT_GREEN": (120, 255, 120),
    "BLUE":        (255, 0, 0),
    "ORANGE":      (0, 140, 255),
    "RED":         (0, 0, 255),
    "CREAM":       (200, 220, 240),
    "LIGHT":       (230, 230, 230),
}


# =========================
# CARGA Y NORMALIZACION DE JSON
# =========================

def cargar_rangos(ruta_json):
    with open(ruta_json, "r", encoding="utf-8") as f:
        rangos_ref = json.load(f)

    out = {}

    for clave_ref, cfg in rangos_ref.items():
        nuevo = {}

        for color, c in cfg.items():
            nom = color.upper()

            if nom == "CREAM":
                nom = "CREAM"
            elif nom == "LIGHT GREEN":
                nom = "LIGHT_GREEN"
            elif nom == "LIGHT-GREEN":
                nom = "LIGHT_GREEN"

            nuevo[nom] = {
                "lower": c["lower"],
                "upper": c["upper"],
                "open": int(c.get("open", 0)),
                "close": int(c.get("close", 0))
            }

        # Parche mínimo para LH3 si sigue como GRAY + GREEN.
        # Si ya corriges el JSON, esta parte simplemente no molesta.
        if clave_ref == "60429-203_M_LH3":
            if "GRAY" in nuevo and "GREEN" in nuevo:
                nuevo = {
                    "LIGHT_GREEN": {
                        "lower": [30, 20, 80],
                        "upper": [85, 160, 255],
                        "open": 1,
                        "close": 3
                    }
                }

        out[clave_ref] = nuevo

    return out


# =========================
# EXTRACCION DE OBJETO EN IMAGENES DE PILLS
# =========================

def extraer_objeto(img):
    h, w = img.shape[:2]
    b = max(4, min(h, w) // 30)

    top = img[:b, :, :]
    bottom = img[h-b:h, :, :]
    left = img[:, :b, :]
    right = img[:, w-b:w, :]

    bordes = np.concatenate([
        top.reshape(-1, 3),
        bottom.reshape(-1, 3),
        left.reshape(-1, 3),
        right.reshape(-1, 3)
    ], axis=0)

    fondo = np.median(bordes, axis=0).astype(np.uint8)

    dist = np.sqrt(np.sum((img.astype(np.float32) - fondo.astype(np.float32)) ** 2, axis=2))
    dist = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, msk = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    msk = cv2.morphologyEx(msk, cv2.MORPH_OPEN, k3, iterations=1)
    msk = cv2.morphologyEx(msk, cv2.MORPH_CLOSE, k9, iterations=2)

    if np.count_nonzero(msk == 255) > msk.size * 0.65:
        msk = cv2.bitwise_not(msk)

    cnts, _ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt_ext = max(cnts, key=cv2.contourArea)

    msk_obj = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(msk_obj, [cnt_ext], -1, 255, -1)

    x, y, w, h = cv2.boundingRect(cnt_ext)
    crop = img[y:y+h, x:x+w].copy()
    msk_crop = msk_obj[y:y+h, x:x+w].copy()

    return cnt_ext, msk_obj, crop, msk_crop, (x, y, w, h)


# =========================
# EXTRACCION DE OBJETO EN CAMARA
# =========================

ROI_X1 = 180
ROI_Y1 = 120
ROI_X2 = 470
ROI_Y2 = 360

def extraer_objeto_camara(frame):
    roi = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2].copy()

    h, w = roi.shape[:2]
    b = max(4, min(h, w) // 25)

    top = roi[:b, :, :]
    bottom = roi[h-b:h, :, :]
    left = roi[:, :b, :]
    right = roi[:, w-b:w, :]

    bordes = np.concatenate([
        top.reshape(-1, 3),
        bottom.reshape(-1, 3),
        left.reshape(-1, 3),
        right.reshape(-1, 3)
    ], axis=0)

    fondo = np.median(bordes, axis=0).astype(np.uint8)

    dist = np.sqrt(np.sum((roi.astype(np.float32) - fondo.astype(np.float32)) ** 2, axis=2))
    dist = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, msk = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    msk = cv2.morphologyEx(msk, cv2.MORPH_OPEN, k3, iterations=1)
    msk = cv2.morphologyEx(msk, cv2.MORPH_CLOSE, k9, iterations=2)

    if np.count_nonzero(msk == 255) > msk.size * 0.65:
        msk = cv2.bitwise_not(msk)

    cnts, _ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return roi, None, None, None, None, None

    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    area_roi = h * w

    if area < area_roi * 0.03:
        return roi, None, None, None, None, None

    x, y, wc, hc = cv2.boundingRect(cnt)

    toca_borde = (x <= 2) or (y <= 2) or (x + wc >= w - 2) or (y + hc >= h - 2)
    if toca_borde:
        return roi, None, None, None, None, None

    msk_obj = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.drawContours(msk_obj, [cnt], -1, 255, -1)

    crop = roi[y:y+hc, x:x+wc].copy()
    msk_crop = msk_obj[y:y+hc, x:x+wc].copy()

    return roi, cnt, msk_obj, crop, msk_crop, (x, y, wc, hc)


# =========================
# BALANCE DE ILUMINACION
# =========================

def balancear_iluminacion(crop):
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)

    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)


# =========================
# COLOR
# =========================

def construir_mascaras_color(crop_bal, msk_crop, cfg, k3, k5):
    hsv = cv2.cvtColor(crop_bal, cv2.COLOR_BGR2HSV)
    mascaras = {}

    for color, c in cfg.items():
        li = np.array(c["lower"], dtype=np.uint8)
        ls = np.array(c["upper"], dtype=np.uint8)

        m = cv2.inRange(hsv, li, ls)
        m = cv2.bitwise_and(m, msk_crop)

        if c["open"] > 0:
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k3, iterations=c["open"])
        if c["close"] > 0:
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k5, iterations=c["close"])

        mascaras[color] = m

    return hsv, mascaras


def centro_hsv(c):
    li = np.array(c["lower"], dtype=np.float32)
    ls = np.array(c["upper"], dtype=np.float32)
    return (
        (li[0] + ls[0]) / 2.0,
        (li[1] + ls[1]) / 2.0,
        (li[2] + ls[2]) / 2.0
    )


def resolver_solape(hsv, mascaras, cfg, msk_obj):
    nombres = list(mascaras.keys())
    if len(nombres) == 1:
        return mascaras

    hh = hsv[:, :, 0].astype(np.float32)
    ss = hsv[:, :, 1].astype(np.float32)
    vv = hsv[:, :, 2].astype(np.float32)

    union = np.zeros_like(msk_obj)
    for nom in nombres:
        union = cv2.bitwise_or(union, mascaras[nom])

    out = {nom: np.zeros_like(msk_obj) for nom in nombres}
    centros = {nom: centro_hsv(cfg[nom]) for nom in nombres}

    ys, xs = np.where(union > 0)

    for yy, xx in zip(ys, xs):
        candidatos = []

        for nom in nombres:
            if mascaras[nom][yy, xx] == 0:
                continue

            ch, cs, cv = centros[nom]

            dh = abs(hh[yy, xx] - ch)
            dh = min(dh, 180 - dh) / 90.0
            ds = abs(ss[yy, xx] - cs) / 255.0
            dv = abs(vv[yy, xx] - cv) / 255.0

            score = 0.70 * dh + 0.20 * ds + 0.10 * dv
            candidatos.append((score, nom))

        candidatos.sort(key=lambda t: t[0])
        out[candidatos[0][1]][yy, xx] = 255

    for nom in nombres:
        c = cfg[nom]
        m = out[nom]

        if c["open"] > 0:
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k3, iterations=c["open"])
        if c["close"] > 0:
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k5, iterations=c["close"])

        out[nom] = cv2.bitwise_and(m, msk_obj)

    return out


def detectar_colores_global(crop_bal, msk_crop, cfg, k3, k5):
    hsv, mascaras = construir_mascaras_color(crop_bal, msk_crop, cfg, k3, k5)
    mascaras = resolver_solape(hsv, mascaras, cfg, msk_crop)

    area_obj = max(cv2.countNonZero(msk_crop), 1)
    colores = []
    contornos = {}

    for color, m in mascaras.items():
        px = cv2.countNonZero(m)
        frac = px / area_obj

        if frac >= 0.05:
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = [c for c in cnts if cv2.contourArea(c) >= area_obj * 0.01]
            if cnts:
                colores.append(color)
                contornos[color] = cnts

    return mascaras, colores, contornos


# =========================
# FORMA
# =========================

def rotar_horizontal(crop_bal, msk_crop):
    cnts, _ = cv2.findContours(msk_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)

    rect = cv2.minAreaRect(cnt)
    (cx, cy), (rw, rh), ang = rect

    if rw < rh:
        ang += 90.0

    M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)

    h, w = crop_bal.shape[:2]

    rot_img = cv2.warpAffine(
        crop_bal, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    rot_msk = cv2.warpAffine(
        msk_crop, M, (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    cnts2, _ = cv2.findContours(rot_msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt2 = max(cnts2, key=cv2.contourArea)

    x, y, wc, hc = cv2.boundingRect(cnt2)

    rot_img = rot_img[y:y+hc, x:x+wc].copy()
    rot_msk = rot_msk[y:y+hc, x:x+wc].copy()

    return rot_img, rot_msk


def detectar_borde_medio(rot_img, rot_msk):
    g = cv2.cvtColor(rot_img, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0)

    inner = cv2.erode(rot_msk, k3, iterations=2)

    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    mag = np.abs(gx)

    h, w = inner.shape
    x1, x2 = int(w * 0.25), int(w * 0.75)
    y1, y2 = int(h * 0.20), int(h * 0.80)

    perfil = []

    for x in range(x1, x2):
        col_mask = inner[y1:y2, x] > 0
        if np.count_nonzero(col_mask) < max(5, int((y2 - y1) * 0.20)):
            perfil.append(0.0)
            continue

        vals = mag[y1:y2, x][col_mask]
        perfil.append(float(np.mean(vals)))

    perfil = np.array(perfil, dtype=np.float32)

    if len(perfil) >= 5:
        perfil = np.convolve(perfil, np.ones(5) / 5.0, mode="same")

    pos_rel = int(np.argmax(perfil))
    seam_x = x1 + pos_rel

    media = float(np.mean(perfil) + 1e-6)
    score = float(np.max(perfil) / media)

    center_err = abs(seam_x - (w / 2.0)) / max(w, 1)

    return seam_x, score, center_err, perfil


def detectar_forma_refinada(crop_bal, msk_crop):
    rot_img, rot_msk = rotar_horizontal(crop_bal, msk_crop)

    cnts, _ = cv2.findContours(rot_msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)

    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)

    hull = cv2.convexHull(cnt)
    area_hull = max(cv2.contourArea(hull), 1.0)

    x, y, w, h = cv2.boundingRect(cnt)
    ar = max(w, h) / max(1, min(w, h))

    circ = 4 * np.pi * area / max(peri * peri, 1.0)
    extent = area / max(w * h, 1.0)
    solidity = area / area_hull

    seam_x, seam_score, center_err, perfil = detectar_borde_medio(rot_img, rot_msk)

    if ar <= 1.18 and circ >= 0.82:
        forma = "circular"
    else:
        if ar >= 1.30 and seam_score >= 1.40 and center_err <= 0.12:
            forma = "pildora"
        else:
            forma = "rectangular"

    return {
        "forma": forma,
        "rot_img": rot_img,
        "rot_msk": rot_msk,
        "cnt": cnt,
        "ar": ar,
        "circ": circ,
        "extent": extent,
        "solidity": solidity,
        "seam_x": seam_x,
        "seam_score": seam_score,
        "center_err": center_err,
        "perfil": perfil
    }


# =========================
# COLOR ESPECIFICO PARA PILDORAS BICOLOR
# =========================

def detectar_colores_pildora_por_borde(rot_img, rot_msk, cfg, seam_x, k3, k5):
    hsv, mascaras = construir_mascaras_color(rot_img, rot_msk, cfg, k3, k5)
    mascaras = resolver_solape(hsv, mascaras, cfg, rot_msk)

    h, w = rot_msk.shape[:2]
    seam_x = int(np.clip(seam_x, int(w * 0.25), int(w * 0.75)))

    left_half = np.zeros_like(rot_msk)
    right_half = np.zeros_like(rot_msk)

    left_half[:, :seam_x] = rot_msk[:, :seam_x]
    right_half[:, seam_x:] = rot_msk[:, seam_x:]

    area_left = max(cv2.countNonZero(left_half), 1)
    area_right = max(cv2.countNonZero(right_half), 1)

    datos_left = []
    datos_right = []

    for color, m in mascaras.items():
        ml = cv2.bitwise_and(m, left_half)
        mr = cv2.bitwise_and(m, right_half)

        frac_l = cv2.countNonZero(ml) / area_left
        frac_r = cv2.countNonZero(mr) / area_right

        datos_left.append((frac_l, color, ml))
        datos_right.append((frac_r, color, mr))

    datos_left.sort(reverse=True)
    datos_right.sort(reverse=True)

    frac_l, color_l, mask_l = datos_left[0]
    frac_r, color_r, mask_r = datos_right[0]

    colores = []
    contornos = {}

    if frac_l >= 0.08:
        colores.append(color_l)
        cnts_l, _ = cv2.findContours(mask_l, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_l = [c for c in cnts_l if cv2.contourArea(c) >= area_left * 0.02]
        if cnts_l:
            contornos[color_l] = cnts_l

    if frac_r >= 0.08:
        cnts_r, _ = cv2.findContours(mask_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_r = [c for c in cnts_r if cv2.contourArea(c) >= area_right * 0.02]

        if color_r not in colores:
            colores.append(color_r)
            if cnts_r:
                contornos[color_r] = cnts_r
        else:
            if cnts_r:
                contornos[color_r] = contornos.get(color_r, []) + cnts_r

    return mascaras, colores, contornos, seam_x


def detectar_colores_por_tipo(crop_bal, msk_crop, cfg, info_forma, k3, k5):
    forma = info_forma["forma"]

    if forma == "pildora" and len(cfg) == 2:
        mascaras_fin, colores_detectados, contornos_color, seam_x = detectar_colores_pildora_por_borde(
            info_forma["rot_img"],
            info_forma["rot_msk"],
            cfg,
            info_forma["seam_x"],
            k3,
            k5
        )

        base_vis = info_forma["rot_img"].copy()
        base_msk = info_forma["rot_msk"].copy()

        cnts_f, _ = cv2.findContours(base_msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt_f = max(cnts_f, key=cv2.contourArea)

        cv2.drawContours(base_vis, [cnt_f], -1, (255, 255, 255), 2)
        cv2.line(base_vis, (seam_x, 0), (seam_x, base_vis.shape[0] - 1), (255, 255, 255), 1)

    else:
        mascaras_fin, colores_detectados, contornos_color = detectar_colores_global(
            crop_bal, msk_crop, cfg, k3, k5
        )

        base_vis = crop_bal.copy()
        cnts_f, _ = cv2.findContours(msk_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt_f = max(cnts_f, key=cv2.contourArea)
        cv2.drawContours(base_vis, [cnt_f], -1, (255, 255, 255), 2)

    return mascaras_fin, colores_detectados, contornos_color, base_vis


# =========================
# PROCESAMIENTO DE IMAGENES DE PILLS
# =========================

def procesar_imagen(ruta_img, clave_ref, rangos_ref, k3, k5, k9, paleta):
    img = cv2.imread(str(ruta_img))

    if img is None:
        print("no se pudo cargar:", ruta_img)
        return

    if clave_ref not in rangos_ref:
        print("sin configuración en json:", clave_ref)
        return

    cnt_ext, msk_obj, crop, msk_crop, bbox = extraer_objeto(img)
    crop_bal = balancear_iluminacion(crop)

    cfg = rangos_ref[clave_ref]
    info_forma = detectar_forma_refinada(crop_bal, msk_crop)

    mascaras_fin, colores_detectados, contornos_color, final = detectar_colores_por_tipo(
        crop_bal, msk_crop, cfg, info_forma, k3, k5
    )

    for color, cnts_c in contornos_color.items():
        for c in cnts_c:
            cv2.drawContours(final, [c], -1, paleta.get(color, (255, 255, 255)), 2)

    txt_col = ", ".join(colores_detectados)

    cv2.putText(final, f"forma: {info_forma['forma']}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(final, f"colores: {txt_col}", (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    vis_ext = img.copy()
    cv2.drawContours(vis_ext, [cnt_ext], -1, (255, 255, 255), 2)

    print("archivo:", ruta_img.name)
    print("clave:", clave_ref)
    print("forma:", info_forma["forma"])
    print("colores:", colores_detectados)
    print("aspect ratio:", round(info_forma["ar"], 3))
    print("circularidad:", round(info_forma["circ"], 3))
    print("extent:", round(info_forma["extent"], 3))
    print("solidity:", round(info_forma["solidity"], 3))

    ver(msk_obj, f"{ruta_img.name} - máscara externa", cmap="gray", tam=(8, 4))
    ver(vis_ext, f"{ruta_img.name} - contorno externo", tam=(8, 4))
    ver(crop_bal, f"{ruta_img.name} - crop balanceado", tam=(8, 4))

    for color, m in mascaras_fin.items():
        ver(m, f"{ruta_img.name} - máscara {color}", cmap="gray", tam=(6, 4))

    ver(final, f"{ruta_img.name} - resultado final", tam=(8, 4))


# =========================
# PROCESAMIENTO DE CAMARA
# =========================

def procesar_frame(frame, clave_ref, rangos_ref, k3, k5, k9, paleta):
    vis = frame.copy()
    cv2.rectangle(vis, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (255, 255, 255), 2)

    if clave_ref not in rangos_ref:
        cv2.putText(vis, "clave sin json", (ROI_X1, ROI_Y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return vis, None

    roi, cnt_ext, msk_obj, crop, msk_crop, bbox = extraer_objeto_camara(frame)

    if crop is None:
        cv2.putText(vis, "sin objeto valido", (ROI_X1, ROI_Y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return vis, None

    crop_bal = balancear_iluminacion(crop)
    cfg = rangos_ref[clave_ref]

    info_forma = detectar_forma_refinada(crop_bal, msk_crop)

    mascaras_fin, colores_detectados, contornos_color, final = detectar_colores_por_tipo(
        crop_bal, msk_crop, cfg, info_forma, k3, k5
    )

    for color, cnts_c in contornos_color.items():
        for c in cnts_c:
            cv2.drawContours(final, [c], -1, paleta.get(color, (255, 255, 255)), 2)

    roi_vis = roi.copy()
    if cnt_ext is not None:
        cv2.drawContours(roi_vis, [cnt_ext], -1, (255, 255, 255), 2)

    vis[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2] = roi_vis

    txt_col = ", ".join(colores_detectados)

    cv2.putText(vis, f"forma: {info_forma['forma']}", (ROI_X1, ROI_Y2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(vis, f"colores: {txt_col}", (ROI_X1, ROI_Y2 + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    datos = {
        "forma": info_forma["forma"],
        "colores": colores_detectados,
        "crop_bal": crop_bal,
        "msk_crop": msk_crop,
        "mascaras_fin": mascaras_fin,
        "contornos_color": contornos_color,
        "final": final,
        "bbox_roi": bbox,
        "ar": info_forma["ar"],
        "circ": info_forma["circ"],
        "extent": info_forma["extent"],
        "solidity": info_forma["solidity"],
        "seam_x": info_forma["seam_x"],
        "seam_score": info_forma["seam_score"],
        "center_err": info_forma["center_err"]
    }

    return vis, datos
RUTA_JSON = Path("hsv_debug_ranges_v2.json")
DIR_IMG = Path("pills")

rangos_ref = cargar_rangos(RUTA_JSON)
imgs = sorted(list(DIR_IMG.glob("*.jpg")) + list(DIR_IMG.glob("*.png")) + list(DIR_IMG.glob("*.jpeg")))

for ruta_img in imgs:
    clave_ref = ruta_img.stem
    if clave_ref not in rangos_ref:
        print("sin configuración en json:", ruta_img.name)
        continue
    procesar_imagen(ruta_img, clave_ref, rangos_ref, k3, k5, k9, paleta)