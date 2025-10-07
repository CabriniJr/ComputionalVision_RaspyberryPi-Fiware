"""
App headless — Picamera2 (CSI) + YOLOv8
- Sem preview (stub pykms/kms)
- Controle de cor no sensor (AWB/ColourGains) + calibração "gray-world" opcional
- Pré-processamento só para IA: WB (grayworld), gamma e CLAHE (Y)
- Baixa latência: leitura rate-limited, sem fila
- MQTT opcional
"""

from __future__ import annotations
import os, time, sys, types
from typing import Dict, Any, Optional, Tuple

# ==============================
# Stub headless p/ evitar dependência do DRM preview
# ==============================
class _ElasticEnum:
    def __getattr__(self, name): return 0
_stub = types.ModuleType("pykms"); _stub.PixelFormat = _ElasticEnum()
sys.modules.setdefault("pykms", _stub)
sys.modules.setdefault("kms", _stub)

import numpy as np
import cv2
from ultralytics import YOLO

try:
    import paho.mqtt.client as mqtt
except Exception:
    mqtt = None

# ==============================
# CONFIG
# ==============================
CONFIG: Dict[str, Any] = {
    # Câmera
    "capture_width": 1296,     # 1296x972 dá melhor ISP que 640x480
    "capture_height": 972,
    "target_process_fps": 10.0,

    # Exposição
    "picam_control_mode": "indoor",  # "auto" | "indoor" | "manual"
    "manual_exposure_us": 12000,
    "manual_analogue_gain": 8.0,
    "indoor_min_exposure_us": 8000,
    "indoor_min_gain": 4.0,

    # COR no SENSOR (Picamera2/libcamera)
    "color": {
        "mode": "auto_grayworld_once",   # "awb" | "manual" | "auto_grayworld_once"
        "awb_mode": "Auto",              # Auto/Incandescent/Fluorescent/Daylight/Cloudy
        "colour_gains": [3.1, 3.15],     # (R,B) se mode="manual"
        "ev": 0.8,                       # ExposureValue
        "saturation": 1.05,
        "contrast": 1.0,
        "sharpness": 1.0
    },

    # PRÉ-PROCESSAMENTO (só para a IA, não altera snapshots salvos)
    "ai_preproc": {
        "wb": "grayworld",             # "off" | "grayworld"
        "gamma": 1.15,                 # 1.0 = off; 1.1–1.4 em ambiente interno
        "clahe": True,                 # melhora contraste no Y (YCrCb)
        "clahe_clip": 2.0,
        "clahe_grid": 8
    },

    # YOLO
    "yolo_model_path": "yolov8n.onnx",
    "imgsz": 256,
    "onnx_input_size": 640,
    "conf": 0.60,
    "iou": 0.30,
    "max_det": 50,
    "only_person": True,

    # Pós-filtros
    "min_person_area_ratio": 0.012,
    "min_person_height_ratio": 0.10,
    "aspect_ratio_range": (0.25, 0.95),
    "roi_norm": None,

    # Fallback ONNX -> .pt
    "yolo_torch_fallback": "yolov8n.pt",
    "fallback_after_seconds": 8,
    "fallback_pt_imgsz": 320,

    # Cadência / debug
    "no_detect_frames_threshold": 10,
    "print_once_per_second": True,
    "debug_save_idle_frame": True,
    "debug_idle_seconds_to_save": 5,
    "debug_snapshot_path": "/tmp/cam_idle.jpg",
    "debug_last_frame_path": "/tmp/cam_last.jpg",

    # Snapshot periódico único (sempre sobrescrevendo)
    "periodic_snapshot_every_s": 5,
    "periodic_snapshot_path": "/tmp/cam_view.jpg",

    # MQTT
    "mqtt": {
        "enabled": True,
        "host": "IP-FIWARE",
        "port": 1883,
        "fiware_service": "smart",
        "fiware_service_path": "/",
        "apikey": "TEF",
        "device_id": "cam001",
        "qos": 0,
        "retain": False,
    }
}

# ==============================
# Utilidades de cor / pré-processamento
# ==============================
def grayworld_gains_bgr(img: np.ndarray, eps: float = 1e-6) -> Tuple[float,float,float]:
    # WB "gray-world": iguala médias de B,G,R
    means = img.reshape(-1,3).mean(axis=0).astype(np.float64) + eps
    g_avg = means[1]
    gains = g_avg / means  # [B,G,R] -> multiplicadores
    return float(gains[0]), float(gains[1]), float(gains[2])

def apply_wb_grayworld_bgr(img: np.ndarray) -> np.ndarray:
    b,g,r = cv2.split(img)
    gb, gg, gr = grayworld_gains_bgr(img)
    b = cv2.multiply(b, gb); g = cv2.multiply(g, gg); r = cv2.multiply(r, gr)
    out = cv2.merge([b,g,r])
    return np.clip(out, 0, 255).astype(np.uint8)

def apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    if gamma is None or abs(gamma-1.0) < 1e-3: return img
    inv = max(0.1, min(5.0, float(gamma)))
    table = ((np.arange(256)/255.0) ** (1.0/inv) * 255.0).astype(np.uint8)
    return cv2.LUT(img, table)

def apply_clahe_y(img: np.ndarray, clip=2.0, grid=8) -> np.ndarray:
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(grid), int(grid)))
    y2 = clahe.apply(y)
    return cv2.cvtColor(cv2.merge([y2, cr, cb]), cv2.COLOR_YCrCb2BGR)

def preprocess_for_ai(img: np.ndarray, pre: Dict[str,Any]) -> np.ndarray:
    out = img
    if pre.get("wb","off") == "grayworld":
        out = apply_wb_grayworld_bgr(out)
    g = float(pre.get("gamma", 1.0))
    if abs(g-1.0) > 1e-3:
        out = apply_gamma(out, g)
    if pre.get("clahe", False):
        out = apply_clahe_y(out, pre.get("clahe_clip",2.0), pre.get("clahe_grid",8))
    return out

# ==============================
# Picamera2 wrapper (sem preview)
# ==============================
class PiCam2Wrapper:
    def __init__(self, width: int, height: int, target_fps: float, cfg: Dict[str,Any]):
        from picamera2 import Picamera2
        from libcamera import Transform, controls

        self.cfg = cfg
        self.picam2 = Picamera2()
        cam_cfg = self.picam2.create_video_configuration(
            main={"size": (width, height), "format": "BGR888"},
            transform=Transform(hflip=0, vflip=0)
        )
        self.picam2.configure(cam_cfg)

        # Exposição base
        preset = (cfg.get("picam_control_mode","auto") or "auto").lower()
        if preset == "manual":
            self.picam2.set_controls({
                "AeEnable": False,
                "ExposureTime": int(cfg.get("manual_exposure_us", 12000)),
                "AnalogueGain": float(cfg.get("manual_analogue_gain", 8.0)),
                "AwbEnable": True
            })
        elif preset == "indoor":
            self.picam2.set_controls({"AeEnable": True, "AwbEnable": True})
        else:
            self.picam2.set_controls({"AeEnable": True, "AwbEnable": True})

        # Tuning de AE para indoor / contraluz
        try:
            self.picam2.set_controls({
                "ExposureValue": float(cfg["color"].get("ev", 0.0)),
                "Sharpness": float(cfg["color"].get("sharpness", 1.0)),
                "Contrast": float(cfg["color"].get("contrast", 1.0)),
                "Saturation": float(cfg["color"].get("saturation", 1.0)),
                "AeMeteringMode": controls.AeMeteringModeEnum.CentreWeighted,
                "AeConstraintMode": controls.AeConstraintModeEnum.Normal,
            })
        except Exception:
            pass

        # Cor no sensor
        mode = (cfg["color"].get("mode","awb") or "awb").lower()
        if mode == "awb":
            awb_name = (cfg["color"].get("awb_mode","Auto") or "Auto").lower()
            awb_map = {
                "auto": controls.AwbModeEnum.Auto,
                "incandescent": controls.AwbModeEnum.Incandescent,
                "fluorescent": controls.AwbModeEnum.Fluorescent,
                "daylight": controls.AwbModeEnum.Daylight,
                "cloudy": controls.AwbModeEnum.Cloudy,
            }
            self.picam2.set_controls({"AwbEnable": True, "AwbMode": awb_map.get(awb_name, controls.AwbModeEnum.Auto)})

        elif mode == "manual":
            # Atenção: ColourGains = (R, B)
            rb = cfg["color"].get("colour_gains", [2.1, 1.15])
            r = float(np.clip(rb[0], 0.5, 8.0))
            b = float(np.clip(rb[1], 0.5, 8.0))
            self.picam2.set_controls({"AwbEnable": False, "ColourGains": (r, b)})

        elif mode == "auto_grayworld_once":
            # mede alguns frames, calcula gains e fixa no sensor
            self.picam2.set_controls({"AwbEnable": False})
            self.picam2.start()
            time.sleep(0.8)
            samples = []
            for _ in range(6):
                f = self.picam2.capture_array()
                if f is not None and f.size:
                    samples.append(f)
                time.sleep(0.05)
            self.picam2.stop()
            if samples:
                samp = np.median(np.stack(samples, axis=0), axis=0).astype(np.uint8)
                gb, gg, gr = grayworld_gains_bgr(samp)  # gains em B,G,R
                # mapear gains BGR -> ColourGains (R, B) e clamp por segurança
                r_gain = float(np.clip(gr, 0.5, 8.0))
                b_gain = float(np.clip(gb, 0.5, 8.0))
                self.picam2.set_controls({"ColourGains": (r_gain, b_gain)})
            # mantém AwbEnable=False para estabilidade

        # Rate-limit leitura
        self.interval = 1.0 / max(1e-6, float(target_fps))
        self.next_t = time.monotonic()

        self.picam2.start()
        time.sleep(0.8)  # warm-up AE/AWB

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        now = time.monotonic()
        if now < self.next_t:
            time.sleep(min(self.next_t - now, 0.004))
        while self.next_t <= time.monotonic():
            self.next_t += self.interval
        try:
            frame = self.picam2.capture_array()
            if frame is None: return False, None
            if frame.dtype != np.uint8: frame = frame.astype(np.uint8, copy=False)
            if not frame.flags["C_CONTIGUOUS"]: frame = np.ascontiguousarray(frame)
            return True, frame
        except Exception:
            return False, None

    def release(self):
        try: self.picam2.stop()
        except Exception: pass

# ==============================
# YOLO / detecção
# ==============================
def is_onnx(path: str) -> bool:
    return path.lower().endswith(".onnx")

def load_detector(path: str):
    m = YOLO(path)
    if not is_onnx(path):
        try: m.fuse()
        except Exception: pass
    return m

def normalize_frame(f):
    if f is None or not isinstance(f, np.ndarray) or f.size == 0: return None
    if f.ndim == 2: f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
    else:
        if f.shape[2] == 4: f = cv2.cvtColor(f, cv2.COLOR_BGRA2BGR)
        elif f.shape[2] != 3: return None
    if f.dtype != np.uint8: f = f.astype(np.uint8, copy=False)
    if not f.flags["C_CONTIGUOUS"]: f = np.ascontiguousarray(f)
    return f

def summarize_classes(cls_ids: np.ndarray) -> Dict[int,int]:
    out: Dict[int,int] = {}
    if cls_ids.size == 0: return out
    for c in cls_ids.astype(int).tolist():
        out[c] = out.get(c, 0) + 1
    return out

def apply_roi_filter(xyxy: np.ndarray, wh: Tuple[int,int], roi_norm):
    if roi_norm is None or xyxy.size == 0: return np.ones((xyxy.shape[0],), dtype=bool)
    W, H = wh
    x0, y0, x1, y1 = roi_norm
    rx0, ry0, rx1, ry1 = x0*W, y0*H, x1*W, y1*H
    cx = (xyxy[:,0] + xyxy[:,2]) * 0.5
    cy = (xyxy[:,1] + xyxy[:,3]) * 0.5
    return (cx >= rx0) & (cx <= rx1) & (cy >= ry0) & (cy <= ry1)

def apply_geom_filters(xyxy: np.ndarray, wh: Tuple[int,int], aratio: float, hratio: float, ar_range: Tuple[float,float]):
    if xyxy.size == 0: return np.zeros((0,), dtype=bool)
    W, H = wh
    areas = (xyxy[:,2]-xyxy[:,0]) * (xyxy[:,3]-xyxy[:,1])
    heights = (xyxy[:,3]-xyxy[:,1])
    widths  = (xyxy[:,2]-xyxy[:,0])
    ar = widths / np.maximum(heights, 1e-6)
    area_ok = areas >= (aratio * W * H)
    h_ok    = heights >= (hratio * H)
    ar_ok   = (ar >= ar_range[0]) & (ar <= ar_range[1])
    return area_ok & h_ok & ar_ok

def detect_persons(model: YOLO, frame_bgr: np.ndarray, imgsz: int, conf: float, iou: float, max_det: int,
                   only_person: bool, post_cfg: Dict[str,Any]):
    r = model.predict(source=frame_bgr, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det,
                      classes=None, verbose=False, device="cpu")[0]
    if (getattr(r, "boxes", None) is None) or (r.boxes is None): return 0, 0.0, {}
    n = int(len(r.boxes))
    if n == 0: return 0, 0.0, {}
    xyxy = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    cls   = r.boxes.cls.cpu().numpy() if getattr(r.boxes, "cls", None) is not None else np.zeros((n,))
    class_hist = summarize_classes(cls)

    if only_person:
        mask = (cls.astype(int) == 0)
        xyxy, confs = xyxy[mask], confs[mask]
    if xyxy.size == 0: return 0, 0.0, class_hist

    H, W = frame_bgr.shape[:2]
    keep = apply_roi_filter(xyxy, (W,H), post_cfg.get("roi_norm"))
    keep &= apply_geom_filters(xyxy, (W,H),
                               post_cfg.get("min_person_area_ratio",0.01),
                               post_cfg.get("min_person_height_ratio",0.10),
                               post_cfg.get("aspect_ratio_range",(0.2,1.2)))
    xyxy, confs = xyxy[keep], confs[keep]
    count = int(xyxy.shape[0]); conf_max = float(confs.max()) if count>0 else 0.0
    return count, conf_max, class_hist

# ==============================
# MQTT
# ==============================
def ul_payload(state: str, count: int) -> str: return f"s|{state}|c|{count}"
def ul_topic(apikey: str, device_id: str) -> str: return f"/{apikey}/{device_id}/attrs"

def try_mqtt_connect(cfg: Dict[str, Any]):
    if mqtt is None or not cfg.get("enabled", True):
        print("MQTT desativado ou paho-mqtt indisponível → print-only."); return None
    try:
        c = mqtt.Client(); c.username_pw_set(cfg["fiware_service"], cfg["fiware_service_path"])
        c.connect(cfg["host"], int(cfg["port"]), keepalive=60); c.loop_start()
        print(f"MQTT conectado em {cfg['host']}:{cfg['port']}"); return c
    except Exception as e:
        print(f"WARN: Falha ao conectar MQTT: {e} → print-only."); return None

def safe_publish(client, apikey, device_id, state, count, qos=0, retain=False):
    if client is None: print({"count": count, "state": state}); return
    try: client.publish(ul_topic(apikey, device_id), ul_payload(state, count), qos=qos, retain=retain)
    except Exception as e:
        print(f"WARN: erro MQTT: {e} → print-only"); print({"count": count, "state": state})

# ==============================
# MAIN
# ==============================
def main():
    cfg = CONFIG; mqtt_cfg = cfg["mqtt"]

    cam = PiCam2Wrapper(cfg["capture_width"], cfg["capture_height"], cfg["target_process_fps"], cfg)
    mode = "picamera2"
    print("Fonte de vídeo: PICAMERA2 (CSI) — HEADLESS")

    model_path = cfg["yolo_model_path"]
    model = load_detector(model_path)
    effective_imgsz = cfg["imgsz"]
    if is_onnx(model_path):
        onnx_size = int(cfg.get("onnx_input_size", 640))
        if effective_imgsz != onnx_size:
            print(f"ONNX detectado — forçando imgsz={onnx_size} (antes: {effective_imgsz})")
            effective_imgsz = onnx_size

    client = try_mqtt_connect(mqtt_cfg)

    # Snapshot periódico único
    snapshot_every_s = int(cfg.get("periodic_snapshot_every_s", 5))
    snapshot_path = cfg.get("periodic_snapshot_path", "/tmp/cam_view.jpg")
    _next_snap = time.time()

    no_detect_frames = 0
    last_print_second = -1
    silent_since = time.time()

    print("Rodando headless... Ctrl+C para sair.")
    try:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None: time.sleep(0.01); continue
            frame = normalize_frame(frame)
            if frame is None: continue

            # snapshot periódico (sempre sobrescrevendo)
            now_ts = time.time()
            if now_ts >= _next_snap:
                try:
                    cv2.imwrite(snapshot_path, frame)
                except Exception as e:
                    print(f"[snapshot] erro ao salvar: {e}")
                _next_snap = now_ts + snapshot_every_s

            # debug 1x/s
            tick_second = int(time.time())
            if tick_second != last_print_second:
                try:
                    h, w = frame.shape[:2]
                    m = float(frame.mean()); s = float(frame.std())
                    print(f"[frame] {w}x{h} mean={m:.1f} std={s:.1f}")
                    cv2.imwrite(cfg.get("debug_last_frame_path","/tmp/cam_last.jpg"), frame)
                except Exception as e:
                    print(f"[frame] Falha ao salvar/medir frame: {e}")

            # PRÉ-PROC só para IA (corrige cor/iluminação SEM alterar snapshots)
            proc = preprocess_for_ai(frame, cfg.get("ai_preproc", {}))

            # Inferência
            try:
                count, conf_max, class_hist = detect_persons(
                    model, proc, effective_imgsz, cfg["conf"], cfg["iou"], cfg["max_det"],
                    cfg["only_person"],
                    {
                        "min_person_area_ratio": cfg["min_person_area_ratio"],
                        "min_person_height_ratio": cfg["min_person_height_ratio"],
                        "aspect_ratio_range": cfg["aspect_ratio_range"],
                        "roi_norm": cfg["roi_norm"],
                    }
                )
            except Exception as e:
                h, w = frame.shape[:2]
                print(f"ERRO inferência: {type(e).__name__}: {e} | frame={w}x{h}")
                continue

            if count>0 or conf_max>0: silent_since = time.time()
            has_people = count > 0
            if has_people: no_detect_frames = 0
            else:          no_detect_frames += 1

            state = "detected" if has_people else (
                "idle" if no_detect_frames >= int(cfg["no_detect_frames_threshold"]) else None
            )

            # snapshot idle (mantido como fallback legado)
            if cfg.get("debug_save_idle_frame", True):
                idle_frames = no_detect_frames if state in (None,"idle") else 0
                if idle_frames >= int(cfg["debug_idle_seconds_to_save"]) * max(1, int(cfg["target_process_fps"])):
                    try:
                        cv2.imwrite(cfg["debug_snapshot_path"], frame)
                        print(f"[debug] Snapshot salvo: {cfg['debug_snapshot_path']}")
                    except Exception as e:
                        print(f"[debug] Falha ao salvar snapshot: {e}")
                    no_detect_frames = 0

            # log/MQTT 1x/s
            debug_line = (f"[debug] mode={mode} count(person)={count} conf_max={conf_max:.2f} "
                          f"imgsz={effective_imgsz} conf={cfg['conf']} iou={cfg['iou']} classes={class_hist}")
            if cfg.get("print_once_per_second", True):
                if state is not None and tick_second != last_print_second:
                    safe_publish(client, mqtt_cfg["apikey"], mqtt_cfg["device_id"],
                                 state=state, count=(count if state=="detected" else 0),
                                 qos=mqtt_cfg["qos"], retain=mqtt_cfg["retain"])
                    print(f"{debug_line} | state={state}")
                    last_print_second = tick_second
            else:
                if state is not None:
                    safe_publish(client, mqtt_cfg["apikey"], mqtt_cfg["device_id"],
                                 state=state, count=(count if state=="detected" else 0),
                                 qos=mqtt_cfg["qos"], retain=mqtt_cfg["retain"])

            # fallback onnx -> pt
            try:
                fallback_pt = cfg.get("yolo_torch_fallback","")
                timeout_s = int(cfg.get("fallback_after_seconds",8))
                if is_onnx(model_path) and fallback_pt and os.path.isfile(fallback_pt):
                    if (time.time() - silent_since) > timeout_s:
                        print("[fallback] ONNX sem detecções → trocando para PyTorch (.pt)")
                        model_path = fallback_pt
                        model = load_detector(model_path)
                        effective_imgsz = min(effective_imgsz, int(cfg.get("fallback_pt_imgsz",320)))
                        silent_since = time.time()
                        no_detect_frames = 0
            except Exception as e:
                print(f"[fallback] erro ao trocar para .pt: {e}")

    finally:
        try: cam.release()
        except Exception: pass
        try: cv2.destroyAllWindows()
        except Exception: pass

if __name__ == "__main__":
    try:
        cv2.setUseOptimized(True); cv2.setNumThreads(2)
    except Exception: pass
    main()
