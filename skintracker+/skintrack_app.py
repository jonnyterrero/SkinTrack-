# skintrack_app.py
# SkinTrack+ — Chronic Skin Condition Tracker (Streamlit)
# -------------------------------------------------------
# Tracks eczema, psoriasis (incl. guttate), keratosis pilaris, acne, melanoma, vitiligo,
# contact dermatitis, and cold sores. Logs symptoms/meds, analyzes images to quantify
# area, ΔE (CIELAB) vs background ring, border irregularity, asymmetry, redness (R/G),
# and supports non-ML segmentation (KMeans / GrabCut), optional U-Net, ArUco scale,
# neutral color-card white balance, trends, med schedule, CSV/PDF export, simulation.
#
# Run:
#   pip install -r requirements.txt
#   streamlit run skintrack_app.py
#
# IMPORTANT: This is NOT a diagnostic tool. Concerning changes -> see a clinician.

import io
import os
import base64
import datetime as dt
import sqlite3
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

import cv2
from skimage.color import rgb2lab, deltaE_ciede2000
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

import plotly.express as px
import plotly.graph_objects as go

import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# -------- Optional ML (U-Net) ----------
try:
    import torch
    import torchvision.transforms as T
    TORCH_OK = True
except Exception:
    TORCH_OK = False

# ---------------------------
# Config & setup
# ---------------------------
st.set_page_config(page_title="SkinTrack+", layout="wide")
DATA_DIR = Path("skintrack_data")
IMAGES_DIR = DATA_DIR / "images"
DB_PATH = DATA_DIR / "skintrack.db"
MODEL_DIR = DATA_DIR / "models"
DATA_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

CONDITIONS = [
    "eczema",
    "psoriasis",
    "guttate psoriasis",
    "keratosis pilaris",
    "cystic/hormonal acne",
    "melanoma",
    "vitiligo",
    "contact dermatitis",
    "cold sores",
]

TRIGGER_SUGGESTIONS = [
    "stress", "sweat/exercise", "fragrance", "detergent",
    "cosmetics", "weather - cold/dry", "weather - hot/humid",
    "pollen", "dust mites", "pet dander", "new products",
]

# ---------------------------
# DB helpers
# ---------------------------
def init_db():
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS lesions(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT,
                condition TEXT
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS records(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lesion_id INTEGER,
                ts TEXT,
                img_path TEXT,
                itch INTEGER,
                pain INTEGER,
                sleep REAL,
                stress INTEGER,
                triggers TEXT,
                new_products TEXT,
                meds_taken TEXT,
                adherence INTEGER,
                notes TEXT,
                area_cm2 REAL,
                redness REAL,
                border_irreg REAL,
                asymmetry REAL,
                depig_deltaE REAL
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS med_schedule(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lesion_id INTEGER,
                name TEXT,
                dose TEXT,
                morning INTEGER,
                afternoon INTEGER,
                evening INTEGER,
                notes TEXT
            );
        """)
        con.commit()

def list_lesions():
    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql_query("SELECT id, label, condition FROM lesions ORDER BY id DESC", con)

def insert_lesion(label, condition):
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("INSERT INTO lesions(label, condition) VALUES(?, ?)", (label, condition))
        con.commit()
        return cur.lastrowid

def insert_record(lesion_id, ts, img_path, itch, pain, sleep, stress, triggers, new_products,
                  meds_taken, adherence, notes, metrics):
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("""
            INSERT INTO records(
                lesion_id, ts, img_path, itch, pain, sleep, stress, triggers, new_products,
                meds_taken, adherence, notes, area_cm2, redness, border_irreg, asymmetry, depig_deltaE
            )
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            lesion_id, ts, img_path, int(itch), int(pain), float(sleep), int(stress),
            triggers, new_products, meds_taken, int(bool(adherence)), notes,
            metrics.get("area_cm2"), metrics.get("redness"), metrics.get("border_irreg"),
            metrics.get("asymmetry"), metrics.get("depig_deltaE")
        ))
        con.commit()

def lesion_history(lesion_id):
    with sqlite3.connect(DB_PATH) as con:
        df = pd.read_sql_query("""
            SELECT * FROM records WHERE lesion_id=? ORDER BY ts ASC
        """, con, params=(lesion_id,))
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"])
    return df

def med_schedule_for(lesion_id):
    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql_query("SELECT * FROM med_schedule WHERE lesion_id=? ORDER BY id ASC", con, params=(lesion_id,))

def upsert_med_schedule(lesion_id, name, dose, morning, afternoon, evening, notes):
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("""
            INSERT INTO med_schedule(lesion_id, name, dose, morning, afternoon, evening, notes)
            VALUES(?,?,?,?,?,?,?)
        """, (lesion_id, name, dose, int(bool(morning)), int(bool(afternoon)), int(bool(evening)), notes))
        con.commit()

# ---------------------------
# Image helpers
# ---------------------------
def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

def ensure_three_channels(img_bgr):
    if img_bgr is None: return None
    if len(img_bgr.shape) == 2:
        return cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    if img_bgr.shape[2] == 4:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
    return img_bgr

def estimate_scale_cm_per_px(img_bgr, use_aruco=False, marker_side_cm=2.0):
    # Fallback ~50 px/cm (0.02 cm/px)
    fallback = 0.02
    if not use_aruco:
        return fallback
    try:
        aruco = cv2.aruco
    except AttributeError:
        return fallback
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    dict_ = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dict_, params)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(corners) == 0:
        return fallback
    c = corners[0].reshape(-1, 2)
    d01 = np.linalg.norm(c[0]-c[1]); d12 = np.linalg.norm(c[1]-c[2])
    d23 = np.linalg.norm(c[2]-c[3]); d30 = np.linalg.norm(c[3]-c[0])
    side_px = np.mean([d01, d12, d23, d30])
    return float(marker_side_cm / (side_px + 1e-6))

def white_balance_neutral_roi(img_bgr, roi_rect):
    # Diagonal white balance from a user-chosen neutral patch
    x, y, w, h = [int(v) for v in roi_rect]
    x = max(0, x); y = max(0, y); w = max(1, w); h = max(1, h)
    patch = img_bgr[y:y+h, x:x+w, :].astype(np.float32)
    if patch.size == 0: return img_bgr
    means = np.mean(patch.reshape(-1,3), axis=0) + 1e-6
    target = np.mean(means)
    gains = target / means
    img = img_bgr.astype(np.float32)
    img[:,:,0] *= gains[0]; img[:,:,1] *= gains[1]; img[:,:,2] *= gains[2]
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

# ---------------------------
# Segmentation (non-ML + optional ML) & metrics
# ---------------------------
def segment_kmeans(img_bgr, condition="eczema", K=3):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    X = img_rgb.reshape(-1,3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _c, labels, centers = cv2.kmeans(X, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(h,w)
    # Heuristic cluster selection
    rg_scores, L_scores = [], []
    for k in range(K):
        m = (labels==k)
        if not np.any(m):
            rg_scores.append(-1e9); L_scores.append(-1e9); continue
        vals = img_rgb[m]
        R = vals[:,0].astype(np.float32); G = vals[:,1].astype(np.float32)+1e-6
        rg_scores.append(float(np.mean(R/G)))
        sample = vals[::max(1, len(vals)//1000)]
        L = rgb2lab(sample.reshape(-1,1,3)/255.0)[:,:,0]
        L_scores.append(float(np.mean(L)))
    if condition == "vitiligo":
        target = int(np.argmax(L_scores))
    elif condition == "melanoma":
        target = int(np.argmin(L_scores))
    else:
        target = int(np.argmax(rg_scores))
    mask = (labels==target).astype(np.uint8)*255
    mask = cv2.medianBlur(mask,5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    return mask

def segment_grabcut_rect(img_bgr, rect):
    # rect = (x,y,w,h) from user; use GrabCut to refine foreground
    x,y,w,h = [int(v) for v in rect]
    x = max(0,x); y = max(0,y); w=max(1,w); h=max(1,h)
    mask = np.zeros(img_bgr.shape[:2], np.uint8)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    try:
        cv2.grabCut(img_bgr, mask, (x,y,w,h), bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    except Exception:
        return np.zeros_like(mask, dtype=np.uint8)
    mask2 = np.where((mask==cv2.GC_FGD) | (mask==cv2.GC_PR_FGD), 255, 0).astype('uint8')
    mask2 = cv2.medianBlur(mask2,5)
    return mask2

# Optional: U-Net hook (expects a binary mask output, same HxW)
def segment_unet(img_bgr) -> Optional[np.ndarray]:
    model_path = MODEL_DIR / "unet_skin.pth"
    if not TORCH_OK or not model_path.exists():
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Very lightweight dummy U-Net loader; replace with your architecture
    class TinyUNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(3, 16, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(16, 1, 1)
            )
        def forward(self, x): return self.conv(x)

    model = TinyUNet().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception:
        return None
    model.eval()

    H, W = img_bgr.shape[:2]
    tfm = T.Compose([
        T.ToPILImage(),
        T.Resize((512, 512)),
        T.ToTensor()
    ])
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = tfm(rgb).unsqueeze(0).to(device)  # 1x3x512x512
    with torch.no_grad():
        y = model(x)  # 1x1x512x512
        y = torch.sigmoid(y)[0,0].cpu().numpy()
    y = (y > 0.5).astype(np.uint8)*255
    mask = cv2.resize(y, (W, H), interpolation=cv2.INTER_NEAREST)
    mask = cv2.medianBlur(mask,5)
    return mask

def ring_mask(mask, inner_pad=6, ring_width=12):
    kernel_inner = np.ones((inner_pad, inner_pad), np.uint8)
    kernel_outer = np.ones((ring_width+inner_pad, ring_width+inner_pad), np.uint8)
    inner = cv2.dilate(mask, kernel_inner, iterations=1)
    outer = cv2.dilate(mask, kernel_outer, iterations=1)
    ring = cv2.subtract(outer, inner)
    return ring

def contour_main(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    return max(cnts, key=cv2.contourArea)

def compute_metrics(img_bgr, mask, cm_per_px=0.02):
    out = dict(area_cm2=None, border_irreg=None, asymmetry=None, redness=None, depig_deltaE=None)
    cnt = contour_main(mask)
    if cnt is None: return out
    area_px = cv2.contourArea(cnt)
    perim_px = cv2.arcLength(cnt, True)
    out["area_cm2"] = float(area_px * (cm_per_px**2))
    out["border_irreg"] = float((perim_px**2)/(4*np.pi*area_px + 1e-6))
    # Asymmetry by halves around centroid
    M = cv2.moments(cnt)
    if M["m00"]>0:
        cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
    else:
        cx,cy = img_bgr.shape[1]//2, img_bgr.shape[0]//2
    left = mask[:, :cx] > 0; right = mask[:, cx:] > 0
    w_min = min(left.shape[1], right.shape[1])
    left = left[:, -w_min:]; right = right[:, :w_min]
    asym_lr = abs(left.sum()-right.sum())/(left.sum()+right.sum()+1e-6)
    top = mask[:cy, :] > 0; bottom = mask[cy:, :] > 0
    h_min = min(top.shape[0], bottom.shape[0])
    top = top[-h_min:, :]; bottom = bottom[:h_min, :]
    asym_tb = abs(top.sum()-bottom.sum())/(top.sum()+bottom.sum()+1e-6)
    out["asymmetry"] = float(max(asym_lr, asym_tb))
    # Redness (R/G)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    m = mask.astype(bool)
    if np.any(m):
        R = rgb[:,:,0][m].astype(np.float32); G = rgb[:,:,1][m].astype(np.float32)+1e-6
        out["redness"] = float(np.mean(R/G))
    # ΔE(CIEDE2000) lesion vs ring background
    ring = ring_mask(mask)
    r = ring.astype(bool)
    if np.any(m) and np.any(r):
        lab = rgb2lab(rgb/255.0)
        lesion_lab = np.mean(lab[m], axis=0, keepdims=True)
        bg_lab = np.mean(lab[r], axis=0, keepdims=True)
        out["depig_deltaE"] = float(deltaE_ciede2000(lesion_lab, bg_lab).mean())
    return out

def overlay_mask(img_bgr, mask, alpha=0.4):
    color_mask = np.zeros_like(img_bgr); color_mask[:,:,2] = mask
    return cv2.addWeighted(img_bgr, 1.0, color_mask, alpha, 0)

# ---------------------------
# PDF Export
# ---------------------------
def export_pdf_summary(lesion_row, hist_df, med_df, out_path):
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("<b>SkinTrack+ Summary</b>", styles["Title"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"Lesion: {lesion_row['label']}  |  Condition: {lesion_row['condition']}", styles["Normal"]))
    story.append(Paragraph(f"Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    story.append(Spacer(1, 10))

    if not hist_df.empty:
        last = hist_df.iloc[-1]
        story.append(Paragraph("<b>Latest Metrics</b>", styles["Heading3"]))
        data = [
            ["Timestamp", str(last["ts"])],
            ["Area (cm²)", f"{last.get('area_cm2', np.nan):.3f}" if pd.notna(last.get('area_cm2')) else "—"],
            ["Border Irregularity", f"{last.get('border_irreg', np.nan):.3f}" if pd.notna(last.get('border_irreg')) else "—"],
            ["Asymmetry", f"{last.get('asymmetry', np.nan):.3f}" if pd.notna(last.get('asymmetry')) else "—"],
            ["Redness (R/G)", f"{last.get('redness', np.nan):.3f}" if pd.notna(last.get('redness')) else "—"],
            ["ΔE (lesion vs bg)", f"{last.get('depig_deltaE', np.nan):.3f}" if pd.notna(last.get('depig_deltaE')) else "—"],
            ["Itch/Pain/Sleep/Stress", f"{last['itch']}/{last['pain']}/{last['sleep']}/{last['stress']}"],
            ["Triggers", last.get("triggers","")],
            ["New products", last.get("new_products","")],
            ["Meds taken", last.get("meds_taken","")],
            ["Adherence", "Yes" if last.get("adherence",0) else "No"],
            ["Notes", last.get("notes","")]
        ]
        t = Table(data, hAlign="LEFT", colWidths=[140, 400])
        t.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
            ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke)
        ]))
        story.append(t)
        story.append(Spacer(1, 10))

    story.append(Paragraph("<b>Medication Schedule</b>", styles["Heading3"]))
    if med_df is not None and not med_df.empty:
        data = [["Name","Dose","Morning","Afternoon","Evening","Notes"]]
        for _,r in med_df.iterrows():
            data.append([r["name"], r["dose"], "✓" if r["morning"] else "",
                         "✓" if r["afternoon"] else "", "✓" if r["evening"] else "", r.get("notes","")])
        t = Table(data, hAlign="LEFT", colWidths=[120,80,70,70,70,150])
        t.setStyle(TableStyle([("GRID", (0,0), (-1,-1), 0.25, colors.grey),
                               ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke)]))
        story.append(t)
    else:
        story.append(Paragraph("No schedule saved.", styles["Normal"]))

    story.append(Spacer(1, 12))
    story.append(Paragraph("<b>Notes</b>", styles["Heading3"]))
    story.append(Paragraph("Photos are stored locally. This PDF summarizes key numbers for your clinician. "
                           "Lighting tip: indirect daylight at ~30–40 cm, keep marker/color card near lesion.", styles["Normal"]))

    doc = SimpleDocTemplate(str(out_path), pagesize=letter)
    doc.build(story)

# ---------------------------
# Simulation (simple 'what-if')
# ---------------------------
def simulate_outcomes(days, start_area_cm2, base_decay=0.01, med_potency=0.02,
                      adherence=1.0, stress_level=3, sleep_hours=7.0, trigger_load=0):
    """
    Very simple discrete-time model of lesion area response.
    area_{t+1} = area_t * (1 - base_decay - med_potency*adherence + 0.005*(stress_level-3)
                           + 0.006*(7-sleep_hours) + 0.01*trigger_load)
    Clipped at >= 0. This is illustrative only.
    """
    area = max(0.0, start_area_cm2)
    ts = []
    for _ in range(days):
        multiplier = 1 - base_decay - med_potency*adherence + 0.005*(stress_level-3) + 0.006*(7-sleep_hours) + 0.01*trigger_load
        multiplier = max(0.0, multiplier)  # prevent negative factor
        area = max(0.0, area * multiplier)
        ts.append(area)
    return np.array(ts)

# ---------------------------
# UI
# ---------------------------
init_db()
st.title("🧴 SkinTrack+ (Prototype)")
st.info("Tip: take photos in **indirect daylight**, keep the camera **~30–40 cm** away, and place the **ArUco marker or color card** near the lesion.")

st.caption("This app helps track chronic skin conditions with photos, metrics, meds, and trends. Not a diagnostic device.")

tab_log, tab_trends, tab_meds, tab_export, tab_sim = st.tabs(["📥 Log Entry", "📈 Trends", "💊 Med Schedule", "📤 Export", "🧪 Simulate"])

# ---------------------------
# LOG ENTRY TAB
# ---------------------------
with tab_log:
    st.subheader("Create/select a lesion")
    col1, col2 = st.columns(2)
    with col1:
        existing = list_lesions()
        lesion_choice = None
        if not existing.empty:
            options = {f"{r['label']} [{r['condition']}] (#{r['id']})": int(r["id"]) for _, r in existing.iterrows()}
            lesion_label = st.selectbox("Pick an existing lesion", list(options.keys()))
            if lesion_label:
                lesion_choice = options[lesion_label]
        else:
            st.info("No lesions yet — create one below.")
    with col2:
        with st.form("new_lesion_form", clear_on_submit=True):
            new_label = st.text_input("New lesion label (e.g., left forearm A)")
            new_condition = st.selectbox("Condition", CONDITIONS, index=0)
            submitted = st.form_submit_button("➕ Create lesion")
        if submitted and new_label:
            lesion_choice = insert_lesion(new_label, new_condition)
            st.success(f"Created lesion #{lesion_choice}: {new_label} [{new_condition}]")

    st.markdown("---")
    st.subheader("Image capture & calibration")

    colA, colB = st.columns([2,1])
    with colA:
        use_camera = st.checkbox("Use camera (otherwise upload)", value=False)
        pil_img = None
        if use_camera:
            cam = st.camera_input("Capture photo")
            if cam is not None:
                pil_img = Image.open(cam)
        else:
            up = st.file_uploader("Upload photo (JPG/PNG)", type=["jpg","jpeg","png"])
            if up is not None:
                pil_img = Image.open(up)

        if pil_img is not None:
            st.caption("Tip: take photos in indirect daylight; keep the camera ~30–40 cm away; place the reference near the lesion.")
            st.image(pil_img, caption="Original", use_container_width=True)

        use_aruco = st.checkbox("Use ArUco for scale (DICT_4X4_50)?", value=False)
        marker_cm = st.number_input("Marker side length (cm)", 0.5, 10.0, 2.0, 0.5)

        st.markdown("**Optional color calibration** — draw a rectangle over a neutral gray/white patch (e.g., color card).")
        if pil_img is not None:
            canvas_cal = st_canvas(
                fill_color="rgba(0,0,0,0)",
                stroke_width=2,
                stroke_color="#00ff00",
                background_image=pil_img,
                update_streamlit=True,
                height=min(480, pil_img.height),
                width=min(700, pil_img.width),
                drawing_mode="rect",
                key="cal_canvas",
            )
        else:
            canvas_cal = None

    with colB:
        itch = st.slider("Itch (0–10)", 0, 10, 0)
        pain = st.slider("Pain (0–10)", 0, 10, 0)
        sleep = st.slider("Sleep (hours)", 0.0, 12.0, 7.0, 0.5)
        stress = st.slider("Stress (0–10)", 0, 10, 0)
        triggers = st.multiselect("Triggers", TRIGGER_SUGGESTIONS)
        new_products = st.text_input("New products used (comma-separated)")
        meds_taken = st.text_input("Meds/topicals used today (comma-separated)")
        adherence = st.checkbox("Took meds as planned today", value=True)
        notes = st.text_area("Notes", placeholder="Routine changes, exposures, etc.")

    st.markdown("**Segmentation — choose one:**")
    seg_mode = st.radio("", ["K-Means (auto)", "GrabCut from drawn box", "U-Net (if available)"], horizontal=True)

    rect_box = None
    if pil_img is not None and seg_mode == "GrabCut from drawn box":
        st.caption("Draw a rectangle around the lesion below (GrabCut).")
        canvas_seg = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=2,
            stroke_color="#ff0066",
            background_image=pil_img,
            update_streamlit=True,
            height=min(480, pil_img.height),
            width=min(700, pil_img.width),
            drawing_mode="rect",
            key="seg_canvas",
        )
        if canvas_seg and canvas_seg.json_data and len(canvas_seg.json_data["objects"])>0:
            obj = canvas_seg.json_data["objects"][0]
            rect_box = (obj["left"], obj["top"], obj["width"], obj["height"])

    if st.button("💾 Analyze & Save record", type="primary", disabled=(lesion_choice is None or pil_img is None)):
        if lesion_choice is None:
            st.warning("Select or create a lesion first.")
        elif pil_img is None:
            st.warning("Provide an image.")
        else:
            # Determine condition
            row = existing[existing["id"]==lesion_choice]
            cond = row.iloc[0]["condition"] if not row.empty else CONDITIONS[0]

            img_bgr = pil_to_bgr(pil_img)
            img_bgr = ensure_three_channels(img_bgr)

            # Color calibration (if rectangle drawn in calibration canvas)
            if canvas_cal and canvas_cal.json_data and len(canvas_cal.json_data["objects"])>0:
                o = canvas_cal.json_data["objects"][0]
                roi_rect = (o["left"], o["top"], o["width"], o["height"])
                img_bgr = white_balance_neutral_roi(img_bgr, roi_rect)

            cm_per_px = estimate_scale_cm_per_px(img_bgr, use_aruco=use_aruco, marker_side_cm=marker_cm)

            # Segmentation
            if seg_mode == "K-Means (auto)":
                mask = segment_kmeans(img_bgr, condition=cond)
            elif seg_mode == "GrabCut from drawn box":
                if rect_box is None:
                    st.warning("Draw a rectangle for GrabCut.")
                    mask = np.zeros(img_bgr.shape[:2], np.uint8)
                else:
                    mask = segment_grabcut_rect(img_bgr, rect_box)
            else:
                m = segment_unet(img_bgr)
                mask = m if m is not None else segment_kmeans(img_bgr, condition=cond)

            metrics = compute_metrics(img_bgr, mask, cm_per_px=cm_per_px)

            # Save image
            ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"lesion{lesion_choice}_{ts}.jpg"
            out_path = IMAGES_DIR / fname
            cv2.imwrite(str(out_path), img_bgr)

            # Persist
            insert_record(
                lesion_id=lesion_choice,
                ts=dt.datetime.now().isoformat(timespec="seconds"),
                img_path=str(out_path),
                itch=itch, pain=pain, sleep=sleep, stress=stress,
                triggers=";".join(triggers), new_products=new_products, meds_taken=meds_taken,
                adherence=adherence, notes=notes, metrics=metrics
            )

            st.success("Record saved.")
            overlay = overlay_mask(img_bgr, mask)
            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Overlay preview", use_container_width=True)
            with st.expander("Computed metrics"):
                st.json(metrics)

# ---------------------------
# TRENDS TAB
# ---------------------------
with tab_trends:
    st.subheader("Per-lesion trends & med-change markers")
    lesions_df = list_lesions()
    if lesions_df.empty:
        st.info("No lesions yet. Create one in the Log Entry tab.")
    else:
        lesion_map = {f"{r['label']} [{r['condition']}] (#{r['id']})": int(r["id"]) for _, r in lesions_df.iterrows()}
        sel = st.selectbox("Choose lesion", list(lesion_map.keys()))
        lesion_id = lesion_map[sel]
        hist = lesion_history(lesion_id)
        if hist.empty:
            st.info("No records yet for this lesion.")
        else:
            # medication change markers (compare meds_taken string changes)
            meds_changed_at = []
            last_meds = None
            for i, r in hist.iterrows():
                meds = (r.get("meds_taken") or "").strip().lower()
                if i == 0:
                    last_meds = meds
                else:
                    if meds != last_meds:
                        meds_changed_at.append(r["ts"])
                        last_meds = meds

            def plot_metric(metric, title):
                if metric in hist and hist[metric].notna().any():
                    fig = px.line(hist, x="ts", y=metric, markers=True, title=title)
                    for t in meds_changed_at:
                        fig.add_vline(x=t, line_width=1, line_dash="dash", line_color="orange")
                    return fig
                return None

            cols = st.columns(3)
            figs = [
                ("area_cm2", "Area (cm²)", cols[0]),
                ("redness", "Redness (R/G)", cols[1]),
                ("border_irreg", "Border irregularity (perim² / 4πA)", cols[2]),
                ("asymmetry", "Asymmetry", cols[0]),
                ("depig_deltaE", "ΔE (lesion vs bg)", cols[1]),
            ]
            for m, title, c in figs:
                f = plot_metric(m, title)
                if f is not None:
                    with c: st.plotly_chart(f, use_container_width=True)

            st.markdown("#### Recent photos")
            gcols = st.columns(5)
            recent = hist.tail(5)
            for i, (_, row) in enumerate(recent.iterrows()):
                try:
                    img = cv2.imread(row["img_path"])
                    if img is not None:
                        gcols[i % 5].image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                                           caption=str(row["ts"]).split(".")[0], use_container_width=True)
                except Exception:
                    pass

# ---------------------------
# MED SCHEDULE TAB
# ---------------------------
with tab_meds:
    st.subheader("Medication schedule (per lesion)")
    lesions_df = list_lesions()
    if lesions_df.empty:
        st.info("No lesions yet.")
    else:
        lesion_map = {f"{r['label']} [{r['condition']}] (#{r['id']})": int(r["id"]) for _, r in lesions_df.iterrows()}
        sel = st.selectbox("Choose lesion to edit schedule", list(lesion_map.keys()), key="medsel")
        lesion_id = lesion_map[sel]

        existing_sched = med_schedule_for(lesion_id)
        if not existing_sched.empty:
            st.dataframe(existing_sched.drop(columns=["lesion_id","id"]), use_container_width=True, hide_index=True)

        st.markdown("**Add schedule row**")
        with st.form("add_med_row", clear_on_submit=True):
            name = st.text_input("Name (e.g., triamcinolone 0.1%)")
            dose = st.text_input("Dose/frequency (e.g., thin layer BID)")
            c1, c2, c3 = st.columns(3)
            with c1: morning = st.checkbox("Morning", value=True)
            with c2: afternoon = st.checkbox("Afternoon", value=False)
            with c3: evening = st.checkbox("Evening", value=True)
            mnotes = st.text_input("Notes")
            submitted = st.form_submit_button("➕ Add")
        if submitted and name:
            upsert_med_schedule(lesion_id, name, dose, morning, afternoon, evening, mnotes)
            st.success("Schedule row added. Reopen this tab to refresh.")

# ---------------------------
# EXPORT TAB
# ---------------------------
with tab_export:
    st.subheader("Export CSV or PDF")
    lesions_df = list_lesions()
    if lesions_df.empty:
        st.info("No data yet.")
    else:
        lesion_map = {f"{r['label']} [{r['condition']}] (#{r['id']})": int(r["id"]) for _, r in lesions_df.iterrows()}
        sel = st.selectbox("Choose lesion to export", list(lesion_map.keys()), key="expsel")
        lesion_id = lesion_map[sel]
        hist = lesion_history(lesion_id)
        if hist.empty:
            st.info("No records for this lesion.")
        else:
            with sqlite3.connect(DB_PATH) as con:
                full = pd.read_sql_query("""
                    SELECT r.*, l.label, l.condition
                    FROM records r LEFT JOIN lesions l ON l.id = r.lesion_id
                    WHERE r.lesion_id=? ORDER BY r.ts ASC
                """, con, params=(lesion_id,))
            st.dataframe(full.drop(columns=["img_path"]), use_container_width=True, hide_index=True)

            # CSV export
            csv_bytes = full.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download CSV", data=csv_bytes, file_name=f"skintrack_lesion{lesion_id}.csv", mime="text/csv")

            # PDF summary
            lesions_df2 = list_lesions()
            lesion_row = lesions_df2[lesions_df2["id"]==lesion_id].iloc[0]
            med_df = med_schedule_for(lesion_id)
            pdf_path = DATA_DIR / f"skintrack_summary_lesion{lesion_id}.pdf"
            if st.button("📄 Generate PDF summary"):
                export_pdf_summary(lesion_row, hist, med_df, pdf_path)
                with open(pdf_path, "rb") as f:
                    st.download_button("⬇️ Download PDF", data=f.read(), file_name=pdf_path.name, mime="application/pdf")

# ---------------------------
# SIMULATION TAB
# ---------------------------
with tab_sim:
    st.subheader("What-if simulation (informational)")
    lesions_df = list_lesions()
    if lesions_df.empty:
        st.info("Create a lesion first.")
    else:
        lesion_map = {f"{r['label']} [{r['condition']}] (#{r['id']})": int(r["id"]) for _, r in lesions_df.iterrows()}
        sel = st.selectbox("Choose lesion for baseline", list(lesion_map.keys()), key="selsel")
        lesion_id = lesion_map[sel]
        hist = lesion_history(lesion_id)
        start_area = float(hist["area_cm2"].dropna().iloc[-1]) if (not hist.empty and hist["area_cm2"].notna().any()) else 2.0

        days = st.slider("Days to simulate", 7, 90, 30, 1)
        base_decay = st.slider("Natural healing rate (per day)", 0.0, 0.05, 0.01, 0.001)
        med_potency = st.slider("Medication potency (per day)", 0.0, 0.1, 0.02, 0.001)
        adherence = st.slider("Adherence (0–1)", 0.0, 1.0, 1.0, 0.05)
        stress_level = st.slider("Stress level (0–10)", 0, 10, 3, 1)
        sleep_hours = st.slider("Sleep (hours/night)", 0.0, 12.0, 7.0, 0.5)
        trigger_load = st.slider("Trigger load (count)", 0, 10, 0, 1)

        y = simulate_outcomes(days, start_area, base_decay, med_potency, adherence, stress_level, sleep_hours, trigger_load)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(1, days+1)), y=y, mode="lines+markers", name="Area (cm²)"))
        fig.update_layout(title="Simulated area trajectory", xaxis_title="Day", yaxis_title="Area (cm²)")
        st.plotly_chart(fig, use_container_width=True)

        st.caption("This simple model is for experimentation only. Real treatment decisions require a clinician.")
