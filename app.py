import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="مساعد معالجة المستمسكات", layout="wide", page_icon="📄")

st.markdown("""
<div dir="rtl" style="text-align: center;">
    <h2>📄 أداة فحص وتعديل المستمسكات</h2>
    <p>ارفع صورة المستمسك، وحدد طريقة القص والمعالجة المطلوبة</p>
</div>
""", unsafe_allow_html=True)

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def auto_detect_and_warp(image):
    orig = image.copy()
    ratio = image.shape[0] / 600.0
    h = 600
    w = int(image.shape[1] * (600.0 / image.shape[0]))
    resized = cv2.resize(image, (w, h))

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edged, kernel, iterations=1)

    cnts, _ = cv2.findContours(dilated.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(c) > (w * h * 0.1):
            return four_point_transform(orig, approx.reshape(4, 2) * ratio), True

    return orig, False

def enhance_image(img, mode="color"):
    if mode == "gray":
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif mode == "scanner":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 11)
    else:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

uploaded_file = st.file_uploader("اختر صورة المستمسك", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.sidebar.header("⚙️ خيارات المعالجة")
    method = st.sidebar.radio("طريقة التعديل:", ["تعديل وقص آلي للزوايا", "قص يدوي بالمنزلقات (دقيق)"])
    output_mode = st.sidebar.radio("نمط الوضوح:", ["ألوان واضحة", "مستند سكنر (أبيض وأسود)", "تدرج رمادي"])
    mode_map = {"ألوان واضحة": "color", "مستند سكنر (أبيض وأسود)": "scanner", "تدرج رمادي": "gray"}

    h, w = img.shape[:2]

    if method == "تعديل وقص آلي للزوايا":
        processed_img, detected = auto_detect_and_warp(img)
        if not detected:
            st.warning("⚠️ لم يتم تمييز حواف المستمسك بدقة بسبب الخلفية. يمكنك الانتقال إلى 'قص يدوي بالمنزلقات' من القائمة الجانبية.")
    else:
        st.sidebar.subheader("حدد أطراف القص (%):")
        top = st.sidebar.slider("قص من الأعلى", 0, 40, 5)
        bottom = st.sidebar.slider("قص من الأسفل", 0, 40, 5)
        left = st.sidebar.slider("قص من اليسار", 0, 40, 5)
        right = st.sidebar.slider("قص من اليمين", 0, 40, 5)

        y1 = int(h * (top / 100.0))
        y2 = int(h * (1.0 - bottom / 100.0))
        x1 = int(w * (left / 100.0))
        x2 = int(w * (1.0 - right / 100.0))

        if y2 > y1 and x2 > x1:
            processed_img = img[y1:y2, x1:x2]
        else:
            processed_img = img

    result = enhance_image(processed_img, mode_map[output_mode])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("الصورة الأصلية")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

    with col2:
        st.subheader("النتيجة المستوية والمعدلة")
        if output_mode in ["مستند سكنر (أبيض وأسود)", "تدرج رمادي"]:
            st.image(result, use_container_width=True)
        else:
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_container_width=True)

        is_success, buffer = cv2.imencode(".png", result)
        st.download_button(
            label="💾 تحميل المستمسك المعدل",
            data=io.BytesIO(buffer),
            file_name="adjusted_document.png",
            mime="image/png"
        )
