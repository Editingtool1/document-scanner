import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="مساعد معالجة المستمسكات", layout="wide", page_icon="📄")

st.markdown("""
<div dir="rtl" style="text-align: center;">
    <h2>📄 أداة المسح الضوئي وتعديل المستمسكات</h2>
    <p>قم برفع صورة الهوية أو المستمسك لتعديل الزوايا تلقائياً وتحسين الجودة</p>
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

def scan_document(image):
    orig = image.copy()
    ratio = image.shape[0] / 500.0
    h = 500
    w = int(image.shape[1] * (500.0 / image.shape[0]))
    resized = cv2.resize(image, (w, h))

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is not None:
        warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    else:
        warped = orig

    return warped

def enhance_image(img, mode="color"):
    if mode == "gray":
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif mode == "scanner":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
    else:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

uploaded_file = st.file_uploader("اختر صورة المستمسك أو قم بسحبها هنا", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("الصورة الأصلية")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

    with st.spinner("جاري تعديل الزوايا والمنظور..."):
        warped = scan_document(img)

    with col2:
        st.subheader("النتيجة المستوية")
        mode = st.radio("نمط الإخراج:", ["ألوان محسنة", "أبيض وأسود سكنر", "تدرج رمادي"], horizontal=True)
        
        mode_map = {"ألوان محسنة": "color", "أبيض وأسود سكنر": "scanner", "تدرج رمادي": "gray"}
        result = enhance_image(warped, mode_map[mode])
        
        if mode == "أبيض وأسود سكنر" or mode == "تدرج رمادي":
            res_disp = result
        else:
            res_disp = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            
        st.image(res_disp, use_container_width=True)

        is_success, buffer = cv2.imencode(".png", result)
        st.download_button(
            label="💾 تحميل الصورة المعدلة",
            data=io.BytesIO(buffer),
            file_name="scanned_document.png",
            mime="image/png"
        )
