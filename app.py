import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="معالج المستمسكات الاحترافي", layout="wide", page_icon="🪪")

st.markdown("""
<div dir="rtl" style="text-align: center;">
    <h2>🪪 أداة ضبط واستعدال زوايا المستمسكات</h2>
    <p>انقر بالفأرة مباشرة على <b>زوايا البطاقة الأربع</b> على الصورة الأصلية بالترتيب لتسويتها تلقائياً.</p>
</div>
""", unsafe_allow_html=True)

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # أعلى يسار
    rect[2] = pts[np.argmax(s)] # أسفل يمين
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # أعلى يمين
    rect[3] = pts[np.argmax(diff)] # أسفل يسار
    return rect

def warp_perspective_points(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # حساب الأبعاد الحقيقية بعد التسوية
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # أبعاد البطاقة المستوية
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def enhance_image(img, mode="color"):
    if mode == "gray":
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif mode == "scanner":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
    else:
        # تحسين ذكي للوضوح والتباين
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

# تهيئة النقاط في الذاكرة
if 'points' not in st.session_state:
    st.session_state.points = []

uploaded_file = st.file_uploader("قم برفع صورة المستمسك (JPG أو PNG)", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # قراءة الصورة
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    h_orig, w_orig = img_bgr.shape[:2]

    # تصغير حجم العرض لتسهيل النقر وتطابق الإحداثيات
    display_width = 650
    scale = display_width / float(w_orig)
    display_height = int(h_orig * scale)
    img_display = cv2.resize(img_bgr, (display_width, display_height))

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. انقر على الزوايا الأربع")
        
        # رسم النقاط التي تم النقر عليها
        img_marked = img_display.copy()
        for idx, pt in enumerate(st.session_state.points):
            cv2.circle(img_marked, (int(pt[0]), int(pt[1])), 8, (0, 0, 255), -1)
            cv2.putText(img_marked, str(idx + 1), (int(pt[0]) + 10, int(pt[1]) + 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # تحويل لـ RGB للعرض
        img_marked_rgb = cv2.cvtColor(img_marked, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_marked_rgb)

        # مكون التقاط إحداثيات النقر
        value = streamlit_image_coordinates(pil_img, key="doc_image")

        if value is not None:
            new_point = (value["x"], value["y"])
            if len(st.session_state.points) < 4:
                if not st.session_state.points or new_point != st.session_state.points[-1]:
                    st.session_state.points.append(new_point)
                    st.rerun()

        if st.button("🔄 إعادة تعيين النقاط"):
            st.session_state.points = []
            st.rerun()

        st.caption(f"النقاط المحددة: {len(st.session_state.points)} من 4")

    with col2:
        st.subheader("2. النتيجة المستوية والمعدلة")
        
        if len(st.session_state.points) == 4:
            # إعادة تحويل الإحداثيات للأبعاد الأصلية
            pts_orig = np.array(st.session_state.points, dtype="float32") / scale
            warped = warp_perspective_points(img_bgr, pts_orig)

            mode = st.radio("نمط الألوان:", ["ألوان أصلية محسنة", "أبيض وأسود سكنر", "رمادي"], horizontal=True)
            mode_map = {"ألوان أصلية محسنة": "color", "أبيض وأسود سكنر": "scanner", "رمادي": "gray"}
            
            result = enhance_image(warped, mode_map[mode])

            if mode in ["أبيض وأسود سكنر", "رمادي"]:
                st.image(result, caption="المستمسك بعد الاستعدال والقص", use_container_width=True)
            else:
                st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="المستمسك بعد الاستعدال والقص", use_container_width=True)

            is_success, buffer = cv2.imencode(".png", result)
            st.download_button(
                label="💾 تحميل المستمسك المعدل",
                data=io.BytesIO(buffer),
                file_name="straight_document.png",
                mime="image/png"
            )
        else:
            st.info("👈 اضغط على زوايا البطاقة الأربع في الصورة على اليمين بالترتيب لتظهر النتيجة هنا فوراً.")
else:
    st.session_state.points = []
