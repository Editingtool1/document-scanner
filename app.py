import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="منظومة معالجة وتنظيم المستمسكات A4", layout="wide", page_icon="📄")

# أبعاد ورقة A4 بدقة 300 DPI للطباعة القياسية (2480 x 3508 بكسل)
A4_WIDTH = 2480
A4_HEIGHT = 3508

if 'a4_docs' not in st.session_state:
    st.session_state.a4_docs = []

st.markdown("""
<div dir="rtl" style="text-align: center;">
    <h2>🪪 منظومة تصحيح زوايا المستمسكات وتجهيز ورق الطباعة A4</h2>
    <p>قم باستعدال زوايا الهويات، وتحديد أماكنها بدقة داخل ورقة A4، وتصديرها بصيغة PDF أو صور جاهزة للطباعة المباشرة.</p>
</div>
""", unsafe_allow_html=True)

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # 1: أعلى يسار
    rect[2] = pts[np.argmax(s)] # 3: أسفل يمين
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # 2: أعلى يمين
    rect[3] = pts[np.argmax(diff)] # 4: أسفل يسار
    return rect

def warp_perspective_points(image, pts):
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
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

def enhance_image(img, mode="color"):
    if mode == "gray":
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif mode == "scanner":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
    else:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

# تقسيم الشاشة إلى تبويبات
tab1, tab2 = st.tabs(["1️⃣ تعديل زوايا المستمسك وتجهيزه", "2️⃣ تنظيم وطباعة ورقة A4 (مستمسكات متعددة)"])

with tab1:
    uploaded_file = st.file_uploader("ارفع صورة المستمسك (جواز، هوية، بطاقة سكن):", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        h, w = img_bgr.shape[:2]

        st.sidebar.markdown("### 🎯 تحديد الزوايا الأربع")
        
        # النقاط الافتراضية
        col_s1, col_s2 = st.sidebar.columns(2)
        with col_s1:
            p1_x = st.number_input("نقطة 1 (X) أعلى يسار", 0, w, int(w * 0.05), step=10)
            p1_y = st.number_input("نقطة 1 (Y) أعلى يسار", 0, h, int(h * 0.05), step=10)
            p4_x = st.number_input("نقطة 4 (X) أسفل يسار", 0, w, int(w * 0.05), step=10)
            p4_y = st.number_input("نقطة 4 (Y) أسفل يسار", 0, h, int(h * 0.95), step=10)
        with col_s2:
            p2_x = st.number_input("نقطة 2 (X) أعلى يمين", 0, w, int(w * 0.95), step=10)
            p2_y = st.number_input("نقطة 2 (Y) أعلى يمين", 0, h, int(h * 0.05), step=10)
            p3_x = st.number_input("نقطة 3 (X) أسفل يمين", 0, w, int(w * 0.95), step=10)
            p3_y = st.number_input("نقطة 3 (Y) أسفل يمين", 0, h, int(h * 0.95), step=10)

        pts = np.array([
            [p1_x, p1_y],
            [p2_x, p2_y],
            [p3_x, p3_y],
            [p4_x, p4_y]
        ], dtype="float32")

        # رسم الخطوط والأرقام التوضيحية (1 إلى 2، 2 إلى 3، 3 إلى 4، 4 إلى 1)
        preview = img_bgr.copy()
        pts_int = pts.astype(np.int32)
        
        # خطوط ملونة وواضحة
        cv2.line(preview, tuple(pts_int[0]), tuple(pts_int[1]), (0, 255, 0), 4) # 1 -> 2
        cv2.line(preview, tuple(pts_int[1]), tuple(pts_int[2]), (0, 255, 0), 4) # 2 -> 3
        cv2.line(preview, tuple(pts_int[2]), tuple(pts_int[3]), (0, 255, 0), 4) # 3 -> 4
        cv2.line(preview, tuple(pts_int[3]), tuple(pts_int[0]), (0, 255, 0), 4) # 4 -> 1

        for idx, p in enumerate(pts_int):
            cv2.circle(preview, (p[0], p[1]), 14, (0, 0, 255), -1)
            cv2.putText(preview, str(idx + 1), (p[0] + 15, p[1] + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.subheader("معاينة التحديد والخطوط")
            st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), use_container_width=True)

        with col_img2:
            st.subheader("النتيجة المستوية المستخرجة")
            warped = warp_perspective_points(img_bgr, pts)
            
            mode = st.radio("نمط الألوان:", ["ألوان محسنة", "أبيض وأسود سكنر", "تدرج رمادي"], horizontal=True)
            mode_map = {"ألوان محسنة": "color", "أبيض وأسود سكنر": "scanner", "تدرج رمادي": "gray"}
            result = enhance_image(warped, mode_map[mode])

            if mode in ["أبيض وأسود سكنر", "تدرج رمادي"]:
                st.image(result, use_container_width=True)
                res_to_save = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
            else:
                res_to_save = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                st.image(res_to_save, use_container_width=True)

            doc_name = st.text_input("اسم هذا المستمسك (مثلاً: البطاقة الوطنية - وجه):", "مستمسك 1")
            if st.button("➕ إضافة هذا المستمسك إلى ورقة A4", type="primary"):
                st.session_state.a4_docs.append({
                    "name": doc_name,
                    "img": res_to_save,
                    "x_pos": 10,   # نسبة مئوية من العرض
                    "y_pos": 10 + len(st.session_state.a4_docs) * 25,  # تموضع تلقائي متدرج
                    "scale": 100   # الحجم بالنسبة المئوية
                })
                st.success(f"تمت إضافة '{doc_name}' إلى ورقة A4 بنجاح! انتقل للتبويب الثاني.")

with tab2:
    st.markdown("### 🖨️ تجهيز وتنسيق ورقة الطباعة A4")
    
    if len(st.session_state.a4_docs) == 0:
        st.info("لم تقم بإضافة أي مستمسك بعد. قم بقص المستمسكات من التبويب الأول ثم اضغط 'إضافة إلى ورقة A4'.")
    else:
        col_ctrl, col_canvas = st.columns([1, 1.3])
        
        with col_ctrl:
            st.subheader("📐 تحكم بمواقع وأحجام المستمسكات")
            
            for i, doc in enumerate(st.session_state.a4_docs):
                with st.expander(f"⚙️ إعدادات: {doc['name']}", expanded=True):
                    doc['x_pos'] = st.slider(f"الموقع الأفقي X (يسار - يمين) - {doc['name']}", 0, 80, doc['x_pos'], key=f"x_{i}")
                    doc['y_pos'] = st.slider(f"الموقع الرأسي Y (أعلى - أسفل) - {doc['name']}", 0, 85, doc['y_pos'], key=f"y_{i}")
                    doc['scale'] = st.slider(f"تكبير / تصغير الحجم (%) - {doc['name']}", 20, 200, doc['scale'], key=f"scale_{i}")
                    
                    if st.button(f"🗑️ حذف {doc['name']}", key=f"del_{i}"):
                        st.session_state.a4_docs.pop(i)
                        st.rerun()

            if st.button("🧹 مسح كافة المستمسكات من الورقة"):
                st.session_state.a4_docs = []
                st.rerun()

            st.markdown("---")
            st.subheader("💾 إعدادات التحميل والضغط")
            export_format = st.selectbox("صيغة التصدير المطلوبة:", ["PDF جاهز للطباعة", "صورة JPG", "صورة PNG عالية الدقة"])
            compress_quality = st.slider("مستوى الجودة وضغط الحجم (%)", 20, 100, 85, help="تقليل الرقم يقلل حجم الملف جداً لمشاركته عبر الواتساب أو رفعه للمواقع الحكومية.")

        # توليد ورقة الـ A4 بيضاء نقية
        a4_sheet = Image.new("RGB", (A4_WIDTH, A4_HEIGHT), (255, 255, 255))
        
        # لصق كل مستمسك في موقعه وحجمه
        for doc in st.session_state.a4_docs:
            doc_img = Image.fromarray(doc['img'])
            # إعادة تحجيم المستمسك بناءً على مقياس A4
            base_w = int(A4_WIDTH * 0.45 * (doc['scale'] / 100.0))
            aspect_ratio = doc_img.height / float(doc_img.width)
            base_h = int(base_w * aspect_ratio)
            
            resized_doc = doc_img.resize((base_w, base_h), Image.Resampling.LANCZOS)
            
            x_px = int(A4_WIDTH * (doc['x_pos'] / 100.0))
            y_px = int(A4_HEIGHT * (doc['y_pos'] / 100.0))
            
            a4_sheet.paste(resized_doc, (x_px, y_px))

        with col_canvas:
            st.subheader("📄 معاينة ورقة A4 النهائية")
            st.image(a4_sheet, caption="ورقة A4 بمقاس الطباعة الحقيقي", use_container_width=True)

            buf = io.BytesIO()
            if export_format == "PDF جاهز للطباعة":
                a4_sheet.save(buf, format="PDF", resolution=300.0, quality=compress_quality)
                mime_type = "application/pdf"
                file_ext = "pdf"
            elif export_format == "صورة JPG":
                a4_sheet.save(buf, format="JPEG", quality=compress_quality, optimize=True)
                mime_type = "image/jpeg"
                file_ext = "jpg"
            else:
                a4_sheet.save(buf, format="PNG", optimize=True)
                mime_type = "image/png"
                file_ext = "png"

            file_size_kb = len(buf.getvalue()) / 1024.0
            st.caption(f"حجم الملف المتوقع: **{file_size_kb:.1f} كيلوبايت** ({file_size_kb/1024:.2f} ميجابايت)")

            st.download_button(
                label=f"📥 تحميل ورقة A4 بصيغة ({file_ext.upper()})",
                data=buf.getvalue(),
                file_name=f"documents_A4.{file_ext}",
                mime=mime_type,
                type="primary",
                use_container_width=True
            )
