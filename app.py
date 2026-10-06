import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="منظومة تنظيم المستمسكات A4", layout="wide", page_icon="🪪")

# قياس ورقة A4 بدقة 300 DPI
A4_WIDTH = 2480
A4_HEIGHT = 3508

if 'a4_items' not in st.session_state:
    st.session_state.a4_items = []

if 'active_cut' not in st.session_state:
    st.session_state.active_cut = None

st.markdown("""
<div dir="rtl" style="text-align: center;">
    <h2>🪪 المنظومة الشاملة لضبط زوايا المستمسكات وتجهيز ورقة A4</h2>
    <p>قم بضبط زوايا المستمسك عبر الخطوط المرقمة، استعداله كالسكانر، وترتيب مستمسكات متعددة في ورقة A4 للطباعة والحفظ.</p>
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

def warp_perspective(image, pts):
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

def enhance_doc(img, mode):
    if mode == "أبيض وأسود سكنر":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
    elif mode == "تدرج رمادي":
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else: # ألوان محسنة
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

# شريط التبويبات الرئيسي
tab1, tab2 = st.tabs(["1️⃣ تعديل وقص زوايا المستمسك", "2️⃣ تنسيق ورقة الطباعة A4 (مستندات متعددة)"])

# ----------------- التبويب الأول -----------------
with tab1:
    col_upload, col_settings = st.columns([2, 1])
    with col_upload:
        up_file = st.file_uploader("ارفع صورة المستمسك (جواز سفر، بطاقة وطنية، بطاقة سكن):", type=['jpg', 'jpeg', 'png'])

    if up_file is not None:
        file_bytes = np.asarray(bytearray(up_file.read()), dtype=np.uint8)
        img_raw = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        h, w = img_raw.shape[:2]

        st.sidebar.markdown("### 🎛️ لوحة ضبط الزوايا الأربع بدقة")
        st.sidebar.info("حرّك المنزلقات لوضع الأرقام 1، 2، 3، 4 على أركان الهوية الأربعة:")

        # إحداثيات الزوايا
        c1, c2 = st.sidebar.columns(2)
        with c1:
            p1_x = st.slider("🔴 1- أعلى يسار (X)", 0, w, int(w * 0.06), step=5)
            p1_y = st.slider("🔴 1- أعلى يسار (Y)", 0, h, int(h * 0.08), step=5)
            p4_x = st.slider("🔴 4- أسفل يسار (X)", 0, w, int(w * 0.08), step=5)
            p4_y = st.slider("🔴 4- أسفل يسار (Y)", 0, h, int(h * 0.90), step=5)
        with c2:
            p2_x = st.slider("🔴 2- أعلى يمين (X)", 0, w, int(w * 0.92), step=5)
            p2_y = st.slider("🔴 2- أعلى يمين (Y)", 0, h, int(h * 0.06), step=5)
            p3_x = st.slider("🔴 3- أسفل يمين (X)", 0, w, int(w * 0.94), step=5)
            p3_y = st.slider("🔴 3- أسفل يمين (Y)", 0, h, int(h * 0.92), step=5)

        pts = np.array([[p1_x, p1_y], [p2_x, p2_y], [p3_x, p3_y], [p4_x, p4_y]], dtype="float32")

        # رسم المعاينة الحية مع الخطوط والأرقام (المطلب 1)
        preview = img_raw.copy()
        pts_i = pts.astype(np.int32)
        
        # خطوط من 1 إلى 2، 2 إلى 3، 3 إلى 4، 4 إلى 1
        cv2.line(preview, tuple(pts_i[0]), tuple(pts_i[1]), (0, 255, 0), 4) # 1 -> 2
        cv2.line(preview, tuple(pts_i[1]), tuple(pts_i[2]), (0, 255, 0), 4) # 2 -> 3
        cv2.line(preview, tuple(pts_i[2]), tuple(pts_i[3]), (0, 255, 0), 4) # 3 -> 4
        cv2.line(preview, tuple(pts_i[3]), tuple(pts_i[0]), (0, 255, 0), 4) # 4 -> 1

        # رسم الدوائر والأرقام
        for idx, p in enumerate(pts_i):
            cv2.circle(preview, (p[0], p[1]), 18, (0, 0, 255), -1)
            cv2.circle(preview, (p[0], p[1]), 22, (255, 255, 255), 2)
            cv2.putText(preview, str(idx + 1), (p[0] + 18, p[1] + 12), cv2.FONT_HERSHEY_DUPLEX, 1.4, (255, 0, 0), 3)

        col_view1, col_view2 = st.columns(2)
        with col_view1:
            st.subheader("1. معاينة التحديد المباشر")
            st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), caption="الخطوط الخضراء تمثل حدود القص النهائية", use_container_width=True)

        with col_view2:
            st.subheader("2. النتيجة المستوية بعد الاستعدال")
            warped = warp_perspective(img_raw, pts)

            filter_mode = st.radio("نمط تصفية الألوان:", ["ألوان محسنة واضحة", "أبيض وأسود سكنر", "تدرج رمادي"], horizontal=True)
            result = enhance_doc(warped, filter_mode)

            if filter_mode in ["أبيض وأسود سكنر", "تدرج رمادي"]:
                final_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
            else:
                final_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

            st.image(final_rgb, caption="المستمسك مفرود ومستقيم بالكامل", use_container_width=True)

            doc_title = st.text_input("اسم هذا المستمسك:", value="البطاقة الوطنية / الجواز")
            if st.button("➕ حفظ وإضافة المستمسك إلى ورقة A4", type="primary", use_container_width=True):
                # حساب نسبة التموضع الافتراضي
                new_y = 6 + (len(st.session_state.a4_items) * 28)
                st.session_state.a4_items.append({
                    "title": doc_title,
                    "img": final_rgb,
                    "x": 10,
                    "y": min(new_y, 70),
                    "scale": 90
                })
                st.success(f"تمت إضافة ({doc_title}) بنجاح إلى ورقة A4! انتقل إلى التبويب الثاني لترتيب الورقة وتحميلها.")

# ----------------- التبويب الثاني -----------------
with tab2:
    st.markdown("### 🖨️ تخصيص، ترتيب، وتحميل ورقة الـ A4")

    if len(st.session_state.a4_items) == 0:
        st.info("ورقة الـ A4 فارغة حالياً. قم بقص المستمسكات من التبويب الأول واضغط 'إضافة المستمسك إلى ورقة A4'.")
    else:
        ctrl_col, canvas_col = st.columns([1.1, 1.4])

        with ctrl_col:
            st.subheader("📍 التحكم الحر بأماكن وأحجام المستمسكات")
            
            for i, item in enumerate(st.session_state.a4_items):
                with st.expander(f"📌 {item['title']}", expanded=True):
                    # موضع حر (المطلب 3)
                    pos_choice = st.selectbox("الموضع السريع:", ["تحديد حر يدوي", "أعلى اليمين", "أعلى اليسار", "منتصف الصفحة", "أسفل اليمين", "أسفل اليسار"], key=f"quick_{i}")
                    if pos_choice == "أعلى اليمين":
                        item['x'], item['y'] = 52, 6
                    elif pos_choice == "أعلى اليسار":
                        item['x'], item['y'] = 6, 6
                    elif pos_choice == "منتصف الصفحة":
                        item['x'], item['y'] = 28, 35
                    elif pos_choice == "أسفل اليمين":
                        item['x'], item['y'] = 52, 60
                    elif pos_choice == "أسفل اليسار":
                        item['x'], item['y'] = 6, 60

                    item['x'] = st.slider(f"أفقي X (يسار ⟷ يمين) %", 0, 85, int(item['x']), key=f"x_pos_{i}")
                    item['y'] = st.slider(f"عمودي Y (أعلى ⟵⟶ أسفل) %", 0, 85, int(item['y']), key=f"y_pos_{i}")
                    item['scale'] = st.slider(f"الحجم والتكبير %", 20, 180, int(item['scale']), key=f"scale_pos_{i}")

                    if st.button(f"🗑️ حذف {item['title']}", key=f"del_btn_{i}"):
                        st.session_state.a4_items.pop(i)
                        st.rerun()

            if st.button("🧹 مسح جميع المستمسكات من الورقة"):
                st.session_state.a4_items = []
                st.rerun()

            st.markdown("---")
            st.subheader("💾 خيارات التصدير والضغط")
            file_format = st.selectbox("اختر صيغة الملف:", ["PDF جاهز للطباعة", "صورة JPG", "صورة PNG عالية الدقة"])
            compression = st.slider("مستوى ضغط الملف والجودة (%):", 15, 100, 85, help="كلما قل الرقم صغر حجم الملف لتسهيل الإرسال.")

        # بناء ورقة A4 وتجميع كافة المستمسكات (المطلب 2 و 6)
        sheet_a4 = Image.new("RGB", (A4_WIDTH, A4_HEIGHT), (255, 255, 255))

        for item in st.session_state.a4_items:
            doc_pil = Image.fromarray(item['img'])
            # حساب الأبعاد المليمترية
            target_w = int(A4_WIDTH * 0.44 * (item['scale'] / 100.0))
            ratio = doc_pil.height / float(doc_pil.width)
            target_h = int(target_w * ratio)

            doc_resized = doc_pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
            x_coord = int(A4_WIDTH * (item['x'] / 100.0))
            y_coord = int(A4_HEIGHT * (item['y'] / 100.0))

            sheet_a4.paste(doc_resized, (x_coord, y_coord))

        with canvas_col:
            st.subheader("📄 معاينة ورقة A4 النهائية")
            st.image(sheet_a4, caption="ورقة A4 الحقيقية جاهزة للطباعة فوراً", use_container_width=True)

            # معالجة الصيغ والضغط (المطلب 4 و 5)
            buffer = io.BytesIO()
            if file_format == "PDF جاهز للطباعة":
                sheet_a4.save(buffer, format="PDF", resolution=300.0, quality=compression)
                m_type = "application/pdf"
                ext = "pdf"
            elif file_format == "صورة JPG":
                sheet_a4.save(buffer, format="JPEG", quality=compression, optimize=True)
                m_type = "image/jpeg"
                ext = "jpg"
            else:
                sheet_a4.save(buffer, format="PNG", optimize=True)
                m_type = "image/png"
                ext = "png"

            size_kb = len(buffer.getvalue()) / 1024.0
            st.info(f"📊 حجم الملف الناتج: **{size_kb:.1f} كيلوبايت** ({size_kb/1024:.2f} ميجابايت)")

            st.download_button(
                label=f"📥 تحميل ورقة المستمسكات بصيغة ({ext.upper()})",
                data=buffer.getvalue(),
                file_name=f"documents_A4.{ext}",
                mime=m_type,
                type="primary",
                use_container_width=True
            )
