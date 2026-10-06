import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="منظومة تنظيم المستمسكات A4", layout="wide", page_icon="🪪")

A4_WIDTH = 2480
A4_HEIGHT = 3508

if 'a4_items' not in st.session_state:
    st.session_state.a4_items = []

# تهيئة إحداثيات النقاط الأربع
if 'pts_dict' not in st.session_state:
    st.session_state.pts_dict = {
        'p1': [50, 50],
        'p2': [500, 50],
        'p3': [500, 350],
        'p4': [50, 350]
    }
if 'active_point' not in st.session_state:
    st.session_state.active_point = '1'

st.markdown("""
<div dir="rtl" style="text-align: center;">
    <h2>🪪 المنظومة الشاملة لضبط زوايا المستمسكات بالماوس والمنزلقات</h2>
    <p>انقر بالفأرة مباشرة على زوايا المستمسك أو استخدم المنزلقات، ثم رتّب مستمسكاتك داخل ورقة A4 للطباعة.</p>
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

tab1, tab2 = st.tabs(["1️⃣ تعديل وقص زوايا المستمسك", "2️⃣ تنسيق ورقة الطباعة A4 (مستندات متعددة)"])

with tab1:
    up_file = st.file_uploader("ارفع صورة المستمسك (جواز، هوية، بطاقة سكن):", type=['jpg', 'jpeg', 'png'])

    if up_file is not None:
        file_bytes = np.asarray(bytearray(up_file.read()), dtype=np.uint8)
        img_raw = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        h_orig, w_orig = img_raw.shape[:2]

        # مقياس عرض الصورة على الشاشة لتسهيل النقر
        disp_w = 680
        scale = disp_w / float(w_orig)
        disp_h = int(h_orig * scale)

        st.sidebar.markdown("### 🖱️ التحديد بالماوس")
        st.session_state.active_point = st.sidebar.radio(
            "اختر النقطة التي تريد تحريكها بالنقر بالماوس:",
            ["1 (أعلى يسار)", "2 (أعلى يمين)", "3 (أسفل يمين)", "4 (أسفل يسار)"]
        )

        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🎛️ لوحة الضبط الدقيق (منزلقات)")

        # استرجاع إحداثيات النقاط
        p1 = st.session_state.pts_dict['p1']
        p2 = st.session_state.pts_dict['p2']
        p3 = st.session_state.pts_dict['p3']
        p4 = st.session_state.pts_dict['p4']

        c1, c2 = st.sidebar.columns(2)
        with c1:
            p1[0] = st.slider("🔴 1- أعلى يسار (X)", 0, w_orig, int(p1[0]), step=2)
            p1[1] = st.slider("🔴 1- أعلى يسار (Y)", 0, h_orig, int(p1[1]), step=2)
            p4[0] = st.slider("🔴 4- أسفل يسار (X)", 0, w_orig, int(p4[0]), step=2)
            p4[1] = st.slider("🔴 4- أسفل يسار (Y)", 0, h_orig, int(p4[1]), step=2)
        with c2:
            p2[0] = st.slider("🔴 2- أعلى يمين (X)", 0, w_orig, int(p2[0]), step=2)
            p2[1] = st.slider("🔴 2- أعلى يمين (Y)", 0, h_orig, int(p2[1]), step=2)
            p3[0] = st.slider("🔴 3- أسفل يمين (X)", 0, w_orig, int(p3[0]), step=2)
            p3[1] = st.slider("🔴 3- أسفل يمين (Y)", 0, h_orig, int(p3[1]), step=2)

        pts = np.array([p1, p2, p3, p4], dtype="float32")

        # رسم المعاينة الحية
        preview = cv2.resize(img_raw, (disp_w, disp_h))
        pts_disp = (pts * scale).astype(np.int32)

        # رسم الخطوط الخضراء (1->2, 2->3, 3->4, 4->1)
        cv2.line(preview, tuple(pts_disp[0]), tuple(pts_disp[1]), (0, 255, 0), 3)
        cv2.line(preview, tuple(pts_disp[1]), tuple(pts_disp[2]), (0, 255, 0), 3)
        cv2.line(preview, tuple(pts_disp[2]), tuple(pts_disp[3]), (0, 255, 0), 3)
        cv2.line(preview, tuple(pts_disp[3]), tuple(pts_disp[0]), (0, 255, 0), 3)

        for idx, p in enumerate(pts_disp):
            cv2.circle(preview, (p[0], p[1]), 12, (0, 0, 255), -1)
            cv2.circle(preview, (p[0], p[1]), 15, (255, 255, 255), 2)
            cv2.putText(preview, str(idx + 1), (p[0] + 12, p[1] + 10), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 0), 2)

        col_view1, col_view2 = st.columns([1.2, 1])

        with col_view1:
            st.subheader("1. انقر بالماوس على زاوية المستمسك:")
            st.caption(f"النقطة المحددة حالياً للنقل بالماوس هي: **{st.session_state.active_point}**")

            # عرض الصورة التفاعلية واستقبال النقر بالماوس
            preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            pil_view = Image.fromarray(preview_rgb)
            coords = streamlit_image_coordinates(pil_view, key="click_crop")

            if coords is not None:
                new_x = int(coords["x"] / scale)
                new_y = int(coords["y"] / scale)
                
                # تحديث النقطة المختارة فور النقر
                active = st.session_state.active_point
                if "1" in active and (abs(p1[0] - new_x) > 3 or abs(p1[1] - new_y) > 3):
                    st.session_state.pts_dict['p1'] = [new_x, new_y]
                    st.rerun()
                elif "2" in active and (abs(p2[0] - new_x) > 3 or abs(p2[1] - new_y) > 3):
                    st.session_state.pts_dict['p2'] = [new_x, new_y]
                    st.rerun()
                elif "3" in active and (abs(p3[0] - new_x) > 3 or abs(p3[1] - new_y) > 3):
                    st.session_state.pts_dict['p3'] = [new_x, new_y]
                    st.rerun()
                elif "4" in active and (abs(p4[0] - new_x) > 3 or abs(p4[1] - new_y) > 3):
                    st.session_state.pts_dict['p4'] = [new_x, new_y]
                    st.rerun()

        with col_view2:
            st.subheader("2. النتيجة المستوية بعد الاستعدال")
            warped = warp_perspective(img_raw, pts)

            filter_mode = st.radio("نمط تصفية الألوان:", ["ألوان محسنة واضحة", "أبيض وأسود سكنر", "تدرج رمادي"], horizontal=True)
            result = enhance_doc(warped, filter_mode)

            if filter_mode in ["أبيض وأسود سكنر", "تدرج رمادي"]:
                final_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
            else:
                final_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

            st.image(final_rgb, caption="المستمسك بعد القص والتسوية", use_container_width=True)

            doc_title = st.text_input("اسم هذا المستمسك:", value="مستمسك رسمي")
            if st.button("➕ حفظ وإضافة المستمسك إلى ورقة A4", type="primary", use_container_width=True):
                new_y = 6 + (len(st.session_state.a4_items) * 28)
                st.session_state.a4_items.append({
                    "title": doc_title,
                    "img": final_rgb,
                    "x": 10,
                    "y": min(new_y, 70),
                    "scale": 90
                })
                st.success(f"تمت إضافة ({doc_title}) بنجاح! انتقل لتبويب ورقة A4.")

with tab2:
    st.markdown("### 🖨️ تخصيص، ترتيب، وتحميل ورقة الـ A4")

    if len(st.session_state.a4_items) == 0:
        st.info("ورقة الـ A4 فارغة حالياً. قم بقص المستمسك من التبويب الأول ثم اضغط 'حفظ وإضافة المستمسك إلى ورقة A4'.")
    else:
        ctrl_col, canvas_col = st.columns([1.1, 1.4])

        with ctrl_col:
            st.subheader("📍 التحكم الحر بمواقع وأحجام المستمسكات")
            
            for i, item in enumerate(st.session_state.a4_items):
                with st.expander(f"📌 {item['title']}", expanded=True):
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

                    item['x'] = st.slider("أفقي X (يسار ⟷ يمين) %", 0, 85, int(item['x']), key=f"x_pos_{i}")
                    item['y'] = st.slider("عمودي Y (أعلى ⟵⟶ أسفل) %", 0, 85, int(item['y']), key=f"y_pos_{i}")
                    item['scale'] = st.slider("الحجم والتكبير %", 20, 180, int(item['scale']), key=f"scale_pos_{i}")

                    if st.button(f"🗑️ حذف {item['title']}", key=f"del_btn_{i}"):
                        st.session_state.a4_items.pop(i)
                        st.rerun()

            if st.button("🧹 مسح جميع المستمسكات من الورقة"):
                st.session_state.a4_items = []
                st.rerun()

            st.markdown("---")
            st.subheader("💾 خيارات التصدير والضغط")
            file_format = st.selectbox("اختر صيغة الملف:", ["PDF جاهز للطباعة", "صورة JPG", "صورة PNG عالية الدقة"])
            compression = st.slider("مستوى ضغط الملف والجودة (%):", 15, 100, 85)

        sheet_a4 = Image.new("RGB", (A4_WIDTH, A4_HEIGHT), (255, 255, 255))

        for item in st.session_state.a4_items:
            doc_pil = Image.fromarray(item['img'])
            target_w = int(A4_WIDTH * 0.44 * (item['scale'] / 100.0))
            ratio = doc_pil.height / float(doc_pil.width)
            target_h = int(target_w * ratio)

            doc_resized = doc_pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
            x_coord = int(A4_WIDTH * (item['x'] / 100.0))
            y_coord = int(A4_HEIGHT * (item['y'] / 100.0))

            sheet_a4.paste(doc_resized, (x_coord, y_coord))

        with canvas_col:
            st.subheader("📄 معاينة ورقة A4 النهائية")
            st.image(sheet_a4, caption="ورقة A4 جاهزة للطباعة فوراً", use_container_width=True)

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
            st.info(f"📊 حجم الملف: **{size_kb:.1f} كيلوبايت** ({size_kb/1024:.2f} ميجابايت)")

            st.download_button(
                label=f"📥 تحميل ورقة المستمسكات بصيغة ({ext.upper()})",
                data=buffer.getvalue(),
                file_name=f"documents_A4.{ext}",
                mime=m_type,
                type="primary",
                use_container_width=True
            )
