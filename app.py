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

if 'pts_dict' not in st.session_state:
    st.session_state.pts_dict = {
        'p1': [50, 50],
        'p2': [500, 50],
        'p3': [500, 350],
        'p4': [50, 350]
    }
if 'active_point' not in st.session_state:
    st.session_state.active_point = '1 (أعلى يسار)'

st.markdown("""
<div dir="rtl" style="text-align: center;">
    <h2>🪪 منظومة تصحيح زوايا المستمسكات وتجهيز ورقة A4</h2>
    <p>استعدل زوايا الهوية، ورتب المستمسكات في ورقة A4 مع معاينة فورية ومباشرة داخل التطبيق.</p>
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
    else:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

tab1, tab2 = st.tabs(["1️⃣ تعديل وقص زوايا المستمسك", "2️⃣ معاينة وترتيب ورقة A4 للطباعة"])

with tab1:
    up_file = st.file_uploader("ارفع صورة المستمسك (جواز، بطاقة موحدة، هوية، سكن):", type=['jpg', 'jpeg', 'png'])

    if up_file is not None:
        file_bytes = np.asarray(bytearray(up_file.read()), dtype=np.uint8)
        img_raw = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        h_orig, w_orig = img_raw.shape[:2]

        disp_w = 680
        scale = disp_w / float(w_orig)
        disp_h = int(h_orig * scale)

        st.sidebar.markdown("### 🖱️ التحديد بالماوس")
        st.session_state.active_point = st.sidebar.radio(
            "اختر النقطة لنقلها فور النقر بالماوس:",
            ["1 (أعلى يسار)", "2 (أعلى يمين)", "3 (أسفل يمين)", "4 (أسفل يسار)"]
        )

        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🎛️ الضبط الدقيق (منزلقات)")

        p1 = st.session_state.pts_dict['p1']
        p2 = st.session_state.pts_dict['p2']
        p3 = st.session_state.pts_dict['p3']
        p4 = st.session_state.pts_dict['p4']

        c1, c2 = st.sidebar.columns(2)
        with c1:
            p1[0] = st.slider("🔴 1- X", 0, w_orig, int(p1[0]), step=2)
            p1[1] = st.slider("🔴 1- Y", 0, h_orig, int(p1[1]), step=2)
            p4[0] = st.slider("🔴 4- X", 0, w_orig, int(p4[0]), step=2)
            p4[1] = st.slider("🔴 4- Y", 0, h_orig, int(p4[1]), step=2)
        with c2:
            p2[0] = st.slider("🔴 2- X", 0, w_orig, int(p2[0]), step=2)
            p2[1] = st.slider("🔴 2- Y", 0, h_orig, int(p2[1]), step=2)
            p3[0] = st.slider("🔴 3- X", 0, w_orig, int(p3[0]), step=2)
            p3[1] = st.slider("🔴 3- Y", 0, h_orig, int(p3[1]), step=2)

        pts = np.array([p1, p2, p3, p4], dtype="float32")

        preview = cv2.resize(img_raw, (disp_w, disp_h))
        pts_disp = (pts * scale).astype(np.int32)

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
            st.subheader("1. انقر بالماوس على ركن البطاقة:")
            preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            coords = streamlit_image_coordinates(Image.fromarray(preview_rgb), key="click_crop_a4")

            if coords is not None:
                new_x = int(coords["x"] / scale)
                new_y = int(coords["y"] / scale)
                act = st.session_state.active_point
                if "1" in act and (abs(p1[0] - new_x) > 3 or abs(p1[1] - new_y) > 3):
                    st.session_state.pts_dict['p1'] = [new_x, new_y]
                    st.rerun()
                elif "2" in act and (abs(p2[0] - new_x) > 3 or abs(p2[1] - new_y) > 3):
                    st.session_state.pts_dict['p2'] = [new_x, new_y]
                    st.rerun()
                elif "3" in act and (abs(p3[0] - new_x) > 3 or abs(p3[1] - new_y) > 3):
                    st.session_state.pts_dict['p3'] = [new_x, new_y]
                    st.rerun()
                elif "4" in act and (abs(p4[0] - new_x) > 3 or abs(p4[1] - new_y) > 3):
                    st.session_state.pts_dict['p4'] = [new_x, new_y]
                    st.rerun()

        with col_view2:
            st.subheader("2. المستمسك بعد الاستعدال")
            warped = warp_perspective(img_raw, pts)

            filter_mode = st.radio("نمط التصفية:", ["ألوان محسنة واضحة", "أبيض وأسود سكنر", "تدرج رمادي"], horizontal=True)
            result = enhance_doc(warped, filter_mode)

            if filter_mode in ["أبيض وأسود سكنر", "تدرج رمادي"]:
                final_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
            else:
                final_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

            st.image(final_rgb, caption="صورة المستمسك مفرودة كالسكانر", use_container_width=True)

            doc_title = st.text_input("اسم المستمسك:", value=f"مستمسك {len(st.session_state.a4_items)+1}")
            if st.button("➕ إضافة المستمسك إلى ورقة A4", type="primary", use_container_width=True):
                new_y = 6 + (len(st.session_state.a4_items) * 26)
                st.session_state.a4_items.append({
                    "id": len(st.session_state.a4_items) + 1,
                    "title": doc_title,
                    "img_np": final_rgb,
                    "x": 8,
                    "y": min(new_y, 70),
                    "scale": 90
                })
                st.success(f"تمت إضافة '{doc_title}' بنجاح! انتقل لتبويب ورقة A4.")

with tab2:
    st.markdown("### 🖨️ معاينة وترتيب ورقة الطباعة A4 المباشرة")

    if len(st.session_state.a4_items) == 0:
        st.info("ورقة الـ A4 فارغة حالياً. قم بقص مستمسك من التبويب الأول واضغط 'إضافة المستمسك إلى ورقة A4'.")
    else:
        col_ctrl, col_canvas = st.columns([1, 1.4])

        with col_ctrl:
            st.subheader("📍 التحكم بمواقع وأحجام المستمسكات")
            
            for i, it in enumerate(st.session_state.a4_items):
                with st.expander(f"📌 إعدادات: {it['title']}", expanded=True):
                    quick_pos = st.selectbox(
                        "موضع سريع:",
                        ["تحديد يدوي", "أعلى اليمين", "أعلى اليسار", "منتصف الصفحة", "أسفل اليمين", "أسفل اليسار"],
                        key=f"qpos_{i}"
                    )
                    if quick_pos == "أعلى اليمين":
                        it['x'], it['y'] = 52, 6
                    elif quick_pos == "أعلى اليسار":
                        it['x'], it['y'] = 6, 6
                    elif quick_pos == "منتصف الصفحة":
                        it['x'], it['y'] = 26, 36
                    elif quick_pos == "أسفل اليمين":
                        it['x'], it['y'] = 52, 62
                    elif quick_pos == "أسفل اليسار":
                        it['x'], it['y'] = 6, 62

                    it['x'] = st.slider("أفقي X (يسار ⟷ يمين) %", 0, 85, int(it['x']), key=f"x_s_{i}")
                    it['y'] = st.slider("عمودي Y (أعلى ⟵⟶ أسفل) %", 0, 85, int(it['y']), key=f"y_s_{i}")
                    it['scale'] = st.slider("الحجم والتكبير %", 30, 180, int(it['scale']), key=f"scale_s_{i}")

                    if st.button(f"🗑️ حذف {it['title']}", key=f"del_item_{i}"):
                        st.session_state.a4_items.pop(i)
                        st.rerun()

            if st.button("🧹 تفريغ جميع المستمسكات"):
                st.session_state.a4_items = []
                st.rerun()

            st.markdown("---")
            st.subheader("💾 خيارات التصدير والضغط")
            out_fmt = st.selectbox("صيغة الملف:", ["PDF جاهز للطباعة", "صورة JPG", "صورة PNG عالية الجودة"])
            comp_level = st.slider("مستوى ضغط الحجم وجودة الملف (%):", 20, 100, 85)

        # توليد ورقة A4 مباشرة ورسم كل المستمسكات فوقها
        sheet_final = Image.new("RGB", (A4_WIDTH, A4_HEIGHT), (255, 255, 255))

        for it in st.session_state.a4_items:
            doc_img = Image.fromarray(it["img_np"])
            target_w = int(A4_WIDTH * 0.44 * (it["scale"] / 100.0))
            ratio = doc_img.height / float(doc_img.width)
            target_h = int(target_w * ratio)

            if target_w > 20 and target_h > 20:
                resized = doc_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                posX = int(A4_WIDTH * (it["x"] / 100.0))
                posY = int(A4_HEIGHT * (it["y"] / 100.0))
                sheet_final.paste(resized, (posX, posY))

        with col_canvas:
            st.subheader("📄 معاينة ورقة A4 المباشرة داخل التطبيق")
            st.image(sheet_final, caption="معاينة حية ومطابقة تماماً لورقة الطباعة", use_container_width=True)

            buf_out = io.BytesIO()
            if out_fmt == "PDF جاهز للطباعة":
                sheet_final.save(buf_out, format="PDF", resolution=300.0, quality=comp_level)
                m_type = "application/pdf"
                f_ext = "pdf"
            elif out_fmt == "صورة JPG":
                sheet_final.save(buf_out, format="JPEG", quality=comp_level, optimize=True)
                m_type = "image/jpeg"
                f_ext = "jpg"
            else:
                sheet_final.save(buf_out, format="PNG", optimize=True)
                m_type = "image/png"
                f_ext = "png"

            file_kb = len(buf_out.getvalue()) / 1024.0
            st.info(f"📊 حجم الملف: **{file_kb:.1f} كيلوبايت**")

            st.download_button(
                label=f"📥 تحميل ورقة A4 بصيغة ({f_ext.upper()})",
                data=buf_out.getvalue(),
                file_name=f"documents_A4.{f_ext}",
                mime=m_type,
                type="primary",
                use_container_width=True
            )
