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

if 'active_doc_idx' not in st.session_state:
    st.session_state.active_doc_idx = 0

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
    <p>استعدل زوايا الهوية، ورتّب مستمسكاتك داخل ورقة A4 للطباعة مع معاينة حية ومباشرة.</p>
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
    else:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

# دالة رسم ورقة A4 ومعاينتها بدقة لضمان ظهور الصور فوراً
def draw_a4_preview(items, preview_w=620):
    preview_h = int(preview_w * (A4_HEIGHT / float(A4_WIDTH)))
    canvas = Image.new("RGB", (preview_w, preview_h), (255, 255, 255))
    
    scale_w = preview_w / float(A4_WIDTH)
    scale_h = preview_h / float(A4_HEIGHT)

    for idx, item in enumerate(items):
        doc_img = Image.fromarray(item["img"])
        
        # الأبعاد بالبكسل على ورقة A4 الأصلية (300 DPI)
        target_w = int(A4_WIDTH * 0.44 * (item["scale"] / 100.0))
        ratio = doc_img.height / float(doc_img.width)
        target_h = int(target_w * ratio)

        # تحويل الأبعاد لشاشة المعاينة
        w_disp = max(20, int(target_w * scale_w))
        h_disp = max(20, int(target_h * scale_h))

        x_disp = int((item["x"] / 100.0) * preview_w)
        y_disp = int((item["y"] / 100.0) * preview_h)

        resized_doc = doc_img.resize((w_disp, h_disp), Image.Resampling.LANCZOS)
        canvas.paste(resized_doc, (x_disp, y_disp))

    return canvas

tab1, tab2 = st.tabs(["1️⃣ تعديل وقص زوايا المستمسك", "2️⃣ معاينة وترتيب ورقة A4 للطباعة"])

# ----------------- التبويب الأول -----------------
with tab1:
    up_file = st.file_uploader("ارفع صورة المستمسك (جواز، هوية، بطاقة سكن):", type=['jpg', 'jpeg', 'png'])

    if up_file is not None:
        file_bytes = np.asarray(bytearray(up_file.read()), dtype=np.uint8)
        img_raw = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        h_orig, w_orig = img_raw.shape[:2]

        disp_w = 680
        scale = disp_w / float(w_orig)
        disp_h = int(h_orig * scale)

        st.sidebar.markdown("### 🖱️ تحديد النقطة بالماوس")
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

            doc_title = st.text_input("اسم المستمسك:", value=f"مستمسك {len(st.session_state.a4_items) + 1}")
            if st.button("➕ حفظ وإضافة المستمسك إلى ورقة A4", type="primary", use_container_width=True):
                # موضع متدرج تلقائي
                new_y = 6 + (len(st.session_state.a4_items) * 26)
                st.session_state.a4_items.append({
                    "id": len(st.session_state.a4_items) + 1,
                    "title": doc_title,
                    "img": final_rgb,
                    "x": 8,
                    "y": min(new_y, 70),
                    "scale": 95
                })
                st.session_state.active_doc_idx = len(st.session_state.a4_items) - 1
                st.success(f"تمت إضافة '{doc_title}' بنجاح! انتقل الآن لتبويب ورقة A4.")

# ----------------- التبويب الثاني -----------------
with tab2:
    st.markdown("### 🖨️ معاينة ورقة A4 التفاعلية وتحديد الأماكن بالماوس")

    if len(st.session_state.a4_items) == 0:
        st.info("ورقة الـ A4 فارغة حالياً. قم بقص المستمسك من التبويب الأول ثم اضغط 'حفظ وإضافة المستمسك إلى ورقة A4'.")
    else:
        col_preview, col_controls = st.columns([1.3, 1.1])

        with col_preview:
            st.subheader("📄 ورقة A4 (انقر بالماوس لتحديد موضع المستمسك):")
            st.caption("💡 **انقر في أي مكان على الورقة** لنقل المستمسك المحدد إلى نقطة النقر فوراً[cite: 16].")

            # توليد ورقة A4 الحية لعرضها في التطبيق
            a4_display = draw_a4_preview(st.session_state.a4_items, preview_w=620)

            # التقاط إحداثيات النقر بالماوس فوق ورقة A4
            click_a4 = streamlit_image_coordinates(a4_display, key="a4_board_click")

            if click_a4 is not None and len(st.session_state.a4_items) > 0:
                cur_i = st.session_state.active_doc_idx
                if cur_i < len(st.session_state.a4_items):
                    new_x_pct = int((click_a4["x"] / 620.0) * 100)
                    new_y_pct = int((click_a4["y"] / float(a4_display.height)) * 100)

                    new_x_pct = max(0, min(80, new_x_pct))
                    new_y_pct = max(0, min(85, new_y_pct))

                    if abs(st.session_state.a4_items[cur_i]["x"] - new_x_pct) > 2 or abs(st.session_state.a4_items[cur_i]["y"] - new_y_pct) > 2:
                        st.session_state.a4_items[cur_i]["x"] = new_x_pct
                        st.session_state.a4_items[cur_i]["y"] = new_y_pct
                        st.rerun()

        with col_controls:
            st.subheader("🎯 تحكم بالمستمسك المحدد")
            
            item_names = [f"{i+1}. {it['title']}" for i, it in enumerate(st.session_state.a4_items)]
            sel_name = st.selectbox(
                "اختر المستمسك النشط للتحكم:", 
                item_names, 
                index=min(st.session_state.active_doc_idx, len(item_names)-1)
            )
            st.session_state.active_doc_idx = item_names.index(sel_name)
            curr = st.session_state.a4_items[st.session_state.active_doc_idx]

            st.markdown("#### ⚡ خيارات التموضع والمقاس")
            quick_p = st.selectbox(
                "أماكن جاهزة بنقرة واحدة:",
                ["تحديد حر (انقر على الورقة)", "أعلى اليمين", "أعلى اليسار", "منتصف الورقة", "أسفل اليمين", "أسفل اليسار"],
                key=f"qp_{st.session_state.active_doc_idx}"
            )
            if quick_p == "أعلى اليمين":
                curr['x'], curr['y'] = 52, 6
            elif quick_p == "أعلى اليسار":
                curr['x'], curr['y'] = 6, 6
            elif quick_p == "منتصف الورقة":
                curr['x'], curr['y'] = 26, 36
            elif quick_p == "أسفل اليمين":
                curr['x'], curr['y'] = 52, 62
            elif quick_p == "أسفل اليسار":
                curr['x'], curr['y'] = 6, 62

            curr['scale'] = st.slider("🔍 المقاس والتكبير (%)", 30, 200, int(curr['scale']), step=5)
            curr['x'] = st.slider("↔️ الموضع الأفقي X (%)", 0, 85, int(curr['x']), step=1)
            curr['y'] = st.slider("↕️ الموضع الرأسي Y (%)", 0, 85, int(curr['y']), step=1)

            col_b1, col_b2 = st.columns(2)
            with col_b1:
                if st.button("🗑️ حذف هذا المستمسك", key="del_one"):
                    st.session_state.a4_items.pop(st.session_state.active_doc_idx)
                    st.session_state.active_doc_idx = 0
                    st.rerun()
            with col_b2:
                if st.button("🧹 تفريغ الورقة بالكامل"):
                    st.session_state.a4_items = []
                    st.session_state.active_doc_idx = 0
                    st.rerun()

            st.markdown("---")
            st.subheader("💾 تحميل الملف والطباعة")
            fmt = st.selectbox("صيغة التصدير:", ["PDF جاهز للطباعة", "صورة JPG", "صورة PNG عالية الجودة"])
            quality = st.slider("مستوى ضغط الحجم وجودة الملف (%):", 20, 100, 85)

            # تجهيز ورقة الطباعة بدقة الطباعة الأصلية 300 DPI
            final_sheet = Image.new("RGB", (A4_WIDTH, A4_HEIGHT), (255, 255, 255))
            for it in st.session_state.a4_items:
                doc_im = Image.fromarray(it["img"])
                w_px = int(A4_WIDTH * 0.44 * (it["scale"] / 100.0))
                ar = doc_im.height / float(doc_im.width)
                h_px = int(w_px * ar)

                if w_px > 20 and h_px > 20:
                    resized = doc_im.resize((w_px, h_px), Image.Resampling.LANCZOS)
                    pos_x = int(A4_WIDTH * (it["x"] / 100.0))
                    pos_y = int(A4_HEIGHT * (it["y"] / 100.0))
                    final_sheet.paste(resized, (pos_x, pos_y))

            buf = io.BytesIO()
            if fmt == "PDF جاهز للطباعة":
                final_sheet.save(buf, format="PDF", resolution=300.0, quality=quality)
                mtype = "application/pdf"
                ext = "pdf"
            elif fmt == "صورة JPG":
                final_sheet.save(buf, format="JPEG", quality=quality, optimize=True)
                mtype = "image/jpeg"
                ext = "jpg"
            else:
                final_sheet.save(buf, format="PNG", optimize=True)
                mtype = "image/png"
                ext = "png"

            kb_size = len(buf.getvalue()) / 1024.0
            st.info(f"📊 حجم الملف: **{kb_size:.1f} كيلوبايت**")

            st.download_button(
                label=f"📥 تحميل ورقة A4 بصيغة ({ext.upper()})",
                data=buf.getvalue(),
                file_name=f"documents_A4.{ext}",
                mime=mtype,
                type="primary",
                use_container_width=True
            )
