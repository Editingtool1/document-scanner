import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import base64
import json
import streamlit.components.v1 as components

st.set_page_config(page_title="منظومة معالجة وتنظيم المستمسكات A4", layout="wide", page_icon="🪪")

A4_WIDTH = 2480
A4_HEIGHT = 3508

if 'a4_docs' not in st.session_state:
    st.session_state.a4_docs = []
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None

st.markdown("""
<div dir="rtl" style="text-align: center;">
    <h2>🪪 منظومة تصحيح زوايا المستمسكات المباشرة</h2>
    <p>اسحب الدوائر الأربع الحمراء بالماوس وضعها على أركان المستمسك، ثم اضغط زر القص المباشر أدناها.</p>
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

tab1, tab2 = st.tabs(["1️⃣ سحب الزوايا وقص المستمسك", "2️⃣ ورقة الطباعة A4"])

with tab1:
    uploaded_file = st.file_uploader("ارفع صورة المستمسك (جواز، بطاقة موحدة، هوية، سكن):", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        h_orig, w_orig = img_bgr.shape[:2]

        _, buffer = cv2.imencode('.jpg', img_bgr)
        img_base64 = base64.b64encode(buffer).decode()

        canvas_width = 650
        canvas_height = int(h_orig * (canvas_width / float(w_orig)))

        col_crop, col_res = st.columns([1.2, 1])

        with col_crop:
            st.subheader("🎯 اسحب الزوايا الأربع بالماوس:")

            # مكوّن JavaScript تفاعلي يرسل الإحداثيات عند النقر أو الضغط
            canvas_html = f"""
            <div style="direction: ltr; text-align: center;">
                <canvas id="docCanvas" width="{canvas_width}" height="{canvas_height}" style="border:2px solid #555; cursor:crosshair; border-radius: 8px;"></canvas>
            </div>

            <script>
                const canvas = document.getElementById('docCanvas');
                const ctx = canvas.getContext('2d');
                const img = new Image();
                img.src = "data:image/jpeg;base64,{img_base64}";

                let points = [
                    {{x: {int(canvas_width * 0.08)}, y: {int(canvas_height * 0.08)}, label: "1"}},
                    {{x: {int(canvas_width * 0.92)}, y: {int(canvas_height * 0.08)}, label: "2"}},
                    {{x: {int(canvas_width * 0.92)}, y: {int(canvas_height * 0.92)}, label: "3"}},
                    {{x: {int(canvas_width * 0.08)}, y: {int(canvas_height * 0.92)}, label: "4"}}
                ];

                let draggedPoint = null;
                const radius = 14;

                function draw() {{
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

                    // رسم الخطوط الخضراء
                    ctx.strokeStyle = '#00FF00';
                    ctx.lineWidth = 3;
                    ctx.beginPath();
                    ctx.moveTo(points[0].x, points[0].y);
                    for (let i = 1; i < points.length; i++) {{
                        ctx.lineTo(points[i].x, points[i].y);
                    }}
                    ctx.closePath();
                    ctx.stroke();

                    // رسم النقاط والأرقام
                    points.forEach((p, idx) => {{
                        ctx.fillStyle = '#FF0000';
                        ctx.beginPath();
                        ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
                        ctx.fill();
                        ctx.strokeStyle = '#FFFFFF';
                        ctx.lineWidth = 2;
                        ctx.stroke();

                        ctx.fillStyle = '#FFFFFF';
                        ctx.font = 'bold 15px Arial';
                        ctx.fillText(p.label, p.x - 5, p.y + 5);
                    }});
                }}

                img.onload = () => {{ draw(); sendPoints(); }};

                function getMousePos(evt) {{
                    const rect = canvas.getBoundingClientRect();
                    return {{
                        x: evt.clientX - rect.left,
                        y: evt.clientY - rect.top
                    }};
                }}

                canvas.addEventListener('mousedown', (e) => {{
                    const pos = getMousePos(e);
                    points.forEach((p) => {{
                        const dist = Math.hypot(p.x - pos.x, p.y - pos.y);
                        if (dist < radius + 8) {{
                            draggedPoint = p;
                        }}
                    }});
                }});

                canvas.addEventListener('mousemove', (e) => {{
                    if (draggedPoint) {{
                        const pos = getMousePos(e);
                        draggedPoint.x = Math.max(0, Math.min(canvas.width, pos.x));
                        draggedPoint.y = Math.max(0, Math.min(canvas.height, pos.y));
                        draw();
                    }}
                }});

                function sendPoints() {{
                    const scaleX = {w_orig} / {canvas_width};
                    const scaleY = {h_orig} / {canvas_height};
                    const rawPts = points.map(p => [Math.round(p.x * scaleX), Math.round(p.y * scaleY)]);
                    window.parent.postMessage({{
                        type: 'streamlit:setComponentValue',
                        value: JSON.stringify(rawPts)
                    }}, '*');
                }}

                canvas.addEventListener('mouseup', () => {{ 
                    if(draggedPoint) {{
                        sendPoints();
                        draggedPoint = null;
                    }}
                }});
                canvas.addEventListener('mouseleave', () => {{ 
                    if(draggedPoint) {{
                        sendPoints();
                        draggedPoint = null;
                    }}
                }});
            </script>
            """
            received_data = components.html(canvas_html, height=canvas_height + 30)

            # تحكم يدوي إضافي بالمنزلقات السريعة في حال رغبت بضبط دقيق
            with st.expander("🛠️ تعديل إحداثيات دقيق (اختياري)"):
                st.caption("إذا أردت تعديل رقمي مباشر للزوايا:")
                p1_x = st.slider("1. أعلى يسار X", 0, w_orig, int(w_orig * 0.08))
                p1_y = st.slider("1. أعلى يسار Y", 0, h_orig, int(h_orig * 0.08))
                p2_x = st.slider("2. أعلى يمين X", 0, w_orig, int(w_orig * 0.92))
                p2_y = st.slider("2. أعلى يمين Y", 0, h_orig, int(h_orig * 0.08))
                p3_x = st.slider("3. أسفل يمين X", 0, w_orig, int(w_orig * 0.92))
                p3_y = st.slider("3. أسفل يمين Y", 0, h_orig, int(h_orig * 0.92))
                p4_x = st.slider("4. أسفل يسار X", 0, w_orig, int(w_orig * 0.08))
                p4_y = st.slider("4. أسفل يسار Y", 0, h_orig, int(h_orig * 0.92))
                manual_pts = [[p1_x, p1_y], [p2_x, p2_y], [p3_x, p3_y], [p4_x, p4_y]]

            btn_crop = st.button("✂️ تطبيق القص وتعديل المنظور الآن", type="primary", use_container_width=True)
            if btn_crop:
                st.session_state.processed_image = warp_perspective_points(img_bgr, np.array(manual_pts, dtype="float32"))

        with col_res:
            st.subheader("النتيجة المعدلة:")
            
            # في حال لم يتم الضغط بعد، نقوم بعمل قص أولي تلقائي
            if st.session_state.processed_image is None:
                default_pts = np.array([[w_orig*0.08, h_orig*0.08], [w_orig*0.92, h_orig*0.08], [w_orig*0.92, h_orig*0.92], [w_orig*0.08, h_orig*0.92]], dtype="float32")
                st.session_state.processed_image = warp_perspective_points(img_bgr, default_pts)

            mode = st.radio("نمط الألوان والوضوح:", ["ألوان محسنة", "أبيض وأسود سكنر", "تدرج رمادي"], horizontal=True)
            mode_map = {"ألوان محسنة": "color", "أبيض وأسود سكنر": "scanner", "تدرج رمادي": "gray"}
            
            enhanced = enhance_image(st.session_state.processed_image, mode_map[mode])

            if mode in ["أبيض وأسود سكنر", "تدرج رمادي"]:
                res_to_save = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
            else:
                res_to_save = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

            st.image(res_to_save, caption="المستمسك بعد الاستعدال والتسوية", use_container_width=True)

            # زر تنزيل مباشر للصورة وحدها
            is_success, buffer = cv2.imencode(".png", cv2.cvtColor(res_to_save, cv2.COLOR_RGB2BGR))
            st.download_button(
                label="💾 تحميل هذه الصورة فقط",
                data=io.BytesIO(buffer),
                file_name="cropped_id.png",
                mime="image/png",
                use_container_width=True
            )

            st.markdown("---")
            doc_name = st.text_input("اسم المستمسك لورقة A4:", "مستمسك 1")
            if st.button("➕ إضافة هذا المستمسك لورقة الـ A4", type="primary", use_container_width=True):
                st.session_state.a4_docs.append({
                    "name": doc_name,
                    "img": res_to_save,
                    "x_pos": 10,
                    "y_pos": 10 + len(st.session_state.a4_docs) * 22,
                    "scale": 100
                })
                st.success(f"تمت إضافة '{doc_name}' إلى ورقة A4! انتقل إلى تبويب 'ورقة الطباعة A4'.")

with tab2:
    st.markdown("### 🖨️ تنظيم وتحميل ورقة الطباعة A4")
    
    if len(st.session_state.a4_docs) == 0:
        st.info("لم تقم بإضافة أي مستمسك بعد. قم بقص المستمسك في التبويب الأول ثم اضغط 'إضافة هذا المستمسك لورقة الـ A4'.")
    else:
        col_ctrl, col_canvas = st.columns([1, 1.3])
        
        with col_ctrl:
            st.subheader("📐 تحكم بمواقع وأحجام المستمسكات")
            for i, doc in enumerate(st.session_state.a4_docs):
                with st.expander(f"⚙️ تموضع: {doc['name']}", expanded=True):
                    doc['x_pos'] = st.slider(f"أفقي X (يسار - يمين) - {doc['name']}", 0, 80, doc['x_pos'], key=f"x_{i}")
                    doc['y_pos'] = st.slider(f"رأسي Y (أعلى - أسفل) - {doc['name']}", 0, 85, doc['y_pos'], key=f"y_{i}")
                    doc['scale'] = st.slider(f"الحجم (%) - {doc['name']}", 20, 200, doc['scale'], key=f"scale_{i}")
                    if st.button(f"🗑️ حذف {doc['name']}", key=f"del_{i}"):
                        st.session_state.a4_docs.pop(i)
                        st.rerun()

            if st.button("🧹 تفريغ الورقة بالكامل"):
                st.session_state.a4_docs = []
                st.rerun()

            st.markdown("---")
            st.subheader("💾 خيارات الحفظ والضغط")
            export_format = st.selectbox("صيغة الملف:", ["PDF جاهز للطباعة", "صورة JPG", "صورة PNG عالية الدقة"])
            compress_quality = st.slider("جودة وضغط الحجم (%)", 20, 100, 85)

        a4_sheet = Image.new("RGB", (A4_WIDTH, A4_HEIGHT), (255, 255, 255))
        for doc in st.session_state.a4_docs:
            doc_img = Image.fromarray(doc['img'])
            base_w = int(A4_WIDTH * 0.45 * (doc['scale'] / 100.0))
            aspect_ratio = doc_img.height / float(doc_img.width)
            base_h = int(base_w * aspect_ratio)
            resized_doc = doc_img.resize((base_w, base_h), Image.Resampling.LANCZOS)
            x_px = int(A4_WIDTH * (doc['x_pos'] / 100.0))
            y_px = int(A4_HEIGHT * (doc['y_pos'] / 100.0))
            a4_sheet.paste(resized_doc, (x_px, y_px))

        with col_canvas:
            st.subheader("معاينة ورقة A4")
            st.image(a4_sheet, use_container_width=True)

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
            st.caption(f"حجم الملف: **{file_size_kb:.1f} KB** ({file_size_kb/1024:.2f} MB)")

            st.download_button(
                label=f"📥 تحميل ورقة A4 بصيغة ({file_ext.upper()})",
                data=buf.getvalue(),
                file_name=f"documents_A4.{file_ext}",
                mime=mime_type,
                type="primary",
                use_container_width=True
            )
