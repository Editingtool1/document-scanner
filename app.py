import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import base64
import json
import streamlit.components.v1 as components
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
    <h2>🪪 منظومة تصحيح المستمسكات والتحكم التفاعلي بالماوس على A4</h2>
    <p>قم باستعدال زوايا الهوية، ثم تحكّم بمكان وحجم المستمسكات داخل ورقة A4 بالسحب والتكبير بالماوس مباشرة.</p>
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

tab1, tab2 = st.tabs(["1️⃣ تعديل وقص زوايا المستمسك", "2️⃣ ورقة A4 التفاعلية (سحب وتحجيم بالماوس)"])

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
            "اختر النقطة لنقلها فور النقر على الصورة:",
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

            doc_title = st.text_input("اسم المستمسك:", value="مستمسك رسمي")
            if st.button("➕ إضافة هذا المستمسك إلى ورقة A4 التفاعلية", type="primary", use_container_width=True):
                # تحويل الصورة إلى base64 لإدراجها في كانفاس الـ A4
                _, buf_img = cv2.imencode('.png', cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR))
                b64_str = base64.b64encode(buf_img).decode()

                # إضافة المستمسك بموضع افتراضي متدرج
                st.session_state.a4_items.append({
                    "id": len(st.session_state.a4_items) + 1,
                    "title": doc_title,
                    "img_b64": b64_str,
                    "img_np": final_rgb,
                    "x": 60,
                    "y": 60 + (len(st.session_state.a4_items) * 160),
                    "scale": 0.4
                })
                st.success(f"تمت إضافة '{doc_title}'! انتقل الآن لتبويب ورقة A4 التفاعلية.")

# ----------------- التبويب الثاني -----------------
with tab2:
    st.markdown("### 🖨️ ورقة A4 التفاعلية: حرّك وغيّر حجم المستمسكات بالماوس")

    if len(st.session_state.a4_items) == 0:
        st.info("ورقة الـ A4 فارغة حالياً. قم بقص مستمسك من التبويب الأول واضغط 'إضافة هذا المستمسك إلى ورقة A4'.")
    else:
        col_canvas_edit, col_export = st.columns([1.5, 1])

        with col_canvas_edit:
            st.subheader("📄 لوحة التحكم التفاعلية على A4")
            st.caption("💡 **طريقة الاستخدام:** انقر على أي مستمسك بالماوس لتحريكه بحرية، أو اسحب المربعات الزرقاء في أركانه لتكبيره وتصغيره.")

            # عرض شاشة A4 تفاعلية بنسبة طول إلى عرض الورقة الحقيقية
            canvas_w = 600
            canvas_h = int(canvas_w * (A4_HEIGHT / float(A4_WIDTH)))

            # تجهيز بيانات المستمسكات
            items_json = json.dumps([
                {"id": it["id"], "src": f"data:image/png;base64,{it['img_b64']}", "x": it["x"], "y": it["y"], "scale": it["scale"]}
                for it in st.session_state.a4_items
            ])

            fabric_html = f"""
            <script src="https://cdnjs.cloudflare.com/ajax/libs/fabric.js/5.3.1/fabric.min.js"></script>
            <div style="direction: ltr; text-align: center;">
                <canvas id="a4Canvas" width="{canvas_w}" height="{canvas_h}" style="border: 2px solid #222; box-shadow: 0 4px 15px rgba(0,0,0,0.2); background: white;"></canvas>
                <div style="margin-top: 12px;">
                    <button onclick="saveA4Positions()" style="background: #1976d2; color: white; border: none; padding: 10px 20px; border-radius: 6px; font-weight: bold; cursor: pointer; font-size: 14px;">
                        💾 حفظ التعديلات والأماكن الجديدة
                    </button>
                    <span id="saveStatus" style="color: green; font-weight: bold; margin-left: 10px; display: none;">✅ تم الحفظ بنجاح!</span>
                </div>
            </div>

            <script>
                const canvas = new fabric.Canvas('a4Canvas');
                const itemsData = {items_json};

                itemsData.forEach((item) => {{
                    fabric.Image.fromURL(item.src, function(img) {{
                        img.set({{
                            left: item.x,
                            top: item.y,
                            scaleX: item.scale,
                            scaleY: item.scale,
                            cornerColor: '#007bff',
                            cornerSize: 12,
                            transparentCorners: false,
                            borderColor: '#28a745',
                            cornerStyle: 'circle'
                        }});
                        img.itemId = item.id;
                        canvas.add(img);
                    }});
                }});

                function saveA4Positions() {{
                    let savedData = [];
                    canvas.getObjects().forEach((obj) => {{
                        savedData.push({{
                            id: obj.itemId,
                            x: Math.round(obj.left),
                            y: Math.round(obj.top),
                            scale: parseFloat(obj.scaleX.toFixed(3))
                        }});
                    }});

                    const jsonStr = JSON.stringify(savedData);
                    navigator.clipboard.writeText(jsonStr).then(() => {{
                        const st = document.getElementById('saveStatus');
                        st.style.display = 'inline';
                        setTimeout(() => {{ st.style.display = 'none'; }}, 3000);
                    }});
                }}
            </script>
            """
            components.html(fabric_html, height=canvas_h + 70)

            # خانة استلام التعديلات لحفظها في خادم التصدير
            saved_coords = st.text_input("كود حفظ الأماكن (اضغط زر الحفظ الأزرق أعلاه ثم الصقه هنا لتحديث ملف الطباعة):", "")
            if saved_coords:
                try:
                    updates = json.loads(saved_coords)
                    for up in updates:
                        for it in st.session_state.a4_items:
                            if it["id"] == up["id"]:
                                it["x"] = up["x"]
                                it["y"] = up["y"]
                                it["scale"] = up["scale"]
                    st.success("تم تحديث مواقع المستمسكات وأحجامها في ملف الطباعة النهائي!")
                except:
                    pass

        with col_export:
            st.subheader("💾 قائمة المستمسكات والتصدير")

            for i, it in enumerate(st.session_state.a4_items):
                col_i1, col_i2 = st.columns([3, 1])
                with col_i1:
                    st.write(f"📄 **{it['title']}**")
                with col_i2:
                    if st.button("حذف", key=f"del_a4_{i}"):
                        st.session_state.a4_items.pop(i)
                        st.rerun()

            if st.button("🧹 تفريغ الورقة بالكامل"):
                st.session_state.a4_items = []
                st.rerun()

            st.markdown("---")
            st.subheader("🖨️ خيارات التحميل والطباعة")
            out_fmt = st.selectbox("صيغة التصدير:", ["PDF جاهز للطباعة", "صورة JPG", "صورة PNG عالية الجودة"])
            comp_level = st.slider("مستوى ضغط الحجم وجودة الملف (%):", 20, 100, 85)

            # توليد ورقة A4 الحقيقية بناءً على الأماكن المحددة بالماوس
            scale_factor = A4_WIDTH / float(canvas_w)
            sheet_final = Image.new("RGB", (A4_WIDTH, A4_HEIGHT), (255, 255, 255))

            for it in st.session_state.a4_items:
                doc_img = Image.fromarray(it["img_np"])
                # حساب الحجم والموقع بناءً على ما حركه المستخدم بالماوس
                target_w = int(doc_img.width * it["scale"] * scale_factor)
                target_h = int(doc_img.height * it["scale"] * scale_factor)
                
                if target_w > 10 and target_h > 10:
                    resized = doc_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                    posX = int(it["x"] * scale_factor)
                    posY = int(it["y"] * scale_factor)
                    sheet_final.paste(resized, (posX, posY))

            # تصدير الملف
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
