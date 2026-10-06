import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import io
from streamlit_image_coordinates import streamlit_image_coordinates


# ============================================================
# إعداد الصفحة
# ============================================================

st.set_page_config(
    page_title="منظومة تصحيح المستمسكات A4",
    layout="wide",
    page_icon="🪪"
)


# ============================================================
# إعدادات A4
# 300 DPI
# ============================================================

A4_PORTRAIT = (2480, 3508)
A4_LANDSCAPE = (3508, 2480)


# ============================================================
# Session State
# ============================================================

defaults = {
    "a4_items": [],
    "active_doc_idx": 0,
    "active_point": "1 (أعلى يسار)",
    "uploaded_image": None,
    "image_name": "",
    "pts_dict": {
        "p1": [50, 50],
        "p2": [500, 50],
        "p3": [500, 350],
        "p4": [50, 350],
    },
    "a4_orientation": "عمودي",
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>

html, body, [class*="css"] {
    font-family: Tahoma, Arial, sans-serif;
}

.main-title {
    text-align:center;
    padding:10px;
}

.main-title h1 {
    color:#75182f;
    margin-bottom:5px;
}

.main-title p {
    color:#666;
}

.section-box {
    background:#fafafa;
    border:1px solid #ddd;
    border-radius:12px;
    padding:12px;
    margin-bottom:15px;
}

.step-title {
    font-size:20px;
    font-weight:bold;
    color:#75182f;
    margin-bottom:10px;
}

.small-info {
    font-size:13px;
    color:#666;
}

.document-card {
    border:1px solid #ddd;
    border-radius:10px;
    padding:10px;
    background:white;
    margin-bottom:8px;
}

.success-box {
    padding:12px;
    border-radius:10px;
    background:#eaf8ef;
    border:1px solid #b8e0c5;
}

.warning-box {
    padding:12px;
    border-radius:10px;
    background:#fff8e6;
    border:1px solid #efd58b;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# العنوان
# ============================================================

st.markdown(
    """
<div class="main-title" dir="rtl">

<h1>🪪 منظومة تصحيح المستمسكات وتجهيز A4</h1>

<p>
تحديد الزوايا 1 → 2 → 3 → 4،
تصحيح المنظور،
تحسين الصورة،
وترتيب مستندات متعددة داخل ورقة A4
</p>

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# دوال هندسية
# ============================================================

def order_points(pts):
    """
    ترتيب النقاط:
    1 = أعلى يسار
    2 = أعلى يمين
    3 = أسفل يمين
    4 = أسفل يسار
    """

    pts = np.asarray(pts, dtype="float32")

    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)

    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def warp_perspective(image, pts):
    """
    تصحيح منظور المستند.
    """

    rect = order_points(pts)

    tl, tr, br, bl = rect

    width_a = np.sqrt(
        ((br[0] - bl[0]) ** 2)
        +
        ((br[1] - bl[1]) ** 2)
    )

    width_b = np.sqrt(
        ((tr[0] - tl[0]) ** 2)
        +
        ((tr[1] - tl[1]) ** 2)
    )

    max_width = max(
        int(width_a),
        int(width_b)
    )

    height_a = np.sqrt(
        ((tr[0] - br[0]) ** 2)
        +
        ((tr[1] - br[1]) ** 2)
    )

    height_b = np.sqrt(
        ((tl[0] - bl[0]) ** 2)
        +
        ((tl[1] - bl[1]) ** 2)
    )

    max_height = max(
        int(height_a),
        int(height_b)
    )

    max_width = max(max_width, 50)
    max_height = max(max_height, 50)

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(
        rect,
        dst
    )

    result = cv2.warpPerspective(
        image,
        matrix,
        (max_width, max_height)
    )

    return result


# ============================================================
# تحسين المستند
# ============================================================

def enhance_doc(
    img,
    mode,
    brightness=100,
    contrast=100,
    sharpness=100,
):
    """
    تحسين الصورة بدون تغيير محتوى المستند.
    """

    if len(img.shape) == 2:
        rgb = cv2.cvtColor(
            img,
            cv2.COLOR_GRAY2RGB
        )
    else:
        rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

    pil = Image.fromarray(rgb)

    # السطوع
    pil = ImageEnhance.Brightness(
        pil
    ).enhance(
        brightness / 100.0
    )

    # التباين
    pil = ImageEnhance.Contrast(
        pil
    ).enhance(
        contrast / 100.0
    )

    # الحدة
    pil = ImageEnhance.Sharpness(
        pil
    ).enhance(
        sharpness / 100.0
    )

    rgb = np.array(pil)

    if mode == "أبيض وأسود سكنر":

        gray = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2GRAY
        )

        result = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,
            10,
        )

        return cv2.cvtColor(
            result,
            cv2.COLOR_GRAY2RGB
        )

    elif mode == "تدرج رمادي":

        gray = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2GRAY
        )

        return cv2.cvtColor(
            gray,
            cv2.COLOR_GRAY2RGB
        )

    else:

        return rgb


# ============================================================
# إنشاء ورقة A4 للمعاينة
# ============================================================

def get_a4_size():
    if st.session_state.a4_orientation == "أفقي":
        return A4_LANDSCAPE

    return A4_PORTRAIT


def draw_a4_preview(
    items,
    preview_w=720,
):
    """
    إنشاء معاينة A4.
    """

    a4_width, a4_height = get_a4_size()

    preview_h = int(
        preview_w
        *
        (
            a4_height /
            float(a4_width)
        )
    )

    canvas = Image.new(
        "RGB",
        (
            preview_w,
            preview_h,
        ),
        (255, 255, 255),
    )

    scale_x = (
        preview_w /
        float(a4_width)
    )

    scale_y = (
        preview_h /
        float(a4_height)
    )

    for item in items:

        doc_img = Image.fromarray(
            item["img"]
        ).convert("RGB")

        target_w = int(
            a4_width
            *
            0.44
            *
            (
                item["scale"] /
                100.0
            )
        )

        target_w = max(
            target_w,
            20
        )

        ratio = (
            doc_img.height /
            float(doc_img.width)
        )

        target_h = int(
            target_w * ratio
        )

        target_h = max(
            target_h,
            20
        )

        resized = doc_img.resize(
            (
                target_w,
                target_h,
            ),
            Image.Resampling.LANCZOS,
        )

        # تدوير المستند
        rotation = item.get(
            "rotation",
            0
        )

        if rotation != 0:

            resized = resized.rotate(
                rotation,
                expand=True,
                resample=Image.Resampling.BICUBIC,
                fillcolor="white",
            )

        display_w = max(
            20,
            int(
                resized.width *
                scale_x
            ),
        )

        display_h = max(
            20,
            int(
                resized.height *
                scale_y
            ),
        )

        resized = resized.resize(
            (
                display_w,
                display_h,
            ),
            Image.Resampling.LANCZOS,
        )

        x = int(
            (
                item["x"] /
                100.0
            )
            *
            preview_w
        )

        y = int(
            (
                item["y"] /
                100.0
            )
            *
            preview_h
        )

        canvas.paste(
            resized,
            (
                x,
                y,
            ),
        )

    return canvas


# ============================================================
# التبويبات
# ============================================================

tab1, tab2 = st.tabs(
    [
        "1️⃣ تصحيح المستند",
        "2️⃣ ترتيب ورقة A4",
    ]
)


# ============================================================
# التبويب الأول
# ============================================================

with tab1:

    st.markdown(
        """
<div class="section-box" dir="rtl">

<div class="step-title">
📐 المرحلة الأولى: تحديد زوايا المستند
</div>

<div class="small-info">

حدد زوايا المستند بالترتيب:

<b>1 أعلى يسار</b> →
<b>2 أعلى يمين</b> →
<b>3 أسفل يمين</b> →
<b>4 أسفل يسار</b>

<br><br>

الخطوط والأرقام تظهر فوق الصورة قبل إجراء التصحيح،
حتى تستطيع التأكد من الشكل النهائي.

</div>

</div>
""",
        unsafe_allow_html=True,
    )


    # --------------------------------------------------------
    # رفع الملف
    # --------------------------------------------------------

    up_file = st.file_uploader(
        "📤 ارفع صورة المستمسك",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
    )


    if up_file is not None:

        # قراءة الصورة
        file_bytes = np.asarray(
            bytearray(
                up_file.read()
            ),
            dtype=np.uint8,
        )

        img_raw = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR,
        )

        if img_raw is None:

            st.error(
                "تعذر قراءة الصورة."
            )

            st.stop()


        h_orig, w_orig = (
            img_raw.shape[:2]
        )


        # ----------------------------------------------------
        # إعادة تعيين النقاط عند رفع صورة جديدة
        # ----------------------------------------------------

        current_name = up_file.name

        if (
            st.session_state.image_name
            != current_name
        ):

            st.session_state.image_name = (
                current_name
            )

            st.session_state.pts_dict = {

                "p1": [
                    int(w_orig * 0.08),
                    int(h_orig * 0.08),
                ],

                "p2": [
                    int(w_orig * 0.92),
                    int(h_orig * 0.08),
                ],

                "p3": [
                    int(w_orig * 0.92),
                    int(h_orig * 0.92),
                ],

                "p4": [
                    int(w_orig * 0.08),
                    int(h_orig * 0.92),
                ],
            }


        # ----------------------------------------------------
        # الشريط الجانبي
        # ----------------------------------------------------

        with st.sidebar:

            st.markdown(
                "## 🎛️ التحكم بالمستند"
            )

            st.markdown(
                "---"
            )

            st.markdown(
                "### 🎯 النقطة التي تريد تحريكها"
            )

            st.session_state.active_point = (
                st.radio(
                    "اختر النقطة:",
                    [
                        "1 (أعلى يسار)",
                        "2 (أعلى يمين)",
                        "3 (أسفل يمين)",
                        "4 (أسفل يسار)",
                    ],
                    index=[
                        "1 (أعلى يسار)",
                        "2 (أعلى يمين)",
                        "3 (أسفل يمين)",
                        "4 (أسفل يسار)",
                    ].index(
                        st.session_state.active_point
                    ),
                )
            )


            st.markdown(
                "---"
            )

            st.markdown(
                "### 📍 الضبط الدقيق"
            )

            p1 = st.session_state.pts_dict[
                "p1"
            ]

            p2 = st.session_state.pts_dict[
                "p2"
            ]

            p3 = st.session_state.pts_dict[
                "p3"
            ]

            p4 = st.session_state.pts_dict[
                "p4"
            ]


            c1, c2 = st.columns(2)


            with c1:

                p1[0] = st.slider(
                    "1-X",
                    0,
                    w_orig,
                    int(p1[0]),
                    step=1,
                )

                p1[1] = st.slider(
                    "1-Y",
                    0,
                    h_orig,
                    int(p1[1]),
                    step=1,
                )


                p4[0] = st.slider(
                    "4-X",
                    0,
                    w_orig,
                    int(p4[0]),
                    step=1,
                )

                p4[1] = st.slider(
                    "4-Y",
                    0,
                    h_orig,
                    int(p4[1]),
                    step=1,
                )


            with c2:

                p2[0] = st.slider(
                    "2-X",
                    0,
                    w_orig,
                    int(p2[0]),
                    step=1,
                )

                p2[1] = st.slider(
                    "2-Y",
                    0,
                    h_orig,
                    int(p2[1]),
                    step=1,
                )


                p3[0] = st.slider(
                    "3-X",
                    0,
                    w_orig,
                    int(p3[0]),
                    step=1,
                )

                p3[1] = st.slider(
                    "3-Y",
                    0,
                    h_orig,
                    int(p3[1]),
                    step=1,
                )


        # ----------------------------------------------------
        # النقاط
        # ----------------------------------------------------

        pts = np.array(
            [
                p1,
                p2,
                p3,
                p4,
            ],
            dtype="float32",
        )


        # ----------------------------------------------------
        # صورة المعاينة
        # ----------------------------------------------------

        display_width = 720

        scale = (
            display_width /
            float(w_orig)
        )

        display_height = int(
            h_orig * scale
        )


        preview = cv2.resize(
            img_raw,
            (
                display_width,
                display_height,
            ),
        )


        pts_display = (
            pts * scale
        ).astype(np.int32)


        # ----------------------------------------------------
        # الخطوط
        # ----------------------------------------------------

        line_color = (
            0,
            255,
            0,
        )


        cv2.line(
            preview,
            tuple(pts_display[0]),
            tuple(pts_display[1]),
            line_color,
            3,
        )

        cv2.line(
            preview,
            tuple(pts_display[1]),
            tuple(pts_display[2]),
            line_color,
            3,
        )

        cv2.line(
            preview,
            tuple(pts_display[2]),
            tuple(pts_display[3]),
            line_color,
            3,
        )

        cv2.line(
            preview,
            tuple(pts_display[3]),
            tuple(pts_display[0]),
            line_color,
            3,
        )


        # ----------------------------------------------------
        # الأرقام
        # ----------------------------------------------------

        for idx, point in enumerate(
            pts_display
        ):

            x = int(point[0])
            y = int(point[1])

            cv2.circle(
                preview,
                (
                    x,
                    y,
                ),
                18,
                (
                    0,
                    0,
                    255,
                ),
                -1,
            )

            cv2.circle(
                preview,
                (
                    x,
                    y,
                ),
                21,
                (
                    255,
                    255,
                    255,
                ),
                2,
            )

            cv2.putText(
                preview,
                str(idx + 1),
                (
                    x - 7,
                    y + 7,
                ),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (
                    255,
                    255,
                    255,
                ),
                2,
            )


        # ----------------------------------------------------
        # أعمدة العرض
        # ----------------------------------------------------

        col1, col2 = st.columns(
            [
                1.25,
                1,
            ]
        )


        # ====================================================
        # الصورة الأصلية
        # ====================================================

        with col1:

            st.markdown(
                "### 1️⃣ الصورة الأصلية وتحديد الزوايا"
            )

            st.caption(
                "اختر النقطة من القائمة الجانبية، ثم انقر على مكانها الصحيح في الصورة."
            )


            preview_rgb = cv2.cvtColor(
                preview,
                cv2.COLOR_BGR2RGB,
            )


            coords = (
                streamlit_image_coordinates(
                    Image.fromarray(
                        preview_rgb
                    ),
                    key="document_corner_selector",
                )
            )


            if coords is not None:

                new_x = int(
                    coords["x"] /
                    scale
                )

                new_y = int(
                    coords["y"] /
                    scale
                )


                active = (
                    st.session_state.active_point
                )


                if active.startswith("1"):

                    st.session_state.pts_dict[
                        "p1"
                    ] = [
                        new_x,
                        new_y,
                    ]

                    st.rerun()


                elif active.startswith("2"):

                    st.session_state.pts_dict[
                        "p2"
                    ] = [
                        new_x,
                        new_y,
                    ]

                    st.rerun()


                elif active.startswith("3"):

                    st.session_state.pts_dict[
                        "p3"
                    ] = [
                        new_x,
                        new_y,
                    ]

                    st.rerun()


                elif active.startswith("4"):

                    st.session_state.pts_dict[
                        "p4"
                    ] = [
                        new_x,
                        new_y,
                    ]

                    st.rerun()


        # ====================================================
        # المعاينة بعد التصحيح
        # ====================================================

        with col2:

            st.markdown(
                "### 2️⃣ معاينة النتيجة"
            )


            warped = warp_perspective(
                img_raw,
                pts,
            )


            st.markdown(
                "#### 🎨 تحسين الصورة"
            )


            filter_mode = st.radio(
                "نمط الصورة:",
                [
                    "ألوان محسنة واضحة",
                    "أبيض وأسود سكنر",
                    "تدرج رمادي",
                ],
                horizontal=True,
            )


            brightness = st.slider(
                "☀️ السطوع",
                50,
                150,
                100,
            )


            contrast = st.slider(
                "◐ التباين",
                50,
                180,
                100,
            )


            sharpness = st.slider(
                "🔎 الحدة",
                50,
                200,
                100,
            )


            final_rgb = enhance_doc(
                warped,
                filter_mode,
                brightness,
                contrast,
                sharpness,
            )


            st.image(
                final_rgb,
                caption="المستند بعد تصحيح المنظور",
                use_container_width=True,
            )


            st.markdown(
                '<div class="success-box">'
                "✓ هذه هي النتيجة التي ستضاف إلى A4."
                "</div>",
                unsafe_allow_html=True,
            )


            st.markdown(
                "### 📝 اسم المستند"
            )


            doc_title = st.text_input(
                "اسم المستند:",
                value=(
                    f"مستند "
                    f"{len(st.session_state.a4_items) + 1}"
                ),
            )


            st.markdown(
                "### 📄 إضافة إلى A4"
            )


            add_to_a4 = st.checkbox(
                "إضافة المستند المصحح إلى ورقة A4",
                value=True,
            )


            if st.button(
                "➕ حفظ المستند",
                type="primary",
                use_container_width=True,
            ):

                if add_to_a4:

                    # ترتيب تلقائي أولي
                    count = len(
                        st.session_state.a4_items
                    )

                    positions = [
                        (6, 6),
                        (52, 6),
                        (6, 38),
                        (52, 38),
                        (6, 70),
                        (52, 70),
                    ]

                    pos = positions[
                        min(
                            count,
                            len(positions) - 1,
                        )
                    ]


                    st.session_state.a4_items.append(
                        {
                            "id": count + 1,
                            "title": doc_title,
                            "img": final_rgb.copy(),
                            "x": pos[0],
                            "y": pos[1],
                            "scale": 95,
                            "rotation": 0,
                        }
                    )


                    st.session_state.active_doc_idx = (
                        len(
                            st.session_state.a4_items
                        ) - 1
                    )


                    st.success(
                        f"تمت إضافة '{doc_title}' إلى A4 بنجاح."
                    )

                else:

                    st.success(
                        "تم تجهيز المستند."
                    )


# ============================================================
# التبويب الثاني
# ============================================================

with tab2:

    st.markdown(
        """
<div class="section-box" dir="rtl">

<div class="step-title">
📄 المرحلة الثانية: ترتيب المستندات داخل A4
</div>

<div class="small-info">

يمكنك إضافة عدة مستندات إلى نفس الصفحة،
ثم تحديد مكان كل مستند بدقة،
وتغيير حجمه ودورانه.

</div>

</div>
""",
        unsafe_allow_html=True,
    )


    # --------------------------------------------------------
    # إعداد اتجاه A4
    # --------------------------------------------------------

    st.markdown(
        "### 📐 إعداد الصفحة"
    )


    orientation = st.radio(
        "اتجاه ورقة A4:",
        [
            "عمودي",
            "أفقي",
        ],
        index=(
            0
            if st.session_state.a4_orientation
            == "عمودي"
            else 1
        ),
        horizontal=True,
    )


    if (
        orientation
        != st.session_state.a4_orientation
    ):

        st.session_state.a4_orientation = (
            orientation
        )

        st.rerun()


    # --------------------------------------------------------
    # إذا لا توجد مستندات
    # --------------------------------------------------------

    if not st.session_state.a4_items:

        st.info(
            "ورقة A4 فارغة. ارجع إلى التبويب الأول وأضف مستنداً."
        )

        st.stop()


    # --------------------------------------------------------
    # اختيار المستند
    # --------------------------------------------------------

    item_names = [
        f"{i + 1}. {item['title']}"
        for i, item in enumerate(
            st.session_state.a4_items
        )
    ]


    selected_name = st.selectbox(
        "🎯 اختر المستند الذي تريد التحكم به:",
        item_names,
        index=min(
            st.session_state.active_doc_idx,
            len(item_names) - 1,
        ),
    )


    selected_idx = item_names.index(
        selected_name
    )


    st.session_state.active_doc_idx = (
        selected_idx
    )


    current = st.session_state.a4_items[
        selected_idx
    ]


    col_preview, col_controls = st.columns(
        [
            1.35,
            1,
        ]
    )


    # ========================================================
    # المعاينة
    # ========================================================

    with col_preview:

        st.markdown(
            "### 🖨️ معاينة A4"
        )


        st.caption(
            "انقر داخل الصفحة لتحديد الموضع الحر للمستند المحدد."
        )


        a4_display = draw_a4_preview(
            st.session_state.a4_items,
            preview_w=720,
        )


        click_a4 = (
            streamlit_image_coordinates(
                a4_display,
                key=f"a4_position_{selected_idx}_{orientation}",
            )
        )


        if click_a4 is not None:

            a4_w = a4_display.width
            a4_h = a4_display.height


            new_x = int(
                (
                    click_a4["x"] /
                    float(a4_w)
                )
                * 100
            )


            new_y = int(
                (
                    click_a4["y"] /
                    float(a4_h)
                )
                * 100
            )


            new_x = max(
                0,
                min(
                    85,
                    new_x,
                ),
            )


            new_y = max(
                0,
                min(
                    90,
                    new_y,
                ),
            )


            current["x"] = new_x
            current["y"] = new_y


            st.rerun()


        st.image(
            a4_display,
            use_container_width=True,
        )


        st.caption(
            f"المستند المحدد: {current['title']} "
            f"| X={current['x']}% "
            f"| Y={current['y']}%"
        )


    # ========================================================
    # التحكم
    # ========================================================

    with col_controls:

        st.markdown(
            "### 🎯 التحكم بالمستند"
        )


        # ----------------------------------------------------
        # أماكن سريعة
        # ----------------------------------------------------

        st.markdown(
            "#### 📍 تحديد المكان"
        )


        quick_position = st.selectbox(
            "اختر مكاناً جاهزاً:",
            [
                "تحديد حر",
                "أعلى اليسار",
                "أعلى الوسط",
                "أعلى اليمين",
                "وسط اليسار",
                "منتصف الصفحة",
                "وسط اليمين",
                "أسفل اليسار",
                "أسفل الوسط",
                "أسفل اليمين",
            ],
        )


        positions = {

            "أعلى اليسار": (3, 3),

            "أعلى الوسط": (28, 3),

            "أعلى اليمين": (53, 3),

            "وسط اليسار": (3, 40),

            "منتصف الصفحة": (28, 40),

            "وسط اليمين": (53, 40),

            "أسفل اليسار": (3, 75),

            "أسفل الوسط": (28, 75),

            "أسفل اليمين": (53, 75),
        }


        if quick_position in positions:

            new_pos = positions[
                quick_position
            ]

            current["x"] = new_pos[0]
            current["y"] = new_pos[1]

            st.rerun()


        # ----------------------------------------------------
        # X / Y
        # ----------------------------------------------------

        st.markdown(
            "#### ↔️ الموضع الدقيق"
        )


        current["x"] = st.slider(
            "X — أفقي",
            0,
            90,
            int(current["x"]),
            step=1,
            key=f"x_{current['id']}",
        )


        current["y"] = st.slider(
            "Y — رأسي",
            0,
            90,
            int(current["y"]),
            step=1,
            key=f"y_{current['id']}",
        )


        # ----------------------------------------------------
        # الحجم
        # ----------------------------------------------------

        st.markdown(
            "#### 🔎 حجم المستند"
        )


        current["scale"] = st.slider(
            "تكبير / تصغير (%)",
            20,
            200,
            int(current["scale"]),
            step=1,
            key=f"scale_{current['id']}",
        )


        # ----------------------------------------------------
        # الدوران
        # ----------------------------------------------------

        st.markdown(
            "#### 🔄 تدوير المستند"
        )


        current["rotation"] = st.slider(
            "زاوية الدوران",
            -180,
            180,
            int(current.get("rotation", 0)),
            step=1,
            key=f"rotation_{current['id']}",
        )


        # ----------------------------------------------------
        # نسخ المستند
        # ----------------------------------------------------

        if st.button(
            "📋 نسخ المستند المحدد",
            use_container_width=True,
        ):

            new_item = current.copy()

            new_item["id"] = (
                len(
                    st.session_state.a4_items
                )
                + 1
            )

            new_item["title"] = (
                current["title"]
                +
                " - نسخة"
            )

            new_item["x"] = min(
                current["x"] + 5,
                85,
            )

            new_item["y"] = min(
                current["y"] + 5,
                90,
            )

            st.session_state.a4_items.append(
                new_item
            )

            st.session_state.active_doc_idx = (
                len(
                    st.session_state.a4_items
                ) - 1
            )

            st.rerun()


        # ----------------------------------------------------
        # حذف
        # ----------------------------------------------------

        if st.button(
            "🗑️ حذف المستند المحدد",
            use_container_width=True,
        ):

            st.session_state.a4_items.pop(
                selected_idx
            )

            if (
                st.session_state.active_doc_idx
                >= len(
                    st.session_state.a4_items
                )
            ):

                st.session_state.active_doc_idx = (
                    max(
                        0,
                        len(
                            st.session_state.a4_items
                        ) - 1,
                    )
                )

            st.rerun()


        # ----------------------------------------------------
        # تفريغ الصفحة
        # ----------------------------------------------------

        if st.button(
            "🧹 تفريغ ورقة A4 بالكامل",
            use_container_width=True,
        ):

            st.session_state.a4_items = []

            st.session_state.active_doc_idx = 0

            st.rerun()


    # ========================================================
    # قائمة المستندات
    # ========================================================

    st.markdown(
        "---"
    )

    st.markdown(
        "### 📚 المستندات الموجودة داخل A4"
    )


    for idx, item in enumerate(
        st.session_state.a4_items
    ):

        col_img, col_info = st.columns(
            [
                1,
                4,
            ]
        )


        with col_img:

            thumb = Image.fromarray(
                item["img"]
            )

            thumb.thumbnail(
                (
                    160,
                    100,
                )
            )

            st.image(
                thumb,
                use_container_width=True,
            )


        with col_info:

            st.markdown(
                f"**{idx + 1}. {item['title']}**"
            )

            st.caption(
                f"X: {item['x']}% | "
                f"Y: {item['y']}% | "
                f"الحجم: {item['scale']}% | "
                f"الدوران: {item.get('rotation', 0)}°"
            )


# ============================================================
# التصدير
# ============================================================

st.markdown(
    "---"
)

st.markdown(
    "## 💾 تحميل الملف النهائي"
)


if len(st.session_state.a4_items) > 0:

    export_col1, export_col2 = st.columns(
        2
    )


    with export_col1:

        export_format = st.selectbox(
            "📁 صيغة الملف:",
            [
                "PDF",
                "JPG",
                "PNG",
                "WEBP",
            ],
        )


        compression_enabled = st.checkbox(
            "🗜️ ضغط الملف قبل التحميل",
            value=True,
        )


        if compression_enabled:

            quality = st.slider(
                "جودة الملف / مستوى الضغط:",
                20,
                100,
                85,
                step=5,
            )

        else:

            quality = 100


    with export_col2:

        st.markdown(
            "### 📊 معلومات الملف"
        )

        st.write(
            f"عدد المستندات: "
            f"**{len(st.session_state.a4_items)}**"
        )

        st.write(
            f"مقاس الصفحة: "
            f"**A4 {st.session_state.a4_orientation}**"
        )


    # ========================================================
    # بناء A4 الأصلية 300 DPI
    # ========================================================

    A4_WIDTH, A4_HEIGHT = get_a4_size()


    final_sheet = Image.new(
        "RGB",
        (
            A4_WIDTH,
            A4_HEIGHT,
        ),
        (
            255,
            255,
            255,
        ),
    )


    for item in st.session_state.a4_items:

        doc_img = Image.fromarray(
            item["img"]
        ).convert("RGB")


        # ----------------------------------------------
        # الحجم
        # ----------------------------------------------

        target_w = int(
            A4_WIDTH
            *
            0.44
            *
            (
                item["scale"] /
                100.0
            )
        )


        target_w = max(
            target_w,
            20,
        )


        ratio = (
            doc_img.height /
            float(doc_img.width)
        )


        target_h = int(
            target_w * ratio
        )


        target_h = max(
            target_h,
            20,
        )


        doc_img = doc_img.resize(
            (
                target_w,
                target_h,
            ),
            Image.Resampling.LANCZOS,
        )


        # ----------------------------------------------
        # الدوران
        # ----------------------------------------------

        rotation = item.get(
            "rotation",
            0,
        )


        if rotation != 0:

            doc_img = doc_img.rotate(
                rotation,
                expand=True,
                resample=Image.Resampling.BICUBIC,
                fillcolor="white",
            )


        # ----------------------------------------------
        # الموقع
        # ----------------------------------------------

        pos_x = int(
            A4_WIDTH
            *
            (
                item["x"] /
                100.0
            )
        )


        pos_y = int(
            A4_HEIGHT
            *
            (
                item["y"] /
                100.0
            )
        )


        # ----------------------------------------------
        # التأكد من عدم خروج المستند بالكامل
        # ----------------------------------------------

        if pos_x >= A4_WIDTH:
            pos_x = A4_WIDTH - 20

        if pos_y >= A4_HEIGHT:
            pos_y = A4_HEIGHT - 20


        # ----------------------------------------------
        # القص الآمن عند الحواف
        # ----------------------------------------------

        if (
            pos_x + doc_img.width
            <= A4_WIDTH
            and
            pos_y + doc_img.height
            <= A4_HEIGHT
        ):

            final_sheet.paste(
                doc_img,
                (
                    pos_x,
                    pos_y,
                ),
            )

        else:

            # إذا خرج جزء من الصورة
            # نقوم بقصه حتى لا يحدث خطأ

            available_w = max(
                1,
                A4_WIDTH - pos_x,
            )

            available_h = max(
                1,
                A4_HEIGHT - pos_y,
            )


            cropped = doc_img.crop(
                (
                    0,
                    0,
                    min(
                        doc_img.width,
                        available_w,
                    ),
                    min(
                        doc_img.height,
                        available_h,
                    ),
                )
            )


            final_sheet.paste(
                cropped,
                (
                    pos_x,
                    pos_y,
                ),
            )


    # ========================================================
    # معاينة نهائية
    # ========================================================

    st.markdown(
        "### 👁️ المعاينة النهائية قبل التحميل"
    )


    final_preview = final_sheet.copy()

    final_preview.thumbnail(
        (
            900,
            1100,
        )
    )


    st.image(
        final_preview,
        use_container_width=False,
    )


    # ========================================================
    # التصدير
    # ========================================================

    output_buffer = io.BytesIO()


    if export_format == "PDF":

        # تحويل إلى JPEG داخلياً للحصول على
        # تحكم أفضل بحجم الملف

        if compression_enabled:

            jpeg_buffer = io.BytesIO()

            final_sheet.save(
                jpeg_buffer,
                format="JPEG",
                quality=quality,
                optimize=True,
            )

            jpeg_buffer.seek(0)

            jpeg_image = Image.open(
                jpeg_buffer
            ).convert("RGB")

            jpeg_image.save(
                output_buffer,
                format="PDF",
                resolution=300.0,
            )

        else:

            final_sheet.save(
                output_buffer,
                format="PDF",
                resolution=300.0,
            )


        mime_type = (
            "application/pdf"
        )

        extension = "pdf"


    elif export_format == "JPG":

        final_sheet.save(
            output_buffer,
            format="JPEG",
            quality=quality
            if compression_enabled
            else 100,
            optimize=True,
        )

        mime_type = (
            "image/jpeg"
        )

        extension = "jpg"


    elif export_format == "PNG":

        final_sheet.save(
            output_buffer,
            format="PNG",
            optimize=True,
        )

        mime_type = (
            "image/png"
        )

        extension = "png"


    else:

        final_sheet.save(
            output_buffer,
            format="WEBP",
            quality=quality
            if compression_enabled
            else 100,
            method=6,
        )

        mime_type = (
            "image/webp"
        )

        extension = "webp"


    file_data = (
        output_buffer.getvalue()
    )


    file_size_kb = (
        len(file_data)
        /
        1024.0
    )


    file_size_mb = (
        file_size_kb
        /
        1024.0
    )


    st.info(
        f"📦 حجم الملف النهائي: "
        f"**{file_size_kb:.1f} KB** "
        f"({file_size_mb:.2f} MB)"
    )


    st.download_button(
        label=(
            f"📥 تحميل A4 "
            f"({extension.upper()})"
        ),
        data=file_data,
        file_name=(
            f"documents_A4."
            f"{extension}"
        ),
        mime=mime_type,
        type="primary",
        use_container_width=True,
    )


else:

    st.info(
        "أضف مستنداً واحداً على الأقل إلى A4 حتى يظهر خيار التحميل."
    )


# ============================================================
# معلومات النظام
# ============================================================

st.markdown(
    "---"
)

st.caption(
    "منظومة تصحيح المستمسكات وتجهيز A4 — "
    "معالجة الصور تتم داخل التطبيق."
)
