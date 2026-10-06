const fileInput =
    document.getElementById("fileInput");

const editorCanvas =
    document.getElementById("editorCanvas");

const previewCanvas =
    document.getElementById("previewCanvas");

const a4Canvas =
    document.getElementById("a4Canvas");


const ectx =
    editorCanvas.getContext("2d");

const pctx =
    previewCanvas.getContext("2d");

const actx =
    a4Canvas.getContext("2d");


let originalImage = null;

let correctedCanvas = null;

let corners = [];

let activeCorner = -1;



/*
    تحميل الصورة
*/

fileInput.addEventListener(
    "change",
    function () {

        const file =
            this.files[0];

        if (!file)
            return;


        const image =
            new Image();


        image.onload =
            function () {

                originalImage = image;

                corners = [];

                setupEditor();

            };


        image.src =
            URL.createObjectURL(file);

    }
);



/*
    إعداد المحرر
*/

function setupEditor() {

    const editor =
        document.getElementById("editor");


    const width =
        editor.clientWidth;

    const height =
        editor.clientHeight;


    const scale =
        Math.min(

            width /
            originalImage.width,

            height /
            originalImage.height

        );


    editorCanvas.width =
        originalImage.width * scale;


    editorCanvas.height =
        originalImage.height * scale;


    corners = [

        {
            x:
                editorCanvas.width * .08,

            y:
                editorCanvas.height * .08
        },

        {
            x:
                editorCanvas.width * .92,

            y:
                editorCanvas.height * .08
        },

        {
            x:
                editorCanvas.width * .92,

            y:
                editorCanvas.height * .92
        },

        {
            x:
                editorCanvas.width * .08,

            y:
                editorCanvas.height * .92
        }

    ];


    drawEditor();


    document.getElementById("status")
        .textContent =
        "حدد زوايا المستند";
}



/*
    رسم الصورة والنقاط
*/

function drawEditor() {

    ectx.clearRect(
        0,
        0,
        editorCanvas.width,
        editorCanvas.height
    );


    ectx.drawImage(

        originalImage,

        0,
        0,

        editorCanvas.width,
        editorCanvas.height

    );


    /*
        الخطوط
    */

    ectx.beginPath();


    corners.forEach(
        (point, index) => {

            if (index === 0)

                ectx.moveTo(
                    point.x,
                    point.y
                );

            else

                ectx.lineTo(
                    point.x,
                    point.y
                );

        }
    );


    ectx.closePath();


    ectx.lineWidth = 3;

    ectx.strokeStyle =
        "#c2183d";

    ectx.stroke();



    /*
        النقاط
    */

    corners.forEach(
        (point, index) => {

            ectx.beginPath();

            ectx.arc(
                point.x,
                point.y,
                13,
                0,
                Math.PI * 2
            );


            ectx.fillStyle =
                "#75182f";

            ectx.fill();


            ectx.fillStyle =
                "white";

            ectx.font =
                "bold 13px Arial";

            ectx.textAlign =
                "center";

            ectx.textBaseline =
                "middle";


            ectx.fillText(

                index + 1,

                point.x,

                point.y

            );

        }
    );

}



/*
    تحديد موقع اللمس
*/

function getPointerPosition(event) {

    const rect =
        editorCanvas
            .getBoundingClientRect();


    return {

        x:
            (event.clientX -
                rect.left)
            *
            editorCanvas.width /
            rect.width,

        y:
            (event.clientY -
                rect.top)
            *
            editorCanvas.height /
            rect.height

    };

}



/*
    بدء تحريك النقطة
*/

editorCanvas.addEventListener(
    "pointerdown",
    function (event) {

        if (!originalImage)
            return;


        const point =
            getPointerPosition(event);


        activeCorner = -1;


        corners.forEach(
            (corner, index) => {

                const distance =
                    Math.hypot(

                        corner.x -
                        point.x,

                        corner.y -
                        point.y

                    );


                if (distance < 30)

                    activeCorner =
                        index;

            }
        );


        if (activeCorner >= 0)

            editorCanvas
                .setPointerCapture(
                    event.pointerId
                );

    }
);



/*
    تحريك النقطة
*/

editorCanvas.addEventListener(
    "pointermove",
    function (event) {

        if (activeCorner < 0)
            return;


        const point =
            getPointerPosition(event);


        corners[activeCorner] = {

            x:
                Math.max(
                    0,
                    Math.min(
                        editorCanvas.width,
                        point.x
                    )
                ),

            y:
                Math.max(
                    0,
                    Math.min(
                        editorCanvas.height,
                        point.y
                    )
                )

        };


        drawEditor();

    }
);



/*
    إيقاف التحريك
*/

editorCanvas.addEventListener(
    "pointerup",
    function () {

        activeCorner = -1;

    }
);



/*
    إعادة الضبط
*/

document
    .getElementById("resetBtn")
    .onclick =
    function () {

        originalImage = null;

        correctedCanvas = null;

        corners = [];

        ectx.clearRect(
            0,
            0,
            editorCanvas.width,
            editorCanvas.height
        );

        pctx.clearRect(
            0,
            0,
            previewCanvas.width,
            previewCanvas.height
        );

        document.getElementById("status")
            .textContent =
            "ارفع صورة للبدء";

    };
