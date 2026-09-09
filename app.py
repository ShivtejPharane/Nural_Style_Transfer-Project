import os
import traceback

import torch

from flask import (
    Flask,
    render_template,
    request,
    send_from_directory
)

from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename

from wtforms import (
    FileField,
    SubmitField,
    FloatField,
    HiddenField
)

from wtforms.validators import InputRequired

from PIL import Image

from torchvision import transforms

# Your existing AdaIN model
from utils.model import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization


# ============================================================
# Flask Configuration
# ============================================================

app = Flask(__name__)

app.config["SECRET_KEY"] = "supersecretkey"

app.config["UPLOAD_FOLDER"] = "static/uploads"

app.config["ALLOWED_EXTENSIONS"] = {
    "png",
    "jpg",
    "jpeg"
}

# Maximum upload size: 20 MB
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

Bootstrap(app)

os.makedirs(
    app.config["UPLOAD_FOLDER"],
    exist_ok=True
)


# ============================================================
# Upload Form
# ============================================================

class UploadForm(FlaskForm):

    content = FileField(
        "Content Image",
        validators=[InputRequired()]
    )

    style = FileField(
        "Style Image",
        validators=[InputRequired()]
    )

    content_path = HiddenField()

    style_path = HiddenField()

    alpha = FloatField(
        "Alpha",
        default=1.0
    )

    submit = SubmitField(
        "Transfer Style"
    )


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("DEVICE:", device)
print("=" * 60)


# ============================================================
# Model Paths
# ============================================================

VGG_PATH = "vgg_normalised.pth"

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

decoder_path = os.path.join(
    BASE_DIR,
    "expriment",
    "BigData01",
    "decoder_1000.pth"
)


# ============================================================
# Load Models
# ============================================================

print("Loading VGG Encoder...")

encoder = VGGEncoder(VGG_PATH).to(device)

print("VGG Encoder loaded successfully.")


print("Loading Decoder...")

decoder = Decoder().to(device)

decoder.load_state_dict(
    torch.load(decoder_path, map_location=device)
)

print("Decoder loaded successfully.")


# Evaluation mode
encoder.eval()
decoder.eval()


print("=" * 60)
print("Models are ready.")
print("=" * 60)


# ============================================================
# Allowed File Check
# ============================================================

def allowed_file(filename):

    return (
        "." in filename
        and
        filename.rsplit(".", 1)[1].lower()
        in app.config["ALLOWED_EXTENSIONS"]
    )


# ============================================================
# Image Transform
# ============================================================

transform = transforms.Compose([
    transforms.Resize(512),
    transforms.ToTensor()
])


# ============================================================
# Style Transfer
# ============================================================

def style_transfer(
    content_image,
    style_image,
    encoder,
    decoder,
    alpha,
    device
):

    # --------------------------------------------------------
    # Convert images to tensors
    # --------------------------------------------------------

    content_tensor = transform(
        content_image
    ).unsqueeze(0).to(device)

    style_tensor = transform(
        style_image
    ).unsqueeze(0).to(device)


    # --------------------------------------------------------
    # Encoder
    # --------------------------------------------------------

    with torch.no_grad():

        content_feats = encoder(
            content_tensor,
            if_test=True
        )

        style_feats = encoder(
            style_tensor,
            if_test=True
        )


        # ----------------------------------------------------
        # Debug: print feature shapes
        # ----------------------------------------------------

        print("\nFeature shapes:")

        if isinstance(content_feats, (list, tuple)):

            print(
                "Content:",
                [
                    feature.shape
                    for feature in content_feats
                ]
            )

            print(
                "Style:",
                [
                    feature.shape
                    for feature in style_feats
                ]
            )

        else:

            print(
                "Content:",
                content_feats.shape
            )

            print(
                "Style:",
                style_feats.shape
            )


        # ----------------------------------------------------
        # Get deepest VGG feature
        # ----------------------------------------------------

        if isinstance(content_feats, (list, tuple)):

            content_feat = content_feats[-1]

            style_feat = style_feats[-1]

        else:

            content_feat = content_feats

            style_feat = style_feats


        print(
            "Content feature used for AdaIN:",
            content_feat.shape
        )

        print(
            "Style feature used for AdaIN:",
            style_feat.shape
        )


        # ----------------------------------------------------
        # AdaIN
        # ----------------------------------------------------

        stylized_feats = adaptive_instance_normalization(
            content_feat,
            style_feat
        )


        print(
            "AdaIN output:",
            stylized_feats.shape
        )


        # ----------------------------------------------------
        # Alpha blending
        # ----------------------------------------------------

        stylized_feats = (
            alpha * stylized_feats
            +
            (1.0 - alpha) * content_feat
        )


        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        stylized_image = decoder(
            stylized_feats
        )


        print(
            "Decoder output:",
            stylized_image.shape
        )


    return stylized_image


# ============================================================
# Save Image
# ============================================================

def save_image(image, path):

    image = image.detach().cpu()

    image = image.squeeze(0)

    image = image.clamp(0, 1)

    image = transforms.ToPILImage()(image)

    image.save(path)

    print(
        "Output saved:",
        path
    )


# ============================================================
# Home Page
# ============================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def index():

    form = UploadForm()

    result_image = None

    content_filename = None

    style_filename = None

    error = None


    # ========================================================
    # POST Request
    # ========================================================

    if request.method == "POST":

        # ----------------------------------------------------
        # Check Content Image
        # ----------------------------------------------------

        if (
            not form.content.data
            or
            not form.content.data.filename
        ):

            error = "Please upload a content image."

            return render_template(
                "index.html",
                form=form,
                result_image=result_image,
                content_image=content_filename,
                style_image=style_filename,
                error=error
            )


        # ----------------------------------------------------
        # Check Style Image
        # ----------------------------------------------------

        if (
            not form.style.data
            or
            not form.style.data.filename
        ):

            error = "Please upload a style image."

            return render_template(
                "index.html",
                form=form,
                result_image=result_image,
                content_image=content_filename,
                style_image=style_filename,
                error=error
            )


        # ----------------------------------------------------
        # Validate Content File
        # ----------------------------------------------------

        if not allowed_file(
            form.content.data.filename
        ):

            error = (
                "Invalid content image format. "
                "Use PNG, JPG or JPEG."
            )

            return render_template(
                "index.html",
                form=form,
                result_image=result_image,
                content_image=content_filename,
                style_image=style_filename,
                error=error
            )


        # ----------------------------------------------------
        # Validate Style File
        # ----------------------------------------------------

        if not allowed_file(
            form.style.data.filename
        ):

            error = (
                "Invalid style image format. "
                "Use PNG, JPG or JPEG."
            )

            return render_template(
                "index.html",
                form=form,
                result_image=result_image,
                content_image=content_filename,
                style_image=style_filename,
                error=error
            )


        # ====================================================
        # Save Content Image
        # ====================================================

        content_filename = secure_filename(
            form.content.data.filename
        )

        content_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            content_filename
        )

        form.content.data.save(
            content_path
        )


        # ====================================================
        # Save Style Image
        # ====================================================

        style_filename = secure_filename(
            form.style.data.filename
        )

        style_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            style_filename
        )

        form.style.data.save(
            style_path
        )


        # ====================================================
        # Perform Style Transfer
        # ====================================================

        try:

            print("\n" + "=" * 60)
            print("STARTING STYLE TRANSFER")
            print("=" * 60)


            # ------------------------------------------------
            # Open images
            # ------------------------------------------------

            content_image = Image.open(
                content_path
            ).convert("RGB")

            style_image = Image.open(
                style_path
            ).convert("RGB")


            print(
                "Content image:",
                content_image.size
            )

            print(
                "Style image:",
                style_image.size
            )


            # ------------------------------------------------
            # Alpha
            # ------------------------------------------------

            alpha = form.alpha.data

            if alpha is None:

                alpha = 1.0

            alpha = float(alpha)

            # Keep alpha between 0 and 1
            alpha = max(
                0.0,
                min(1.0, alpha)
            )


            print(
                "Alpha:",
                alpha
            )


            # ------------------------------------------------
            # Style Transfer
            # ------------------------------------------------

            stylized_image = style_transfer(
                content_image,
                style_image,
                encoder,
                decoder,
                alpha,
                device
            )


            # ------------------------------------------------
            # Output filename
            # ------------------------------------------------

            base_name = os.path.splitext(
                content_filename
            )[0]

            result_filename = (
                "stylized_"
                +
                base_name
                +
                ".png"
            )


            result_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                result_filename
            )


            # ------------------------------------------------
            # Save output
            # ------------------------------------------------

            save_image(
                stylized_image,
                result_path
            )


            result_image = result_filename


            print("=" * 60)
            print("STYLE TRANSFER COMPLETED")
            print("=" * 60)


        except Exception as e:

            print("\n" + "=" * 60)
            print("ERROR DURING STYLE TRANSFER")
            print("=" * 60)

            traceback.print_exc()

            error = str(e)


    # ========================================================
    # Render HTML
    # ========================================================

    return render_template(
        "index.html",
        form=form,
        result_image=result_image,
        content_image=content_filename,
        style_image=style_filename,
        error=error
    )


# ============================================================
# Serve Uploaded Images
# ============================================================

@app.route(
    "/uploads/<filename>"
)
def send_image(filename):

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


# ============================================================
# Run Flask Server
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("STARTING FLASK SERVER")
    print("=" * 60)

    print(
        "Open this URL in your browser:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print("=" * 60)


    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
