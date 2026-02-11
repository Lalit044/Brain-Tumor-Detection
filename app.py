from flask import Flask, render_template, request, send_file
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2

app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
model = tf.keras.models.load_model("model/brain_tumor_model.h5")
# Build model once to define inputs/outputs
dummy_input = tf.zeros((1, 224, 224, 3))
model(dummy_input)


class_labels = ['glioma', 'meningioma', 'pituitary', 'no_tumor']

# ------------------ Tumor Info ------------------
tumor_info = {
    "glioma": {
        "about": "Glioma is a tumor that begins in the glial cells.",
        "symptoms": ["Persistent headaches", "Seizures", "Blurred vision"],
        "treatment": ["Surgery", "Radiation therapy", "Chemotherapy"],
        "advice": "Consult a neurologist immediately."
    },
    "meningioma": {
        "about": "Meningioma develops from protective membranes of the brain.",
        "symptoms": ["Vision problems", "Headaches", "Memory issues"],
        "treatment": ["Surgery", "Radiation therapy"],
        "advice": "Consult a neurosurgeon."
    },
    "pituitary": {
        "about": "Pituitary tumors affect hormone production.",
        "symptoms": ["Hormonal imbalance", "Vision changes"],
        "treatment": ["Medication", "Surgery"],
        "advice": "Consult an endocrinologist."
    },
    "no_tumor": {
        "about": "No tumor detected.",
        "symptoms": ["If symptoms persist, consult doctor."],
        "treatment": ["Routine monitoring"],
        "advice": "Follow up if symptoms continue."
    }
}

# ------------------ Prediction ------------------
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = round(100 * np.max(prediction), 2)

    return predicted_class, confidence


# ------------------ Grad-CAM ------------------
def generate_gradcam(img_path, prediction_index):

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    base_model = model.layers[0]  # MobileNetV2

    # Get last convolutional layer
    last_conv_layer = None
    for layer in reversed(base_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break

    if last_conv_layer is None:
        return None

    # Create model that outputs conv layer output
    conv_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=last_conv_layer.output
    )

    with tf.GradientTape() as tape:
        conv_output = conv_model(img_array)
        tape.watch(conv_output)

        # Forward pass manually through remaining layers
        x = conv_output
        x = model.layers[1](x)  # GlobalAveragePooling
        x = model.layers[2](x)  # Dense
        x = model.layers[3](x)  # Dropout
        predictions = model.layers[4](x)  # Final Dense

        loss = predictions[:, prediction_index]

    grads = tape.gradient(loss, conv_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]

    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8
    heatmap = heatmap.numpy()

    img_original = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img_original.shape[1], img_original.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img_original, 0.6, heatmap, 0.4, 0)

    gradcam_path = "static/gradcam.jpg"
    cv2.imwrite(gradcam_path, superimposed_img)

    return gradcam_path


# ------------------ PDF ------------------
def generate_pdf_report(prediction, confidence, risk_level):
    file_path = "static/report.pdf"
    doc = SimpleDocTemplate(file_path, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("<b>AI Brain Tumor Diagnosis Report</b>", styles["Title"]))
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph(f"Diagnosis: {prediction}", styles["Normal"]))
    elements.append(Paragraph(f"Confidence: {confidence}%", styles["Normal"]))
    elements.append(Paragraph(f"Risk Level: {risk_level}", styles["Normal"]))
    elements.append(Paragraph(f"Date: {datetime.now()}", styles["Normal"]))

    doc.build(elements)
    return file_path


# ------------------ Confusion Matrix ------------------
def generate_confusion_matrix():
    test_dir = "dataset/test"

    datagen = ImageDataGenerator(rescale=1./255)

    test_generator = datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=class_labels,
                yticklabels=class_labels,
                cmap="Blues")

    cm_path = "static/confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()

    return cm_path


# ------------------ Routes ------------------
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/team')
def team():
    return render_template("team.html")

@app.route('/confusion-matrix')
def confusion_matrix_page():
    cm_path = generate_confusion_matrix()
    return render_template("confusion_matrix.html", cm_image=cm_path)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    result, confidence = predict_image(filepath)

    # Risk Level
    if confidence >= 85:
        risk_level = "High Confidence"
        risk_color = "green"
    elif confidence >= 60:
        risk_level = "Moderate Confidence"
        risk_color = "orange"
    else:
        risk_level = "Low Confidence"
        risk_color = "red"

    prediction_index = class_labels.index(result)
    gradcam_path = generate_gradcam(filepath, prediction_index)

    report_path = generate_pdf_report(result, confidence, risk_level)

    info = tumor_info.get(result)

    return render_template(
        "result.html",
        prediction=result,
        confidence=confidence,
        risk_level=risk_level,
        risk_color=risk_color,
        image_path=filepath,
        gradcam_image=gradcam_path,
        report_path=report_path,
        about=info["about"],
        symptoms=info["symptoms"],
        treatment=info["treatment"],
        advice=info["advice"]
    )

@app.route('/download-report')
def download_report():
    return send_file("static/report.pdf", as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
