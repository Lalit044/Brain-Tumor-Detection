from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load model
model = load_model("model/brain_tumor_model.h5")

# Test dataset path
test_dir = "dataset/test"

IMG_SIZE = 224
BATCH_SIZE = 32

# Create test generator
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Evaluate model
loss, acc = model.evaluate(test_generator)

print("Test Accuracy:", acc)