import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
def build_model(num_classes):
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model
num_classes = 2
model = build_model(num_classes)
model.summary()
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    'D:\\Drone_Detection\\DroneOrBirds',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    epochs=10,
    steps_per_epoch=len(train_generator),
    verbose=1
)
test_datagen = ImageDataGenerator(rescale=1.0/255)

test_generator = test_datagen.flow_from_directory(
    'D:\\Drone_Detection\\DroneOrBirds',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_loss, test_acc = model.evaluate(test_generator, verbose=2)
print("\nTest accuracy:", test_acc)
import numpy as np
from tensorflow.keras.preprocessing import image
def predict_drone(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    not_drone_probability = prediction[0][0]

    if not_drone_probability > 0.5:
        return "Not a drone"
    else:
        return "Drone"
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
image_path = '/bird.jpg'
img = mpimg.imread(image_path)
plt.imshow(img)
plt.axis('off')  # Hides the axes
plt.show()
prediction = predict_drone(image_path)
print("Prediction:", prediction)