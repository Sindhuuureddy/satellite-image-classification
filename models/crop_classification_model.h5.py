from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Example function to train a model
def train_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')  # 3 categories: rabi, kharif, other
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Example for training the model (replace with your data)
    train_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory('path_to_train_data', target_size=(224, 224), batch_size=32, class_mode='categorical')

    model.fit(train_generator, epochs=10)
    model.save("crop_classification_model.h5.py")  # Save the trained model
    print("Model trained and saved as 'crop_classification_model.h5'")

train_model()  # Call this function to train and save your model
