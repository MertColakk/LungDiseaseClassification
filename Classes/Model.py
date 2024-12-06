import os
from keras import layers, models, utils, callbacks
import matplotlib.pyplot as plt

# train 4244,  valid 1818 , test 1070
class LungModel:
    def __init__(self,data_dir:str):
        self.class_names = ['COVID-19','NORMAL','PNEUMONIA','TUBERCULOSIS']
        self.data_dir = data_dir

        # Create runs folder
        try:
            os.mkdir("LungDiseaseClassification/runs")
        except FileExistsError:
            print("Directory exists!")
        except PermissionError:
            print("No permission in this folder!")
        except Exception as e:
            print(f"An error occured {e}")

        # Datasets
        self.train_dataset = utils.image_dataset_from_directory(
            os.path.join(self.data_dir, "train"),
            image_size=(224, 224),
            batch_size=64
        )
        self.test_dataset = utils.image_dataset_from_directory(
            os.path.join(self.data_dir, "test"),
            image_size=(224, 224),
            batch_size=64
        )
        self.validation_dataset = utils.image_dataset_from_directory(
            os.path.join(self.data_dir, "val"),
            image_size=(224, 224),
            batch_size=64
        )

        # Normalize Dataset
        self.normalization_layer = layers.Rescaling(1. / 255)

        self.train_dataset = self.train_dataset.map(lambda x, y: (self.normalization_layer(x), y))
        self.validation_dataset = self.validation_dataset.map(lambda x, y: (self.normalization_layer(x), y))

        # Create model
        self.model = models.Sequential([
            # Layers
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(512, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),

            # Fully Connected Layers
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(4, activation='softmax')
        ])

        # Compile
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, epoch_size: int):

        callback = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        history = self.model.fit(
            self.train_dataset,
            validation_data=self.validation_dataset,
            epochs=epoch_size,
            callbacks=[callback]
        )

        self.visualize_train(history)

        self.save_model()

    def visualize_train(self, history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(len(acc))

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Train Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Loss')
        plt.show()

    def save_model(self):
        save_path = os.path.join('runs', 'best_weight.h5')
        self.model.save(save_path)

