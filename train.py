from Classes.Model import LungModel

if __name__ == "__main__":
    # Create Model
    model = LungModel("LungDiseaseClassification/Data")

    # Train model
    model.train(15)