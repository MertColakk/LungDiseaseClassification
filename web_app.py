import numpy as np
from flask import Flask, request, render_template
from Classes.Web_Model import Web_Model
from PIL import Image

# Create Web App
app = Flask(__name__,template_folder='LungDiseaseClassification/Templates')

model = Web_Model("LungProject/runs/best_weight.h5",['COVID-19','NORMAL','PNEUMONIA','TUBERCULOSIS'])

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")  
        if file:
            try:
                image = Image.open(file).convert("RGB")

                prediction = model.predict(image.copy())

                return render_template("result.html", prediction=prediction)
            except Exception as e:
                print(f"Error processing the image: {e}")
                return render_template("index.html", error="Invalid image file. Please try again.")
        
        return render_template("index.html", error="Please upload an image.")
    return render_template("index.html")


    # For GET requests, simply render the index page.
    return render_template("/index.html")


if __name__ == "__main__":
    app.run(debug=True)