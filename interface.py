import gradio as gr
import numpy
from tensorflow.keras.models import load_model


model = load_model("./model/brain_tumor_model.h5")

def preprocess_and_predict(image):
	image = image.resize((256, 256))
	pix = numpy.array(image.getdata()).reshape(-1, 256, 256, 3)
	prediction = model.predict(pix)
	if prediction[0] > 0.5:
		return "Tumorous"
	return "Healthy"

image_input = gr.Image(type="pil")

app = gr.Interface(fn=preprocess_and_predict, inputs=image_input, outputs="text")
app.launch(inbrowser=True)