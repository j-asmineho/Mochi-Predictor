from flask_cors import CORS  
from flask import Flask, jsonify, request
import joblib
import numpy as np
from diffusers import StableDiffusionPipeline
import torch

app = Flask(__name__)
CORS(app)

# Load your trained model
model = joblib.load('model.pkl')

# Load Stable Diffusion locally
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
pipe = pipe.to("cpu")

@app.route('/predict')
def predict():
    time = float(request.args.get('time', 8.5))
    
    # Predict activity
    hour_sin = np.sin(2 * np.pi * time/24)
    hour_cos = np.cos(2 * np.pi * time/24)
    activity = model.predict([[hour_sin, hour_cos, 5, 0, 0, 0, 0]])[0]
    
    # Generate image
    prompt = f"Cute shiba inu {activity}, digital art style"
    image = pipe(prompt).images[0]
    image.save("static/mochi.png")
    
    return jsonify({
        'activity': activity,
        'image_url': '/static/mochi.png',
        'prompt': prompt
    })

if __name__ == '__main__':
    app.run(debug=True)