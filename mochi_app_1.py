from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from datetime import datetime
import random
import os
import requests
import base64
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Load your trained Random Forest model
with open('mochi_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Activity descriptions and image prompts
ACTIVITIES = {
    'sleeping': {
        'description': "Mochi is curled up and snoozing peacefully 💤",
        'prompt': "A cute fluffy white dog sleeping curled up in a cozy bed, soft lighting, anime style"
    },
    'eating': {
        'description': "Mochi is munching on some yummy food 🍖",
        'prompt': "An adorable white fluffy dog eating from a bowl with happy expression, food pieces flying, cartoon style"
    },
    'playing': {
        'description': "Mochi is playing with toys and having fun 🎾",
        'prompt': "A cute white dog playing joyfully with colorful toys in a sunny living room, watercolor style"
    },
    'walking': {
        'description': "Mochi is out for a walk exploring the world 🦮",
        'prompt': "Fluffy white dog walking happily on a leash in a park with trees and flowers, digital art style"
    },
    'barking': {
        'description': "Mochi is barking at something interesting 🐕",
        'prompt': "Small white dog barking excitedly with ears perked up, comic book style with motion lines"
    }
}

def generate_ai_image(prompt):
    """Generate image using Stability AI API"""
    API_URL = "https://api.stability.ai/v1/generation/stable-diffusion-v1-6/text-to-image"
    API_KEY = os.getenv('STABILITY_API_KEY')  # Get your key from stability.ai
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "text_prompts": [{"text": prompt, "weight": 1}],
        "cfg_scale": 7,
        "height": 512,
        "width": 512,
        "steps": 30,
        "samples": 1,
        "style_preset": "anime"  # Makes images more cute/anime-like
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        # Save generated image
        data = response.json()
        image_data = data["artifacts"][0]["base64"]
        image_bytes = base64.b64decode(image_data)
        
        # Save to static folder with unique filename
        filename = f"mochi_{int(time.time())}.png"
        filepath = os.path.join("static", filename)
        with open(filepath, "wb") as f:
            f.write(image_bytes)
            
        return f"/static/{filename}"
        
    except Exception as e:
        print(f"Image generation failed: {str(e)}")
        # Fallback to placeholder images
        return f"/static/mochi_placeholder_{random.randint(1,2)}.jpg"

@app.route('/')
def home():
    return render_template('mochi.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Parse time and day
        time_str = data['time']
        day_of_week = data['day']
        
        # Convert to model features (adjust based on your model)
        hour = int(time_str.split(':')[0])
        minute = int(time_str.split(':')[1])
        day_encoded = ['mon','tue','wed','thu','fri','sat','sun'].index(day_of_week.lower())
        
        # Create feature array for model
        features = np.array([[hour, minute, day_encoded]])
        
        # Get prediction
        activity = model.predict(features)[0]
        activity_details = ACTIVITIES.get(activity, {
            'description': f"Mochi is {activity}",
            'prompt': f"A cute white dog {activity}"
        })
        
        # Generate AI image
        image_url = generate_ai_image(activity_details['prompt'])
        
        return jsonify({
            'status': 'success',
            'activity': activity,
            'description': activity_details['description'],
            'image_url': image_url,
            'time': time_str,
            'day': day_of_week
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    # Create static directory if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)