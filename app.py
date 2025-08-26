# NOTE: Ensure 'class_indices.json' is present in the project directory.
# This file is required for class label mapping in predictions.
# If not present, download it from the model repository or contact the administrator.

from flask import Flask, render_template, request, redirect, url_for, session, flash
import os

# Ensure this env var is set BEFORE TensorFlow imports to suppress oneDNN op-order noise
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from supabase import create_client
from dotenv import load_dotenv
import bcrypt
import pandas as pd
import json
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'default-secret-key')
UPLOAD_FOLDER = 'static/uploads'

# Make user session available to all templates
@app.context_processor
def inject_user():
    return dict(user=session.get('user'))

# Initialize Supabase client
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_ANON_KEY')

if not supabase_url or not supabase_key:
    raise ValueError("Missing Supabase credentials. Please set SUPABASE_URL and SUPABASE_ANON_KEY in .env file")

try:
    supabase = create_client(supabase_url, supabase_key)
except Exception as e:
    raise ValueError(f"Failed to initialize Supabase client: {str(e)}")

def get_supabase_client():

    access_token = session.get('access_token')
    if access_token:

        return create_client(supabase_url, supabase_key, options={
            "headers": {
                "Authorization": f"Bearer {access_token}"
            }
        })
    return supabase

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your model (updated)
MODEL_PATH = 'mobilenetv2_soil_finetuned_final.h5'
CLASS_INDICES_PATH = 'class_indices.json'

def _try_load_model(path):
    """
    Try loading a Keras .h5 model. If it fails due to common architecture mismatches
    provide actionable error messages and try fallback savedmodel dir if present.
    """
    # 1) direct .h5
    try:
        print(f"Attempting to load Keras model from: {path}")
        return load_model(path, compile=False)
    except Exception as e:
        err = str(e)
        print("Primary .h5 load failed:", err)
        # 2) fallback: try SavedModel directory with same base name
        savedmodel_dir = os.path.splitext(path)[0]
        if os.path.isdir(savedmodel_dir):
            try:
                print(f"Attempting to load SavedModel from directory: {savedmodel_dir}")
                return load_model(savedmodel_dir, compile=False)
            except Exception as e2:
                print("SavedModel fallback failed:", str(e2))
                # continue to raise informative error below
                err = f"{err}\nSavedModel fallback error: {e2}"

        # 3) Specific mismatch diagnostics
        if "expects 1 input(s)" in err or "received 2 input tensors" in err or "expects" in err and "input" in err:
            raise ValueError(
                "Model load failed due to an input/tensor mismatch. Likely causes:\n"
                "- The .h5 was saved from a functional/multi-input model different from the current environment.\n"
                "- The model uses custom layers/objects not available in this environment.\n\n"
                "Recommended actions:\n"
                "1) Re-save the model on the training machine as a SavedModel directory: `model.save('model_dir', save_format='tf')` and copy the directory here.\n"
                "2) Or save architecture + weights separately: `open('model.json','w').write(model.to_json())` and `model.save_weights('weights.h5')`, then load architecture+weights.\n"
                "3) If custom layers were used, load with `load_model(..., custom_objects={...})`.\n\n"
                f"Original load errors:\n{err}"
            )
        # 4) otherwise re-raise the original exception
        raise

# validate files
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}. Please ensure the file exists in the project directory or provide a SavedModel folder named '{os.path.splitext(MODEL_PATH)[0]}'.")

if not os.path.exists(CLASS_INDICES_PATH):
    raise FileNotFoundError(f"Class indices file not found: {CLASS_INDICES_PATH}. Please ensure the file exists in the project directory.")

# Load class mapping from JSON early (needed for reconstruction fallback)
with open(CLASS_INDICES_PATH, 'r') as f:
    class_indices = json.load(f)
idx2class = {v: k for k, v in class_indices.items()}
num_classes = len(class_indices)


def _reconstruct_mobilenetv2(num_classes: int):
    """Rebuild a typical MobileNetV2 fine-tune head and return the model.
    Assumptions:
    - input size 224x224x3
    - include_top=False
    - GlobalAveragePooling2D
    - Dense num_classes softmax
    """
    base = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
    x = layers.GlobalAveragePooling2D(name='avg_pool')(base.output)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    return Model(inputs=base.input, outputs=outputs, name='mobilenetv2_soil')

# Try loading model from .h5, SavedModel, or architecture+weights
try:
    model = _try_load_model(MODEL_PATH)
except Exception as e_h5:
    print("Primary and SavedModel load failed. Trying model.json + weights.h5 fallback.")
    model = None
    model_json_path = 'mobilenetv2_architecture_fine_tune.json'
    weights_path = 'mobilenetv2_fine_tune.weights.h5'
    if os.path.exists(model_json_path) and os.path.exists(weights_path):
        try:
            with open(model_json_path, 'r') as f:
                model = model_from_json(f.read())
            model.load_weights(weights_path)
            print("Loaded model from mobilenetv2_architecture_fine_tune.json and weights.")
        except Exception as e_json:
            print("JSON + weights load failed:", str(e_json))
    # If still not loaded, try reconstructing the model and loading weights
    if model is None and os.path.exists(weights_path):
        try:
            print("Attempting to reconstruct MobileNetV2 and load weights…")
            model = _reconstruct_mobilenetv2(num_classes)
            model.load_weights(weights_path)
            print("Loaded weights into reconstructed MobileNetV2 model.")
        except Exception as e_rec:
            print("Reconstruction + weights load failed:", str(e_rec))
    if model is None:
        # Re-raise the original error for visibility if all fallbacks fail
        raise e_h5

IMG_SIZE = 224

def preprocess_image(img_path):
    img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    prediction_result = None
    confidence = None
    image_url = None
    all_preds = {}
    # Use labels from class_indices mapping
    if request.method == 'POST':
        file = request.files.get('image')
        if file and file.filename:
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            img_tensor = preprocess_image(filepath)
            preds = model.predict(img_tensor)[0]

            predicted_index = int(np.argmax(preds))
            prediction_result = idx2class.get(predicted_index, str(predicted_index))
            confidence = float(preds[predicted_index]) * 100
            image_url = filepath
            all_preds = {idx2class.get(i, str(i)): float(preds[i]) * 100 for i in range(len(preds))}

    return render_template(
        'prediction.html',
        prediction=prediction_result,
        confidence=confidence,
        image_url=image_url,
        all_preds=all_preds
    )

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        if 'user' not in session:
            flash('Please sign in to send a message.', 'error')
            return redirect(url_for('sign_in'))

        user_info = session['user']
        message = request.form.get('message')

        if not message:
            flash('Please enter a message.', 'error')
            return redirect(url_for('contact'))

        try:
            query_data = {
                'name': user_info['name'],
                'email': user_info['email'],
                'message': message,
                'user_id': user_info['id']
            }
            
            # Debug logging
            print("User info:", user_info)
            print("Attempting to insert:", query_data)
            
            result = supabase.from_('contact_queries').insert(query_data).execute()
            print("Insert result:", result)
            
            flash('Your message has been sent successfully!', 'success')
            return redirect(url_for('contact'))

        except Exception as e:
            print("Error details:", str(e))
            flash('An error occurred while sending your message. Please try again.', 'error')
            return redirect(url_for('contact'))
            
    return render_template('contact.html')

# Middleware to check authentication for protected routes
@app.before_request


def check_auth():
    protected_routes = ['prediction', 'contact']
    if request.endpoint in protected_routes and 'user' not in session:
        # Do not flash for contact page, only redirect
        if request.endpoint == 'contact':
            return redirect(url_for('sign_in'))
        # Only flash if not already on sign-in page
        if request.endpoint != 'sign_in':
            if not session.get('_flashed_signin_warning'):
                flash('Please sign in to access this feature.', 'warning')
                session['_flashed_signin_warning'] = True
        return redirect(url_for('sign_in'))
    session.pop('_flashed_signin_warning', None)

@app.route('/sign-in', methods=['GET', 'POST'])
def sign_in():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        try:
            user_data_response = supabase.from_('users').select("*").eq('email', email).single().execute()
            
            if not user_data_response.data:
                raise Exception("User not found or incorrect credentials")

            user_data = user_data_response.data
            stored_password = user_data['password']
            if not bcrypt.checkpw(password.encode('utf-8'), stored_password.encode('utf-8')):
                raise Exception("Incorrect password")

            session['user'] = {
                'id': user_data['id'],
                'email': user_data['email'],
                'name': user_data['name']
            }
            return redirect(url_for('index'))
        except Exception as e:
            print("Sign-in error:", str(e))
            flash(str(e), 'error')
            return render_template('sign-in.html', error=str(e))

    return render_template('sign-in.html')

@app.route('/sign-up', methods=['GET', 'POST'])
def sign_up():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')

        if not all([name, email, password]):
            flash('All fields are required', 'error')
            return render_template('sign-up.html', error='All fields are required')

        try:
            existing_user = supabase.from_('users').select('id').eq('email', email).execute()
            if existing_user.data:
                raise Exception("Email already registered")

            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

            user_data = {"name": name, "email": email, "password": password_hash}
            
            result = supabase.from_('users').insert(user_data).execute()

            if not result.data:
                raise Exception("Failed to create user account")

            user_id = result.data[0]['id']
            session['user'] = {'id': user_id, 'email': email, 'name': name}
            flash('Account created and signed in successfully!', 'success')
            return redirect(url_for('index'))
        except Exception as e:
            print("Sign-up error:", str(e))
            flash('Unable to create account. Email may already be registered.', 'error')
            return render_template('sign-up.html', error='Unable to create account')

    return render_template('sign-up.html')

@app.route('/sign-out')
def sign_out():
    session.clear()
    flash('You have been signed out.', 'success')
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
