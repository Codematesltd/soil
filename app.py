from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from supabase import create_client, SupabaseException
from dotenv import load_dotenv
import bcrypt
import pandas as pd
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
except SupabaseException as e:
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

# Load your model
model = load_model('soil_fertility_model.h5')

# Class labels
labels = sorted(['high', 'medium', 'low'])

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(180, 180))
    img_array = image.img_to_array(img) / 255.0
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
    image_url = None
    if request.method == 'POST':
        file = request.files.get('image')
        if file and file.filename:
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            img_tensor = preprocess_image(filepath)
            preds = model.predict(img_tensor)
            predicted_index = np.argmax(preds)
            prediction_result = labels[predicted_index]
            image_url = filepath

    return render_template('prediction.html', prediction=prediction_result, image_url=image_url)

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
        flash('Please sign in to access this feature.', 'warning')
        return redirect(url_for('sign_in'))

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
