"""
Apple Leaf Disease Detection - Flask Web Application
Uses Keras model first, falls back to Hugging Face if available
"""

import os
import json
import sqlite3
import hashlib
from datetime import datetime, timedelta
import numpy as np
from PIL import Image
from flask import (Flask, render_template, request, jsonify,
                   url_for, redirect, session, flash)
from werkzeug.utils import secure_filename
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Try importing tensorflow/keras
try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
    print("✅ TensorFlow/Keras available")
except ImportError:
    TF_AVAILABLE = False
    print("⚠️ TensorFlow not available")

# Try importing transformers
try:
    from transformers import AutoModelForImageClassification, AutoImageProcessor
    import torch
    HF_AVAILABLE = True
    print("✅ Hugging Face Transformers available")
except ImportError:
    HF_AVAILABLE = False
    print("⚠️ Hugging Face not available")

# ==============================================
# PATH HELPER
# ==============================================
def writable(filename):
    """Writable path: script dir on Windows, /tmp on Linux/HF"""
    if os.name == 'nt':
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    return os.path.join('/tmp', filename)

# ==============================================
# CONFIG
# ==============================================
class Config:
    IMAGE_SIZE    = 224
    IMAGE_SHAPE   = (224, 224)
    MODEL_PATH    = 'apple_model.h5'  # Your Keras model
    CLASSES_PATH  = 'classes.json'
    DB_PATH       = writable('users.db')
    UPLOAD_FOLDER = 'static/uploads'
    ALLOWED_EXT   = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    MAX_FILE_SIZE = 16 * 1024 * 1024
    THRESHOLD     = 60
    SECRET_KEY    = os.environ.get(
        'SECRET_KEY',
        'hf-apple-disease-2024-xK9!mP2@qZ#nL9$wR%vT5^yU7&cA1*bE6(dG)'
    )
    
    DISEASE_INFO = {
        "BLACK APPLE ROT LEAVES": {
            "name": "Black Apple Rot",
            "description": "Fungal disease causing fruit and leaf rot",
            "symptoms": "Brown spots, decaying tissue, foul smell",
            "treatment": "Apply fungicide, remove infected leaves, improve air circulation",
            "prevention": "Proper spacing, avoid overhead watering, regular pruning",
            "severity": "High", "color": "#dc3545"
        },
        "HEALTHY LEAVES": {
            "name": "Healthy Leaf",
            "description": "Normal healthy apple leaf",
            "symptoms": "Green color, no spots, normal texture",
            "treatment": "No treatment needed. Continue good care practices.",
            "prevention": "Regular monitoring, proper watering, balanced fertilization",
            "severity": "None", "color": "#28a745"
        },
        "CEDAR RUST LEAVES": {
            "name": "Cedar Rust",
            "description": "Fungal disease causing orange, gelatinous growths on leaves",
            "symptoms": "Orange, yellow, or red spots, leaf deformation",
            "treatment": "Apply fungicide, remove infected leaves, improve air circulation",
            "prevention": "Avoid planting susceptible species near junipers, proper spacing",
            "severity": "Medium", "color": "#ffc107"
        },
        "SCAB LEAVES": {
            "name": "Apple Scab",
            "description": "Common fungal disease of apple trees",
            "symptoms": "Olive-green to black spots, cracked leaves",
            "treatment": "Sulfur spray, remove infected leaves, fungicide application",
            "prevention": "Resistant varieties, proper sanitation, timely pruning",
            "severity": "High", "color": "#17a2b8"
        }
    }

# ==============================================
# FLASK APP
# ==============================================
app = Flask(__name__)
app.secret_key = Config.SECRET_KEY

app.config['SECRET_KEY']                 = Config.SECRET_KEY
app.config['UPLOAD_FOLDER']              = Config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH']         = Config.MAX_FILE_SIZE
app.config['SESSION_PERMANENT']          = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
app.config['SESSION_COOKIE_HTTPONLY']    = True
app.config['SESSION_COOKIE_NAME']        = 'apple_session'
app.config['SESSION_COOKIE_SAMESITE']    = 'None'
app.config['SESSION_COOKIE_SECURE']      = True

os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

# ==============================================
# DATABASE (same as before)
# ==============================================
class DB:
    def __init__(self):
        self.path = Config.DB_PATH
        print(f"📁 DB: {self.path}")
        self._init()

    def _init(self):
        try:
            with sqlite3.connect(self.path) as c:
                c.executescript('''
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        email TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL,
                        full_name TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_login TIMESTAMP,
                        is_active BOOLEAN DEFAULT 1
                    );
                    CREATE TABLE IF NOT EXISTS user_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER,
                        image_path TEXT,
                        prediction TEXT,
                        confidence REAL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users(id)
                    );
                ''')
            print("✅ DB initialized")
        except Exception as e:
            print(f"❌ DB error: {e}")
            raise

    def _hash(self, pw):
        return hashlib.sha256(pw.encode()).hexdigest()

    def register(self, username, email, password, full_name=""):
        try:
            with sqlite3.connect(self.path) as c:
                c.execute(
                    'INSERT INTO users (username,email,password,full_name) VALUES (?,?,?,?)',
                    (username, email, self._hash(password), full_name)
                )
            return {'success': True}
        except sqlite3.IntegrityError as e:
            msg = 'Username already exists' if 'username' in str(e) else 'Email already registered'
            return {'success': False, 'error': msg}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def login(self, username, password):
        try:
            with sqlite3.connect(self.path) as c:
                cur = c.execute(
                    '''SELECT id, username, email, full_name FROM users
                       WHERE (username=? OR email=?) AND password=? AND is_active=1''',
                    (username, username, self._hash(password))
                )
                user = cur.fetchone()
                if not user:
                    return {'success': False, 'error': 'Invalid username or password'}
                c.execute(
                    'UPDATE users SET last_login=CURRENT_TIMESTAMP WHERE id=?',
                    (user[0],)
                )
            return {
                'success': True,
                'user': {
                    'id':        user[0],
                    'username':  user[1],
                    'email':     user[2],
                    'full_name': user[3]
                }
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def save_history(self, user_id, image_path, prediction, confidence):
        try:
            with sqlite3.connect(self.path) as c:
                c.execute(
                    'INSERT INTO user_history (user_id,image_path,prediction,confidence) VALUES (?,?,?,?)',
                    (user_id, image_path, prediction, confidence)
                )
            return True
        except Exception as e:
            print(f"History error: {e}")
            return False

    def get_history(self, user_id, limit=20):
        try:
            with sqlite3.connect(self.path) as c:
                cur = c.execute(
                    '''SELECT image_path, prediction, confidence, timestamp
                       FROM user_history WHERE user_id=?
                       ORDER BY timestamp DESC LIMIT ?''',
                    (user_id, limit)
                )
                return cur.fetchall()
        except Exception as e:
            print(f"History fetch error: {e}")
            return []

db = DB()

# ==============================================
# MODEL - Try Keras first, then Hugging Face
# ==============================================
class Model:
    def __init__(self):
        self.model = None
        self.classes = None
        self.model_type = None
        self._load()
    
    def _load(self):
        # First try loading Keras model
        if TF_AVAILABLE and os.path.exists(Config.MODEL_PATH):
            try:
                self.model = load_model(Config.MODEL_PATH)
                print("✅ Keras model loaded from", Config.MODEL_PATH)
                self.model_type = "keras"
            except Exception as e:
                print(f"❌ Failed to load Keras model: {e}")
                self.model = None
        
        # If Keras model not available, try Hugging Face
        if self.model is None and HF_AVAILABLE:
            try:
                print("🔄 Loading Hugging Face model...")
                self.processor = AutoImageProcessor.from_pretrained("mesabo/agri-plant-disease-resnet50")
                self.model = AutoModelForImageClassification.from_pretrained("mesabo/agri-plant-disease-resnet50")
                self.model_type = "huggingface"
                print("✅ Hugging Face model loaded")
            except Exception as e:
                print(f"❌ Failed to load Hugging Face model: {e}")
                self.model = None
        
        # If no model available, use dummy
        if self.model is None:
            print("⚠️ Using dummy model")
            self.model_type = "dummy"
        
        # Load classes
        try:
            if os.path.exists(Config.CLASSES_PATH):
                with open(Config.CLASSES_PATH) as f:
                    self.classes = json.load(f)
            else:
                self.classes = list(Config.DISEASE_INFO.keys())
            print(f"✅ Classes: {self.classes}")
        except Exception as e:
            self.classes = list(Config.DISEASE_INFO.keys())
    
    def predict(self, arr):
        if self.model_type == "keras":
            return self._predict_keras(arr)
        elif self.model_type == "huggingface":
            return self._predict_huggingface(arr)
        else:
            return self._predict_dummy(arr)
    
    def _predict_keras(self, arr):
        try:
            preds = self.model.predict(arr, verbose=0)[0]
            results = {cn: round(float(preds[i]*100), 2) 
                      for i, cn in enumerate(self.classes)}
            bi = int(np.argmax(preds))
            bc = self.classes[bi]
            bconf = float(preds[bi] * 100)
            top3 = [
                {'class': self.classes[i],
                 'confidence': float(preds[i]*100),
                 'info': Config.DISEASE_INFO.get(self.classes[i], {})}
                for i in np.argsort(preds)[-3:][::-1]
            ]
            return {
                'success': True, 'predictions': results,
                'best_class': bc, 'best_confidence': bconf,
                'is_confident': bconf >= Config.THRESHOLD,
                'top_predictions': top3
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _predict_huggingface(self, arr):
        try:
            # Convert to PIL Image
            img_array = (arr[0] * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_array)
            
            # Process and predict
            inputs = self.processor(images=pil_image, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Map to your 4 classes
            probs_np = probs.numpy()[0]
            aggregated = {c: 0.0 for c in self.classes}
            
            # Simple mapping based on keywords
            for i, prob in enumerate(probs_np):
                label = self.model.config.id2label[i].lower()
                conf = float(prob * 100)
                
                if "black" in label or "rot" in label:
                    aggregated["BLACK APPLE ROT LEAVES"] = max(aggregated["BLACK APPLE ROT LEAVES"], conf)
                elif "healthy" in label:
                    aggregated["HEALTHY LEAVES"] = max(aggregated["HEALTHY LEAVES"], conf)
                elif "rust" in label:
                    aggregated["CEDAR RUST LEAVES"] = max(aggregated["CEDAR RUST LEAVES"], conf)
                elif "scab" in label:
                    aggregated["SCAB LEAVES"] = max(aggregated["SCAB LEAVES"], conf)
            
            # Normalize
            total = sum(aggregated.values())
            if total > 0:
                for k in aggregated:
                    aggregated[k] = (aggregated[k] / total) * 100
            
            best_class = max(aggregated, key=aggregated.get)
            best_conf = aggregated[best_class]
            top3 = sorted(
                [{'class': k, 'confidence': v, 'info': Config.DISEASE_INFO.get(k, {})} 
                 for k, v in aggregated.items()],
                key=lambda x: x['confidence'], reverse=True
            )[:3]
            
            return {
                'success': True, 'predictions': aggregated,
                'best_class': best_class, 'best_confidence': best_conf,
                'is_confident': best_conf >= Config.THRESHOLD,
                'top_predictions': top3
            }
        except Exception as e:
            return self._predict_dummy(arr)
    
    def _predict_dummy(self, arr):
        dummy_preds = {c: 25.0 for c in self.classes}
        return {
            'success': True,
            'predictions': dummy_preds,
            'best_class': 'HEALTHY LEAVES',
            'best_confidence': 25.0,
            'is_confident': False,
            'top_predictions': [{'class': c, 'confidence': 25.0, 'info': Config.DISEASE_INFO.get(c, {})} 
                               for c in self.classes][:3]
        }

ml = Model()

# ==============================================
# HELPERS
# ==============================================
def preprocess(path):
    try:
        img = Image.open(path).convert('RGB').resize(Config.IMAGE_SHAPE)
        arr = np.array(img, dtype=np.float32) / 255.0
        return np.expand_dims(arr, 0)
    except Exception as e:
        print(f"Preprocess error: {e}")
        return None

def allowed(filename):
    return ('.' in filename and
            filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXT)


def make_charts(predictions):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    classes = list(predictions.keys())
    confs   = list(predictions.values())
    colors  = [Config.DISEASE_INFO.get(c, {}).get('color', '#999') for c in classes]

    # Bar
    bars = axes[0].bar(classes, confs, color=colors, alpha=0.8)
    axes[0].set_ylabel('Confidence (%)')
    axes[0].set_title('Disease Prediction Confidence')
    axes[0].set_ylim([0, 100])
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].axhline(y=Config.THRESHOLD, color='red', linestyle='--',
                    alpha=0.5, label=f'Threshold ({Config.THRESHOLD}%)')
    for bar, c in zip(bars, confs):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                     f'{c:.1f}%', ha='center', va='bottom', fontsize=9)
    axes[0].legend()

    # Pie
    filtered = {k: v for k, v in predictions.items() if v > 5} or predictions
    total = sum(filtered.values())
    if total < 100:
        filtered['Other'] = round(100-total, 2)
    pc = [Config.DISEASE_INFO.get(k, {'color': '#999'})['color'] for k in filtered]
    axes[1].pie(filtered.values(), labels=filtered.keys(),
                autopct='%1.1f%%', colors=pc, startangle=90)
    axes[1].set_title('Prediction Distribution')
    axes[1].axis('equal')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode()

# ==============================================
# IMAGE PROCESSING (Supports ALL formats)
# ==============================================
def convert_to_rgb(image):
    """Convert any image to RGB format"""
    try:
        if image.mode in ('RGBA', 'LA', 'P'):
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            if image.mode == 'RGBA':
                rgb_image.paste(image, mask=image.split()[-1])
            else:
                rgb_image.paste(image)
            return rgb_image
        elif image.mode != 'RGB':
            return image.convert('RGB')
        return image
    except Exception as e:
        print(f"Convert error: {e}")
        return image.convert('RGB')

def preprocess_image(filepath):
    """Enhanced preprocessing that handles ALL image types"""
    try:
        # Open image with PIL
        img = Image.open(filepath)
        
        # Fix orientation based on EXIF data
        try:
            img = ImageOps.exif_transpose(img)
        except:
            pass
        
        # Convert to RGB
        img = convert_to_rgb(img)
        
        # Resize image (maintain aspect ratio then crop to center)
        img.thumbnail((Config.IMAGE_SIZE, Config.IMAGE_SIZE), Image.Resampling.LANCZOS)
        
        # Create square image with padding if needed
        new_img = Image.new('RGB', (Config.IMAGE_SIZE, Config.IMAGE_SIZE), (255, 255, 255))
        x_offset = (Config.IMAGE_SIZE - img.width) // 2
        y_offset = (Config.IMAGE_SIZE - img.height) // 2
        new_img.paste(img, (x_offset, y_offset))
        
        # Convert to array and normalize
        arr = np.array(new_img, dtype=np.float32) / 255.0
        return np.expand_dims(arr, 0)
        
    except UnidentifiedImageError:
        print(f"Cannot identify image file: {filepath}")
        return None
    except Exception as e:
        print(f"Preprocess error: {e}")
        return None

def allowed_file(filename):
    """Check if file extension is allowed"""
    if not filename or '.' not in filename:
        return True  # Let PIL handle unknown files
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in Config.ALLOWED_EXT or True  # Allow all files, PIL will validate

def validate_image(file_stream):
    """Validate if file is actually an image"""
    try:
        img = Image.open(file_stream)
        img.verify()
        file_stream.seek(0)
        return True
    except:
        file_stream.seek(0)
        return False
    
# ==============================================
# ROUTES (same as before)
# ==============================================

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash('Please login first', 'warning')
        return redirect(url_for('login'))
    return render_template('index.html',
                           disease_info=Config.DISEASE_INFO,
                           threshold=Config.THRESHOLD,
                           user=session.get('user'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        if not username or not password:
            flash('Username and password are required!', 'error')
            return render_template('login.html')

        result = db.login(username, password)

        if result['success']:
            session.clear()
            session.permanent = True
            session['user_id']  = result['user']['id']
            session['username'] = result['user']['username']
            session['user']     = result['user']
            flash('Login successful!', 'success')
            next_page = request.args.get('next')
            return redirect(next_page or url_for('dashboard'))

        flash(result['error'], 'error')

    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username  = request.form.get('username', '').strip()
        email     = request.form.get('email', '').strip()
        password  = request.form.get('password', '').strip()
        confirm   = request.form.get('confirm_password', '').strip()
        full_name = request.form.get('full_name', '').strip()

        if not all([username, email, password]):
            flash('All fields are required!', 'error')
            return render_template('signup.html')
        if password != confirm:
            flash('Passwords do not match!', 'error')
            return render_template('signup.html')
        if len(password) < 6:
            flash('Password must be at least 6 characters!', 'error')
            return render_template('signup.html')

        result = db.register(username, email, password, full_name)
        if result['success']:
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        flash(result['error'], 'error')

    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('landing'))

@app.route('/profile')
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    history_raw = db.get_history(session['user_id'])
    
    # Convert history data to proper types
    formatted_history = []
    for item in history_raw:
        # item[0] = image_path, item[1] = prediction, item[2] = confidence, item[3] = timestamp
        
        # Fix confidence - handle bytes and string directly
        confidence_val = item[2]
        if isinstance(confidence_val, bytes):
            # Try to convert bytes to float directly
            try:
                # Convert bytes to string and then to float
                confidence_val = float(confidence_val.decode('utf-8', errors='ignore').strip())
            except:
                # If that fails, try to interpret bytes as float
                try:
                    import struct
                    confidence_val = struct.unpack('f', confidence_val)[0]
                except:
                    confidence_val = 50.0  # Default value
        elif isinstance(confidence_val, str):
            try:
                confidence_val = float(confidence_val)
            except:
                confidence_val = 50.0
        else:
            try:
                confidence_val = float(confidence_val)
            except:
                confidence_val = 50.0
        
        # Fix timestamp
        timestamp_val = item[3]
        if isinstance(timestamp_val, bytes):
            timestamp_val = timestamp_val.decode('utf-8', errors='ignore')
        elif timestamp_val is None:
            timestamp_val = str(datetime.now())
        
        formatted_history.append({
            'image_path': item[0],
            'prediction': item[1],
            'confidence': confidence_val,
            'timestamp': timestamp_val
        })
    
    return render_template('profile.html',
                           user=session.get('user'), 
                           history=formatted_history)

@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    rows = db.get_history(session['user_id'])
    formatted = [{
        'filename':   os.path.basename(r[0]),
        'path':       url_for('static', filename='uploads/' + r[0]),
        'prediction': r[1],
        'confidence': r[2],
        'date':       r[3] if isinstance(r[3], str) else str(r[3])
    } for r in rows]
    return render_template('history.html',
                           images=formatted, user=session.get('user'))

@app.route('/about')
def about():
    return render_template('about.html',
                           disease_info=Config.DISEASE_INFO,
                           class_names=ml.classes,
                           user=session.get('user'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return jsonify({'error': 'Please login first'}), 401
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['file']
        if not file.filename:
            return jsonify({'error': 'No file selected'}), 400
        if not allowed(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400

        fname = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        file.save(filepath)

        arr = preprocess(filepath)
        if arr is None:
            return jsonify({'error': 'Error processing image'}), 500

        result = ml.predict(arr)
        if not result['success']:
            return jsonify({'error': result['error']}), 500

        db.save_history(session['user_id'], fname,
                        result['best_class'], result['best_confidence'])

        chart = make_charts(result['predictions'])
        disease_info = Config.DISEASE_INFO.get(
            result['best_class'], Config.DISEASE_INFO['HEALTHY LEAVES'])

        return jsonify({
            'success': True,
            'filename': fname,
            'filepath': url_for('static', filename=f'uploads/{fname}'),
            'predictions': result['predictions'],
            'best_class': result['best_class'],
            'best_confidence': result['best_confidence'],
            'is_confident': result['is_confident'],
            'top_predictions': result['top_predictions'],
            'disease_info': disease_info,
            'charts': {'bar': chart, 'pie': chart}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==============================================
# MAIN
# ==============================================
if __name__ == '__main__':
    print(f"""
    🍎 APPLE LEAF DISEASE DETECTION
    ================================
    🤖 Model Type: {ml.model_type}
    📁 Model File: {Config.MODEL_PATH if os.path.exists(Config.MODEL_PATH) else 'Not found'}
    📁 DB: {Config.DB_PATH}
    🌐 http://127.0.0.1:5000
    ================================
    """)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)