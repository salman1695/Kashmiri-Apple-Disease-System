"""
Apple Leaf Disease Detection - Flask Web Application
Local TensorFlow Weights Model Version
"""

import os
import json
import sqlite3
import hashlib
from datetime import datetime, timedelta
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    url_for,
    redirect,
    session,
    flash
)

from werkzeug.utils import secure_filename

import io
import base64

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# ==============================================
# TENSORFLOW
# ==============================================

try:
    import tensorflow as tf

    from tensorflow.keras.models import Sequential

    from tensorflow.keras.layers import (
        Conv2D,
        MaxPooling2D,
        Flatten,
        Dense,
        Dropout
    )

    TF_AVAILABLE = True

    print("✅ TensorFlow available")

except ImportError:

    TF_AVAILABLE = False

    print("❌ TensorFlow not available")


# ==============================================
# PATH HELPER
# ==============================================

def writable(filename):

    if os.name == 'nt':
        return os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            filename
        )

    return os.path.join('/tmp', filename)


# ==============================================
# CONFIG
# ==============================================

class Config:

    IMAGE_SIZE = 224

    IMAGE_SHAPE = (224, 224)

    MODEL_DIR = 'apple_model'

    WEIGHTS_PATH = os.path.join(
        MODEL_DIR,
        'model.weights.h5'
    )

    CLASSES_PATH = 'classes.json'

    DB_PATH = writable('users.db')

    UPLOAD_FOLDER = 'static/uploads'

    ALLOWED_EXT = {
        'png',
        'jpg',
        'jpeg',
        'gif',
        'bmp',
        'webp'
    }

    MAX_FILE_SIZE = 16 * 1024 * 1024

    THRESHOLD = 60

    SECRET_KEY = os.environ.get(
        'SECRET_KEY',
        'apple-disease-secret-key'
    )

    DISEASE_INFO = {

        "BLACK APPLE ROT LEAVES": {
            "name": "Black Apple Rot",
            "description": "Fungal disease causing fruit and leaf rot",
            "symptoms": "Brown spots, decaying tissue",
            "treatment": "Apply fungicide and remove infected leaves",
            "prevention": "Proper pruning and sanitation",
            "severity": "High",
            "color": "#dc3545"
        },

        "HEALTHY LEAVES": {
            "name": "Healthy Leaf",
            "description": "Healthy apple leaf",
            "symptoms": "Green healthy appearance",
            "treatment": "No treatment needed",
            "prevention": "Maintain regular care",
            "severity": "None",
            "color": "#28a745"
        },

        "CEDAR RUST LEAVES": {
            "name": "Cedar Rust",
            "description": "Rust fungal infection",
            "symptoms": "Orange/yellow spots",
            "treatment": "Use fungicide",
            "prevention": "Avoid nearby junipers",
            "severity": "Medium",
            "color": "#ffc107"
        },

        "SCAB LEAVES": {
            "name": "Apple Scab",
            "description": "Common fungal disease",
            "symptoms": "Dark olive spots",
            "treatment": "Apply sulfur spray",
            "prevention": "Proper sanitation",
            "severity": "High",
            "color": "#17a2b8"
        }
    }


# ==============================================
# FLASK APP
# ==============================================

app = Flask(__name__)

app.secret_key = Config.SECRET_KEY

app.config['SECRET_KEY'] = Config.SECRET_KEY

app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER

app.config['MAX_CONTENT_LENGTH'] = Config.MAX_FILE_SIZE

app.config['SESSION_PERMANENT'] = True

app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)

os.makedirs(
    Config.UPLOAD_FOLDER,
    exist_ok=True
)


# ==============================================
# DATABASE
# ==============================================

class DB:

    def __init__(self):

        self.path = Config.DB_PATH

        print(f"📁 Database: {self.path}")

        self._init()

    def _init(self):

        with sqlite3.connect(self.path) as c:

            c.executescript('''

                CREATE TABLE IF NOT EXISTS users (

                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    username TEXT UNIQUE,

                    email TEXT UNIQUE,

                    password TEXT,

                    full_name TEXT,

                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

                );

                CREATE TABLE IF NOT EXISTS user_history (

                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    user_id INTEGER,

                    image_path TEXT,

                    prediction TEXT,

                    confidence REAL,

                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP

                );

            ''')

        print("✅ Database initialized")

    def _hash(self, password):

        return hashlib.sha256(
            password.encode()
        ).hexdigest()

    def register(
        self,
        username,
        email,
        password,
        full_name=""
    ):

        try:

            with sqlite3.connect(self.path) as c:

                c.execute(
                    '''
                    INSERT INTO users
                    (username,email,password,full_name)
                    VALUES (?,?,?,?)
                    ''',
                    (
                        username,
                        email,
                        self._hash(password),
                        full_name
                    )
                )

            return {
                'success': True
            }

        except sqlite3.IntegrityError:

            return {
                'success': False,
                'error': 'Username or email already exists'
            }

    def login(
        self,
        username,
        password
    ):

        with sqlite3.connect(self.path) as c:

            cur = c.execute(
                '''
                SELECT id, username, email, full_name
                FROM users
                WHERE (username=? OR email=?)
                AND password=?
                ''',
                (
                    username,
                    username,
                    self._hash(password)
                )
            )

            user = cur.fetchone()

            if not user:

                return {
                    'success': False,
                    'error': 'Invalid credentials'
                }

            return {
                'success': True,
                'user': {
                    'id': user[0],
                    'username': user[1],
                    'email': user[2],
                    'full_name': user[3]
                }
            }

    def save_history(
        self,
        user_id,
        image_path,
        prediction,
        confidence
    ):

        with sqlite3.connect(self.path) as c:

            c.execute(
                '''
                INSERT INTO user_history
                (user_id,image_path,prediction,confidence)
                VALUES (?,?,?,?)
                ''',
                (
                    user_id,
                    image_path,
                    prediction,
                    confidence
                )
            )

    def get_history(
        self,
        user_id
    ):

        with sqlite3.connect(self.path) as c:

            cur = c.execute(
                '''
                SELECT image_path,
                       prediction,
                       confidence,
                       timestamp

                FROM user_history

                WHERE user_id=?

                ORDER BY timestamp DESC
                ''',
                (user_id,)
            )

            return cur.fetchall()


db = DB()


# ==============================================
# MODEL
# ==============================================

class Model:

    def __init__(self):

        self.model = None

        self.classes = None

        self.load()

    def build_model(self):

        model = Sequential([

            Conv2D(
                32,
                (3, 3),
                activation='relu',
                input_shape=(224, 224, 3)
            ),

            MaxPooling2D(2, 2),

            Conv2D(
                64,
                (3, 3),
                activation='relu'
            ),

            MaxPooling2D(2, 2),

            Conv2D(
                128,
                (3, 3),
                activation='relu'
            ),

            MaxPooling2D(2, 2),

            Flatten(),

            Dense(
                128,
                activation='relu'
            ),

            Dropout(0.5),

            Dense(
                4,
                activation='softmax'
            )
        ])

        return model

    def load(self):

        try:

            if os.path.exists(Config.CLASSES_PATH):

                with open(Config.CLASSES_PATH) as f:

                    self.classes = json.load(f)

            else:

                self.classes = list(
                    Config.DISEASE_INFO.keys()
                )

            print("✅ Classes loaded")

            self.model = self.build_model()

            self.model.load_weights(
                Config.WEIGHTS_PATH
            )

            print("✅ Model weights loaded")

        except Exception as e:

            print(f"❌ Model load failed: {e}")

            self.model = None

    def predict(self, arr):

        try:

            preds = self.model.predict(
                arr,
                verbose=0
            )[0]

            results = {

                cls: round(
                    float(preds[i] * 100),
                    2
                )

                for i, cls in enumerate(
                    self.classes
                )
            }

            best_index = int(
                np.argmax(preds)
            )

            best_class = self.classes[
                best_index
            ]

            best_conf = float(
                preds[best_index] * 100
            )

            top3 = [

                {
                    'class': self.classes[i],

                    'confidence': float(
                        preds[i] * 100
                    ),

                    'info': Config.DISEASE_INFO.get(
                        self.classes[i],
                        {}
                    )
                }

                for i in np.argsort(preds)[-3:][::-1]
            ]

            return {

                'success': True,

                'predictions': results,

                'best_class': best_class,

                'best_confidence': best_conf,

                'is_confident': (
                    best_conf >= Config.THRESHOLD
                ),

                'top_predictions': top3
            }

        except Exception as e:

            return {
                'success': False,
                'error': str(e)
            }


ml = Model()


# ==============================================
# HELPERS
# ==============================================

def allowed(filename):

    return (
        '.' in filename and
        filename.rsplit(
            '.',
            1
        )[1].lower() in Config.ALLOWED_EXT
    )


def preprocess(filepath):

    try:

        img = Image.open(filepath)

        img = ImageOps.exif_transpose(img)

        img = img.convert('RGB')

        img = img.resize(
            Config.IMAGE_SHAPE
        )

        arr = np.array(
            img,
            dtype=np.float32
        ) / 255.0

        arr = np.expand_dims(arr, 0)

        return arr

    except Exception as e:

        print(f"Preprocess error: {e}")

        return None


def make_charts(predictions):

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5)
    )

    classes = list(
        predictions.keys()
    )

    confs = list(
        predictions.values()
    )

    colors = [

        Config.DISEASE_INFO.get(
            c,
            {}
        ).get(
            'color',
            '#999'
        )

        for c in classes
    ]

    bars = axes[0].bar(
        classes,
        confs,
        color=colors
    )

    axes[0].set_ylim([0, 100])

    axes[0].set_title(
        'Prediction Confidence'
    )

    axes[0].tick_params(
        axis='x',
        rotation=45
    )

    for bar, c in zip(bars, confs):

        axes[0].text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 1,
            f'{c:.1f}%',
            ha='center'
        )

    axes[1].pie(
        confs,
        labels=classes,
        autopct='%1.1f%%'
    )

    axes[1].axis('equal')

    buf = io.BytesIO()

    plt.tight_layout()

    plt.savefig(
        buf,
        format='png'
    )

    plt.close()

    return base64.b64encode(
        buf.getvalue()
    ).decode()


# ==============================================
# ROUTES
# ==============================================

@app.route('/')
def landing():

    return render_template(
        'landing.html'
    )


@app.route('/dashboard')
def dashboard():

    if 'user_id' not in session:

        return redirect(
            url_for('login')
        )

    return render_template(
        'index.html',
        disease_info=Config.DISEASE_INFO,
        user=session.get('user')
    )


@app.route('/login', methods=['GET', 'POST'])
def login():

    if request.method == 'POST':

        username = request.form.get(
            'username',
            ''
        )

        password = request.form.get(
            'password',
            ''
        )

        result = db.login(
            username,
            password
        )

        if result['success']:

            session['user_id'] = result['user']['id']

            session['user'] = result['user']

            return redirect(
                url_for('dashboard')
            )

        flash(
            result['error'],
            'error'
        )

    return render_template(
        'login.html'
    )


@app.route('/signup', methods=['GET', 'POST'])
def signup():

    if request.method == 'POST':

        username = request.form.get('username')

        email = request.form.get('email')

        password = request.form.get('password')

        full_name = request.form.get('full_name')

        result = db.register(
            username,
            email,
            password,
            full_name
        )

        if result['success']:

            flash(
                'Registration successful',
                'success'
            )

            return redirect(
                url_for('login')
            )

        flash(
            result['error'],
            'error'
        )

    return render_template(
        'signup.html'
    )


@app.route('/logout')
def logout():

    session.clear()

    return redirect(
        url_for('landing')
    )


@app.route('/profile')
def profile():

    if 'user_id' not in session:

        return redirect(
            url_for('login')
        )

    history = db.get_history(
        session['user_id']
    )

    return render_template(
        'profile.html',
        history=history,
        user=session.get('user')
    )


@app.route('/predict', methods=['POST'])
def predict():

    if 'user_id' not in session:

        return jsonify({
            'error': 'Please login first'
        }), 401

    try:

        if 'file' not in request.files:

            return jsonify({
                'error': 'No file uploaded'
            }), 400

        file = request.files['file']

        if file.filename == '':

            return jsonify({
                'error': 'No file selected'
            }), 400

        if not allowed(file.filename):

            return jsonify({
                'error': 'Invalid file type'
            }), 400

        filename = (
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
            f"{secure_filename(file.filename)}"
        )

        filepath = os.path.join(
            app.config['UPLOAD_FOLDER'],
            filename
        )

        file.save(filepath)

        arr = preprocess(filepath)

        if arr is None:

            return jsonify({
                'error': 'Image processing failed'
            }), 500

        result = ml.predict(arr)

        if not result['success']:

            return jsonify({
                'error': result['error']
            }), 500

        db.save_history(
            session['user_id'],
            filename,
            result['best_class'],
            result['best_confidence']
        )

        chart = make_charts(
            result['predictions']
        )

        disease_info = Config.DISEASE_INFO.get(
            result['best_class'],
            {}
        )

        return jsonify({

            'success': True,

            'filename': filename,

            'filepath': url_for(
                'static',
                filename=f'uploads/{filename}'
            ),

            'predictions': result['predictions'],

            'best_class': result['best_class'],

            'best_confidence': result['best_confidence'],

            'is_confident': result['is_confident'],

            'top_predictions': result['top_predictions'],

            'disease_info': disease_info,

            'charts': {
                'bar': chart,
                'pie': chart
            }
        })

    except Exception as e:

        return jsonify({
            'error': str(e)
        }), 500


# ==============================================
# MAIN
# ==============================================

if __name__ == '__main__':

    print(f"""

    🍎 APPLE LEAF DISEASE DETECTION
    =====================================

    🤖 TensorFlow Local Model

    📁 Weights:
    {Config.WEIGHTS_PATH}

    🌐 http://127.0.0.1:5000

    =====================================

    """)

    port = int(
        os.environ.get(
            'PORT',
            5000
        )
    )

    app.run(
        host='0.0.0.0',
        port=port,
        debug=False
    )


    