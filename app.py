from flask import Flask, request, jsonify, render_template # pyright: ignore[reportMissingImports]

from flask_sqlalchemy import SQLAlchemy
import joblib
import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.database import db, SuspiciousAccount

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///soc_threat.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

# ─────────────────────────────────────────────
# FIX 1: Load only models that actually EXIST
# No more crash if a model file is missing
# ─────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models  = {}
scalers = {}

ALL_PLATFORMS = ['twitter', 'instagram', 'facebook', 'linkedin']

for p in ALL_PLATFORMS:
    model_path  = os.path.join(BASE, f'model/{p}_model.pkl')
    scaler_path = os.path.join(BASE, f'model/{p}_scaler.pkl')
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        models[p]  = joblib.load(model_path)
        scalers[p] = joblib.load(scaler_path)
        print(f'  Loaded model: {p}')
    else:
        print(f'  WARNING: Model not found for {p} — will use combined model as fallback')

# FIX 2: Always load the combined fallback model
combined_model  = None
combined_scaler = None
combined_model_path  = os.path.join(BASE, 'model/model.pkl')
combined_scaler_path = os.path.join(BASE, 'model/scaler.pkl')

if os.path.exists(combined_model_path) and os.path.exists(combined_scaler_path):
    combined_model  = joblib.load(combined_model_path)
    combined_scaler = joblib.load(combined_scaler_path)
    print('  Loaded combined fallback model')
else:
    print('  CRITICAL: Combined model also missing! Run train_model.py first.')

# FIX 3: Load features list safely
features_path = os.path.join(BASE, 'model/features.pkl')
if os.path.exists(features_path):
    features = joblib.load(features_path)
else:
    # Default features if file is missing
    features = [
        'followers_count', 'verified', 'retweet_count',
        'mention_count', 'bio_length', 'has_location',
        'follower_ratio', 'activity_score', 'profile_score'
    ]
    print('  WARNING: features.pkl not found — using default feature list')


def get_threat_level(confidence, prediction):
    """Determine threat level based on prediction and confidence."""
    if prediction == 0:
        return 'Low'
    if confidence >= 0.85:
        return 'Critical'
    elif confidence >= 0.70:
        return 'High'
    elif confidence >= 0.50:
        return 'Medium'
    else:
        return 'Low'


# FIX 4: Input validation helper
def validate_and_parse(data):
    """
    Validates all input fields.
    Returns (parsed_dict, error_message).
    If error_message is not None, validation failed.
    """
    errors = []

    def safe_float(key, default=0.0, min_val=0.0, max_val=1e9):
        val = data.get(key, default)
        try:
            val = float(val)
        except (TypeError, ValueError):
            errors.append(f"'{key}' must be a number, got: {data.get(key)}")
            return default
        if val < min_val:
            errors.append(f"'{key}' cannot be negative, got: {val}")
            return min_val
        if val > max_val:
            val = max_val  # cap silently
        return val

    def safe_bool(key):
        val = data.get(key, 0)
        try:
            val = int(float(val))
            return 1 if val >= 1 else 0
        except (TypeError, ValueError):
            errors.append(f"'{key}' must be 0 or 1, got: {data.get(key)}")
            return 0

    parsed = {
        'followers_count': safe_float('followers_count', 0, 0, 10_000_000),
        'verified':        safe_bool('verified'),
        'retweet_count':   safe_float('retweet_count',  0, 0, 1_000_000),
        'mention_count':   safe_float('mention_count',  0, 0, 1_000_000),
        'bio_length':      safe_float('bio_length',     0, 0, 10_000),
        'has_location':    safe_bool('has_location'),
    }

    # Engineered features
    parsed['follower_ratio'] = parsed['followers_count'] / (parsed['mention_count'] + 1)
    parsed['activity_score'] = parsed['retweet_count']   / (parsed['followers_count'] + 1)
    parsed['profile_score']  = (parsed['verified'] +
                                 parsed['has_location'] +
                                 (1 if parsed['bio_length'] > 0 else 0))

    if errors:
        return None, '; '.join(errors)
    return parsed, None


# Create database tables on startup
with app.app_context():
    db.create_all()


@app.route('/')
def index():
    return render_template('dashboard.html')


# FIX 5: /api/predict — full error handling + input validation
@app.route('/api/predict', methods=['POST'])
def predict():
    # Check Content-Type
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON. Set Content-Type: application/json'}), 400

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({'error': 'Invalid or empty JSON body'}), 400

    try:
        # Validate platform
        platform = str(data.get('platform', 'twitter')).lower().strip()
        if platform not in ALL_PLATFORMS:
            return jsonify({
                'error': f"Invalid platform '{platform}'. Choose from: {ALL_PLATFORMS}"
            }), 400

        # Validate username
        username = str(data.get('username', 'unknown')).strip()
        if not username:
            username = 'unknown'

        # Validate and parse all numeric inputs
        parsed, error = validate_and_parse(data)
        if error:
            return jsonify({'error': f'Input validation failed: {error}'}), 400

        # Build input dataframe
        input_df = pd.DataFrame([parsed])[features]

        # FIX 1 continued: Use platform model if available, else fallback to combined
        if platform in models and platform in scalers:
            scaler = scalers[platform]
            model  = models[platform]
        elif combined_model is not None:
            scaler = combined_scaler
            model  = combined_model
            print(f'  Using combined model as fallback for platform: {platform}')
        else:
            return jsonify({'error': 'No trained model available. Please run train_model.py first.'}), 503

        input_scaled = scaler.transform(input_df)
        prediction   = int(model.predict(input_scaled)[0])
        proba        = model.predict_proba(input_scaled)[0]
        confidence   = float(proba[prediction])
        threat_level = get_threat_level(confidence, prediction)

        # Save to database
        account = SuspiciousAccount(
            platform        = platform,
            username        = username,
            followers_count = int(parsed['followers_count']),
            verified        = int(parsed['verified']),
            retweet_count   = int(parsed['retweet_count']),
            mention_count   = int(parsed['mention_count']),
            bio_length      = int(parsed['bio_length']),
            has_location    = int(parsed['has_location']),
            prediction      = 'Suspicious' if prediction == 1 else 'Legitimate',
            confidence      = round(confidence * 100, 2),
            threat_level    = threat_level
        )
        db.session.add(account)
        db.session.commit()

        return jsonify({
            'username':     username,
            'platform':     platform,
            'prediction':   'Suspicious' if prediction == 1 else 'Legitimate',
            'confidence':   round(confidence * 100, 2),
            'threat_level': threat_level
        }), 200

    except Exception as e:
        # FIX 5: Never show raw Python error to user
        db.session.rollback()
        print(f'  ERROR in /api/predict: {e}')
        return jsonify({'error': 'An internal error occurred. Please check server logs.'}), 500


@app.route('/api/accounts', methods=['GET'])
def get_accounts():
    try:
        platform = request.args.get('platform', None)
        if platform:
            platform = platform.lower().strip()
            accounts = (SuspiciousAccount.query
                        .filter_by(platform=platform)
                        .order_by(SuspiciousAccount.created_at.desc())
                        .all())
        else:
            accounts = (SuspiciousAccount.query
                        .order_by(SuspiciousAccount.created_at.desc())
                        .all())
        return jsonify([a.to_dict() for a in accounts]), 200
    except Exception as e:
        print(f'  ERROR in /api/accounts: {e}')
        return jsonify({'error': 'Could not retrieve accounts.'}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    try:
        stats = {}
        for p in ALL_PLATFORMS:
            total      = SuspiciousAccount.query.filter_by(platform=p).count()
            suspicious = SuspiciousAccount.query.filter_by(platform=p, prediction='Suspicious').count()
            stats[p]   = {
                'total':      total,
                'suspicious': suspicious,
                'legitimate': total - suspicious
            }
        total_all      = SuspiciousAccount.query.count()
        suspicious_all = SuspiciousAccount.query.filter_by(prediction='Suspicious').count()
        stats['all']   = {
            'total':      total_all,
            'suspicious': suspicious_all,
            'legitimate': total_all - suspicious_all
        }

        # FIX: Also return which platforms have trained models loaded
        stats['loaded_models'] = list(models.keys())

        return jsonify(stats), 200
    except Exception as e:
        print(f'  ERROR in /api/stats: {e}')
        return jsonify({'error': 'Could not retrieve stats.'}), 500


@app.route('/api/clear', methods=['DELETE'])
def clear_accounts():
    try:
        SuspiciousAccount.query.delete()
        db.session.commit()
        return jsonify({'message': 'All records cleared successfully!'}), 200
    except Exception as e:
        db.session.rollback()
        print(f'  ERROR in /api/clear: {e}')
        return jsonify({'error': 'Could not clear records.'}), 500


# FIX 6: debug=False for final submission
# Change to debug=True only when you are testing locally
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
