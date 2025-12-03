"""
Flask API for Botnet Detection
Provides REST API endpoints for botnet detection

Project: Botnet Detection with Machine Learning
Category: ML Projects
Developer: RSK World
Founder: Molla Samser
Designer & Tester: Rima Khatun
Contact: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Address: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
Website: https://rskworld.in
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add scripts to path
sys.path.append(str(Path(__file__).parent.parent / 'scripts'))
from detect_botnet import BotnetDetector

app = Flask(__name__)
CORS(app)

# Initialize detector
detector = None

def init_detector():
    """Initialize botnet detector"""
    global detector
    try:
        detector = BotnetDetector(model_path='data/models/', model_name='random_forest')
        return True
    except Exception as e:
        print(f"Error initializing detector: {e}")
        return False


@app.route('/')
def home():
    """API home endpoint"""
    return jsonify({
        'message': 'Botnet Detection API',
        'version': '1.0.0',
        'developer': 'RSK World',
        'endpoints': {
            '/': 'API information',
            '/health': 'Health check',
            '/detect': 'Detect botnet activities (POST)',
            '/detect/batch': 'Batch detection (POST)'
        }
    })


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'detector_loaded': detector is not None
    })


@app.route('/detect', methods=['POST'])
def detect():
    """
    Single detection endpoint
    Expects JSON with network traffic features
    """
    if detector is None:
        return jsonify({'error': 'Detector not initialized'}), 500
    
    try:
        data = request.json
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Perform detection
        results = detector.detect(df)
        
        if results is not None and len(results) > 0:
            result = results.iloc[0]
            return jsonify({
                'is_botnet': bool(result['is_botnet']),
                'botnet_probability': float(result.get('botnet_probability', 0)),
                'status': 'success'
            })
        else:
            return jsonify({'error': 'Detection failed'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/detect/batch', methods=['POST'])
def detect_batch():
    """
    Batch detection endpoint
    Expects JSON array with network traffic features
    """
    if detector is None:
        return jsonify({'error': 'Detector not initialized'}), 500
    
    try:
        data = request.json
        
        if not isinstance(data, list):
            return jsonify({'error': 'Expected array of records'}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Perform detection
        results = detector.detect(df)
        
        if results is not None:
            # Convert results to JSON
            results_dict = results.to_dict('records')
            return jsonify({
                'results': results_dict,
                'total': len(results),
                'botnet_count': int(results['is_botnet'].sum()),
                'status': 'success'
            })
        else:
            return jsonify({'error': 'Detection failed'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    print("Initializing Botnet Detection API...")
    if init_detector():
        print("Detector initialized successfully")
        print("Starting Flask server...")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("Failed to initialize detector. Please train models first.")

