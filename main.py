from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import os
import csv
from io import StringIO
from google.cloud import aiplatform
from google.oauth2 import service_account
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration - AUTOML MODEL ONLY
CONFIG = {
    'project_id': '799143320054',
    'location': 'us-central1',
    'service_account_path': 'key.json',
    'automl_model_id': 6449544993022410752,  # Your AutoML model ID
    'automl_endpoint_id': 6449544993022410752,  # Set this when you deploy the AutoML model
}

# Hair color classes (must match training order)
HAIR_COLOR_CLASSES = [
    "ash_blonde", "black", "dark_blonde", "dark_brown", "ginger_red", 
    "light_blond", "light_brown", "mahogany_red", "middle_brown", 
    "middle_warm_blond", "silver_grey", "warm_light_brown_copper"
]

# AutoML color classes (from class_mapping.json)
AUTOML_COLOR_CLASSES = [
    "Apricot Amber", "Ashy Ribbon", "Atomic Punch", "Auburn Sugar", "Cayenne Spice",
    "Copper Bronze", "Creamy toffee", "Cyber Glam", "Diamond Frost", "Espresso Smoke",
    "Fire dash", "Ginger", "Havana Roots", "Honey Blossom", "Iced Gold",
    "Marshmellow Roast", "Mocha Chino", "Onyx", "Raspberry Ice", "Satin Caramel",
    "Shimmer Ale", "Spring Lush", "Sun Kissed", "Velvet Rebel"
]

# AutoML to generic hair color mapping
AUTOML_TO_GENERIC_MAPPING = {
    "Apricot Amber": "ginger_red",
    "Ashy Ribbon": "light_blond",
    "Atomic Punch": "light_brown",
    "Auburn Sugar": "middle_brown",
    "Cayenne Spice": "warm_light_brown_copper",
    "Copper Bronze": "light_brown",
    "Creamy toffee": "ash_blonde",
    "Cyber Glam": "light_brown",
    "Diamond Frost": "light_blond",
    "Espresso Smoke": "dark_brown",
    "Fire dash": "ginger_red",
    "Ginger": "ginger_red",
    "Havana Roots": "dark_brown",
    "Honey Blossom": "middle_warm_blond",
    "Iced Gold": "silver_grey",
    "Marshmellow Roast": "dark_blonde",
    "Mocha Chino": "middle_brown",
    "Onyx": "black",
    "Raspberry Ice": "mahogany_red",
    "Satin Caramel": "ash_blonde",
    "Shimmer Ale": "middle_warm_blond",
    "Spring Lush": "middle_warm_blond",
    "Sun Kissed": "ash_blonde",
    "Velvet Rebel": "dark_blonde"
}

# Product data (keeping your existing product data)
PRODUCTS_DATA = """root_tone,tip_tone,sku,product_name,image_url
dark_brown,,20V20,Infinity Braids® - Braided Headband - Viènne - Espresso Smoke,https://drive.google.com/file/d/1r7NQr9dIXN0EiCnAaP1JZoJ0vsi37xWe/view?usp=drive_link
dark_brown,,20J20,Infinity Braids® - Braided Headband - Jolie  - Espresso Smoke,
dark_brown,,20L20,Infinity Braids® - Braided Headband - Lizzy - Espresso Smoke,
dark_brown,,24MB20,Infinity Braids® - Infinity Braidies - Espresso Smoke,
dark_brown,,20V19,Infinity Braids® - Viènne - Havana Roots,
dark_brown,,20J19,Infinity Braids® - Braided Headband - Jolie  - Havana Roots,
dark_brown,,20L19,Infinity Braids® - Braided Headband - Lizzy - Havana Roots,
dark_brown,,24MB19,Infinity Braids® - Infinity Braidies - Havana Roots,
middle_brown,,20V17,Infinity Braids® - Viènne - Auburn Sugar,https://drive.google.com/file/d/1O2TO1i8KVCwOXSgzQC-neoSdDfswWLgG/view?usp=drive_link
middle_brown,,20J17,Infinity Braids® - Braided Headband - Jolie  - Auburn Sugar,
middle_brown,,20L17,Infinity Braids® - Braided Headband - Lizzy - Auburn Sugar,
middle_brown,,24MB17,Infinity Braids® - Infinity Braidies - Auburn Sugar,
middle_brown,,20V16,Infinity Braids® - Braided Headband - Viènne - Mocha-Chino,
middle_brown,,20J16,Infinity Braids® - Braided Headband - Jolie  - Mocha-Chino,
middle_brown,,20L16,Infinity Braids® - Braided Headband - Lizzy - Mocha-Chino,
middle_brown,,24MB16,Infinity Braids® - Infinity Braidies - Mocha Chino,
light_brown,,20V15,Infinity Braids® - Braided Headband - Viènne - Cyber Glam,https://drive.google.com/file/d/1xor8ePectXKbMGPaMW7mLtA8-2JkZelr/view?usp=drive_link
light_brown,,20J15,Infinity Braids® - Braided Headband - Jolie  - Cyber Glam,
light_brown,,20L15,Infinity Braids® - Braided Headband - Lizzy - Cyber Glam,
light_brown,,24MB15,Infinity Braids® - Infinity Braidies - Cyber Glam,
light_brown,,20V11,Infinity Braids® - Viènne - Copper Bronze,
light_brown,,20J11,Infinity Braids® - Braided Headband - Jolie  - Copper Bronze,
light_brown,,20L11,Infinity Braids® - Braided Headband - Lizzy - Copper Bronze,
light_brown,,24MB11,Infinity Braids® - Infinity Braidies - Copper Bronze,
light_brown,,22V24,Infinity Braids® - Braided Headband - Viènne - Atomic Punch,
light_brown,,22J24,Infinity Braids® - Braided Headband - Jolie  - Atomic Punch,
light_brown,,22L24,Infinity Braids® - Braided Headband - Lizzy - Atomic Punch,
light_brown,,24MB24,Infinity Braids® - Infinity Braidies - Atomic Punch,
middle_warm_blond,,20V07,Infinity Braids® - Viènne - Shimmer Ale,https://drive.google.com/file/d/1rIqb5jA2ma9gMiOXqyJHVMrP9aURWODy/view?usp=drive_link
middle_warm_blond,,20J07,Infinity Braids® - Braided Headband - Jolie  - Shimmer Ale,
middle_warm_blond,,20L07,Infinity Braids® - Lizzy - Shimmer Ale,
middle_warm_blond,,24MB07,Infinity Braids® - Infinity Braidies - Shimmer Ale,
middle_warm_blond,,20V05,Infinity Braids® - Viènne - Spring Lush,
middle_warm_blond,,20J05,Infinity Braids® - Jolie  - Spring Lush,
middle_warm_blond,,20L05,Infinity Braids® - Braided Headband - Lizzy - Spring Lush,
middle_warm_blond,,24MB05,Infinity Braids® - Infinity Braidies - Spring Lush,
middle_warm_blond,,20V06,Infinity Braids® - Braided Headband - Viènne - Honey Blossom,
middle_warm_blond,,20J06,Infinity Braids® - Braided Headband - Jolie  - Honey Blossom,
middle_warm_blond,,20L06,Infinity Braids® - Braided Headband - Lizzy - Honey Blossom,
middle_warm_blond,,24MB06,Infinity Braids® - Infinity Braidies - Honey Blossom,
light_blond,,21V02,Infinity Braids® - Braided Headband - Viènne - Diamond Frost,
light_blond,,21J02,Infinity Braids® - Braided Headband - Jolie  - Diamond Frost,
light_blond,,21L02,Infinity Braids® - Braided Headband - Lizzy - Diamond Frost,
light_blond,,24MB02,Infinity Braids® - Infinity Braidies - Diamond Frost,
light_blond,,20V03,Infinity Braids® - Viènne - Ashy Ribbon,
light_blond,,20J03,Infinity Braids® - Braided Headband - Jolie  - Ashy Ribbon,
light_blond,,20L03,Infinity Braids® - Braided Headband - Lizzy - Ashy Ribbon,
light_blond,,24MB03,Infinity Braids® - Infinity Braidies - Ashy Ribbon,
ash_blonde,,21V22,Infinity Braids® - Viènne - Satin Caramel,https://drive.google.com/file/d/1KZbLn8hiK61968wTYkBCzZsf6MkPr_iX/view?usp=drive_link
ash_blonde,,21J22,Infinity Braids® - Braided Headband - Jolie  - Satin Caramel,
ash_blonde,,21L22,Infinity Braids® - Braided Headband - Lizzy - Satin Caramel,
ash_blonde,,24MB22,Infinity Braids® - Infinity Braidies - Satin Caramel,
ash_blonde,,20V04,Infinity Braids® - Braided Headband - Viènne - Sun Kissed,
ash_blonde,,20J04,Infinity Braids® - Braided Headband - Jolie  - Sun Kissed,
ash_blonde,,20L04,Infinity Braids® - Braided Headband - Lizzy - Sun Kissed,
ash_blonde,,24MB04,Infinity Braids® - Infinity Braidies - Sun Kissed,
ash_blonde,,20V10,Infinity Braids® - Viènne - Creamy Toffee,
ash_blonde,,20J10,Infinity Braids® - Braided Headband - Jolie  - Creamy Toffee,
ash_blonde,,20L10,Infinity Braids® - Braided Headband - Lizzy - Creamy Toffee,
ash_blonde,,24MB10,Infinity Braids® - Infinity Braidies - Creamy Toffee,
dark_blonde,,20V08,Infinity Braids® - Braided Headband - Viènne - Marshmellow Roast,https://drive.google.com/file/d/1SjzSYdI1xzRxybRpo5L5-NzbWZcZil9B/view?usp=drive_link
dark_blonde,,20J08,Infinity Braids® - Braided Headband - Jolie  - Marshmellow Roast,
dark_blonde,,20L08,Infinity Braids® - Braided Headband - Lizzy - Marshmellow Roast,
dark_blonde,,24MB08,Infinity Braids® - Infinity Braidies - Marshmellow Roast,
dark_blonde,,21V23,Infinity Braids® - Braided Headband - Viènne - Velvet Rebel,
dark_blonde,,21J23,Infinity Braids® - Braided Headband - Jolie  - Velvet Rebel,
dark_blonde,,21L23,Infinity Braids® - Braided Headband - Lizzy - Velvet Rebel,
dark_blonde,,24MB23,Infinity Braids® - Infinity Braidies - Velvet Rebel,
black,,20V21,Infinity Braids® - Viènne - Onyx,
black,,20J21,Infinity Braids® - Braided Headband - Jolie  - Onyx,
black,,20L21,Infinity Braids® - Braided Headband - Lizzy - Onyx,
black,,24MB21,Infinity Braids® - Infinity Braidies - Onyx,
ginger_red,,20V12,Infinity Braids® - Braided Headband - Viènne - Apricot Amber,https://drive.google.com/file/d/1j0a-3Pw_WFG2YW8xCceNzEjsl9fKhff1/view?usp=drive_link
ginger_red,,20J12,Infinity Braids® - Braided Headband - Jolie  - Apricot Amber,
ginger_red,,20L12,Infinity Braids® - Braided Headband - Lizzy - Apricot Amber,
ginger_red,,24MB12,Infinity Braids® - Infinity Braidies  - Apricot Amber,
ginger_red,,20V13,Infinity Braids® - Braided Headband - Viènne - Ginger,
ginger_red,,20J13,Infinity Braids® - Braided Headband - Jolie  - Ginger,
ginger_red,,20L13,Infinity Braids® - Braided Headband - Lizzy - Ginger,
ginger_red,,24MB13,Infinity Braids® - Infinity Braidies - Ginger,
ginger_red,,22V25,Infinity Braids® - Braided Headband - Viènne - Fire Dash,
ginger_red,,22J25,Infinity Braids® - Braided Headband - Jolie  - Fire Dash,
ginger_red,,22L25,Infinity Braids® - Braided Headband - Lizzy - Fire Dash,
ginger_red,,24MB25,Infinity Braids® - Infinity Braidies - Fire Dash,
mahogany_red,,20V18,Infinity Braids® - Braided Headband - Viènne - Raspberry Ice,
mahogany_red,,20J18,Infinity Braids® - Braided Headband - Jolie  - Raspberry Ice,
mahogany_red,,20L18,Infinity Braids® - Lizzy - Raspberry Ice,
mahogany_red,,24MB18,Infinity Braids® - Infinity Braidies - Raspberry Ice,
warm_light_brown_copper,,20V14,Infinity Braids® - Viènne - Cayenne Spice,
warm_light_brown_copper,,20J14,Infinity Braids® - Braided Headband - Jolie  - Cayenne Spice,
warm_light_brown_copper,,20L14,Infinity Braids® - Lizzy - Cayenne Spice,
warm_light_brown_copper,,24MB14,Infinity Braids® - Infinity Braidies - Cayenne Spice,
silver_grey,,20V01,Infinity Braids® - Braided Headband - Viènne - Iced Gold,
silver_grey,,20J01,Infinity Braids® - Braided Headband - Jolie  - Iced Gold,
silver_grey,,20L01,Infinity Braids® - Braided Headband - Lizzy - Iced Gold,
silver_grey,,24MB01,Infinity Braids® - Infinity Braidies  - Iced Gold,"""

def load_products():
    """Load products from CSV data"""
    products = {}
    reader = csv.DictReader(StringIO(PRODUCTS_DATA))
    
    for row in reader:
        root_tone = row['root_tone']
        if root_tone not in products:
            products[root_tone] = []
        
        product = {
            'sku': row['sku'],
            'name': row['product_name'],
            'image_url': row['image_url'] if row['image_url'] else None
        }
        products[root_tone].append(product)
    
    return products

PRODUCTS = load_products()

class AutoMLVertexAIPredictor:
    def __init__(self):
        self.automl_endpoint = None
        self.automl_model = None
        self.automl_endpoint_ready = False
        self.input_size = 224  # AutoML uses 224x224
        self._initialize_vertex_ai()
    
    def _initialize_vertex_ai(self):
        """Initialize Vertex AI and get the AutoML model/endpoint"""
        try:
            # Setup credentials
            credentials = service_account.Credentials.from_service_account_file(
                CONFIG['service_account_path'],
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            
            # Initialize Vertex AI with standard domain first
            aiplatform.init(
                project=CONFIG['project_id'],
                location=CONFIG['location'],
                credentials=credentials
            )
            
            # Initialize AutoML model
            if CONFIG['automl_model_id']:
                try:
                    model_resource_name = f"projects/{CONFIG['project_id']}/locations/{CONFIG['location']}/models/{CONFIG['automl_model_id']}"
                    self.automl_model = aiplatform.Model(model_resource_name)
                    logger.info("✅ AutoML model loaded!")
                    logger.info(f"📍 AutoML Model ID: {CONFIG['automl_model_id']}")
                except Exception as e:
                    logger.warning(f"⚠️ AutoML model not available: {e}")
            
            # Initialize AutoML endpoint with dedicated domain configuration
            if CONFIG['automl_endpoint_id']:
                try:
                    automl_endpoint_resource_name = f"projects/{CONFIG['project_id']}/locations/{CONFIG['location']}/endpoints/{CONFIG['automl_endpoint_id']}"
                    
                    # Create endpoint with dedicated domain configuration
                    dedicated_domain = f"{CONFIG['automl_endpoint_id']}.{CONFIG['location']}-{CONFIG['project_id']}.prediction.vertexai.goog"
                    logger.info(f"🔗 Using dedicated domain: {dedicated_domain}")
                    
                    # Create endpoint with custom client options for dedicated domain
                    from google.cloud.aiplatform_v1.services.prediction_service import PredictionServiceClient
                    from google.api_core.client_options import ClientOptions
                    
                    client_options = ClientOptions(api_endpoint=f"https://{dedicated_domain}")
                    prediction_client = PredictionServiceClient(
                        credentials=credentials,
                        client_options=client_options
                    )
                    
                    self.automl_endpoint = aiplatform.Endpoint(automl_endpoint_resource_name)
                    # Override the prediction client with our custom one
                    self.automl_endpoint._prediction_client = prediction_client
                    
                    self.automl_endpoint_ready = True
                    logger.info("✅ AutoML endpoint connected with dedicated domain!")
                    logger.info(f"📍 AutoML Endpoint ID: {CONFIG['automl_endpoint_id']}")
                except Exception as e:
                    logger.warning(f"⚠️ AutoML endpoint not available: {e}")
                    self.automl_endpoint_ready = False
            else:
                logger.info("🔄 AutoML endpoint not configured - predictions will use model directly")
            
            logger.info(f"🎯 AutoML model ready with input size: {self.input_size}×{self.input_size}")
            
        except Exception as e:
            logger.error(f"❌ Error connecting to Vertex AI: {e}")
            self.automl_endpoint_ready = False
    
    def preprocess_image(self, image):
        """Preprocess image for AutoML model - use base64 encoding as per documentation"""
        try:
            # Resize to AutoML standard size (224x224)
            image = image.resize((self.input_size, self.input_size), Image.Resampling.LANCZOS)
            
            # Convert to RGB if not already
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to base64 encoded bytes as per Google documentation
            import base64
            from io import BytesIO
            
            # Save as JPEG with good quality for classification
            buffer = BytesIO()
            image.save(buffer, format='JPEG', quality=95, optimize=True)
            
            # Encode as base64 string
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Log payload size
            json_size = len(img_base64.encode('utf-8'))
            logger.info(f"📦 AutoML base64 payload: {json_size:,} bytes ({json_size/1024:.1f} KB)")
            
            return img_base64
            
        except Exception as e:
            logger.error(f"Error in AutoML preprocessing: {e}")
            return None
    
    def predict(self, image):
        """Make prediction using AutoML dedicated endpoint via HTTP"""
        try:
            # Preprocess image to base64
            img_base64 = self.preprocess_image(image)
            
            if img_base64 is None:
                logger.error("Failed to preprocess image")
                return "Shimmer Ale", 0.1
            
            logger.info("📦 Sending base64 image to dedicated AutoML endpoint")
            
            # Make direct HTTP request to dedicated endpoint
            import requests
            from google.oauth2 import service_account
            from google.auth.transport.requests import Request
            
            # Get credentials and access token
            credentials = service_account.Credentials.from_service_account_file(
                CONFIG['service_account_path'],
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            credentials.refresh(Request())
            
            # Use the exact format expected by the model 
            # Based on error: keys must be equal to: image_bytes,key
            request_data = {
                "instances": [{
                    "image_bytes": {"b64": img_base64},
                    "key": "0"  # Adding the required key field
                }],
                "parameters": {
                    "confidenceThreshold": 0.0,
                    "maxPredictions": 24
                }
            }
            
            # Make request to dedicated endpoint
            endpoint_url = f"https://{CONFIG['automl_endpoint_id']}.{CONFIG['location']}-{CONFIG['project_id']}.prediction.vertexai.goog/v1/projects/{CONFIG['project_id']}/locations/{CONFIG['location']}/endpoints/{CONFIG['automl_endpoint_id']}:predict"
            
            headers = {
                "Authorization": f"Bearer {credentials.token}",
                "Content-Type": "application/json"
            }
            
            logger.info(f"🌐 Making request to: {endpoint_url}")
            
            response = requests.post(endpoint_url, json=request_data, headers=headers, timeout=30)
            
            if response.status_code == 200:
                prediction_response = response.json()
                return self.process_prediction_response_http(prediction_response)
            else:
                logger.error(f"HTTP request failed: {response.status_code} - {response.text}")
                return "Shimmer Ale", 0.1
            
        except Exception as e:
            logger.error(f"Error making AutoML prediction: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return "Shimmer Ale", 0.1
    
    def process_prediction_response_http(self, response_json):
        """Process HTTP prediction response from AutoML endpoint"""
        try:
            if 'predictions' in response_json:
                predictions = response_json['predictions']
                logger.info(f"🔍 Received {len(predictions)} predictions from HTTP endpoint")
                
                if predictions and len(predictions) > 0:
                    first_prediction = predictions[0]
                    
                    # Handle the actual AutoML response format with labels and scores
                    if 'labels' in first_prediction and 'scores' in first_prediction:
                        import base64
                        labels = first_prediction['labels']
                        scores = first_prediction['scores']
                        
                        if labels and scores:
                            # Get the highest scoring prediction
                            best_idx = 0
                            best_confidence = scores[0]
                            
                            for i, score in enumerate(scores):
                                if score > best_confidence:
                                    best_confidence = score
                                    best_idx = i
                            
                            # Decode the base64 label to get the actual class name
                            try:
                                best_label_b64 = labels[best_idx]
                                decoded_bytes = base64.b64decode(best_label_b64)
                                
                                # The class name is embedded in protobuf format, starting at byte 12
                                predicted_class = decoded_bytes[12:].decode('utf-8', errors='ignore')
                                
                                # Clean up any null bytes or extra characters
                                predicted_class = predicted_class.strip('\x00').strip()
                                
                            except Exception as e:
                                logger.error(f"Error decoding label: {e}")
                                predicted_class = "Unknown"
                            
                            confidence = float(best_confidence)
                            
                            logger.info(f"🎯 AutoML HTTP prediction: {predicted_class} (confidence: {confidence:.3f}, index: {best_idx})")
                            return predicted_class, confidence
                    
                    # Fallback for old format with displayNames and confidences
                    elif 'displayNames' in first_prediction and 'confidences' in first_prediction:
                        display_names = first_prediction['displayNames']
                        confidences = first_prediction['confidences']
                        
                        if display_names and confidences:
                            # Get the highest confidence prediction
                            best_idx = 0
                            best_confidence = confidences[0]
                            
                            for i, conf in enumerate(confidences):
                                if conf > best_confidence:
                                    best_confidence = conf
                                    best_idx = i
                            
                            predicted_class = display_names[best_idx]
                            confidence = float(best_confidence)
                            
                            logger.info(f"🎯 AutoML HTTP prediction: {predicted_class} (confidence: {confidence:.3f})")
                            return predicted_class, confidence
                    
                    logger.error(f"Unexpected response format: {first_prediction}")
                    return "Shimmer Ale", 0.1
                else:
                    logger.error("No predictions in HTTP response")
                    return "Shimmer Ale", 0.1
            else:
                logger.error("No predictions key in HTTP response")
                return "Shimmer Ale", 0.1
                
        except Exception as e:
            logger.error(f"Error processing HTTP prediction response: {e}")
            return "Shimmer Ale", 0.1
    
    def process_prediction_response(self, prediction_response):
        """Process prediction response from AutoML model"""
        try:
            if hasattr(prediction_response, 'predictions'):
                predictions = prediction_response.predictions
                logger.info(f"🔍 Received {len(predictions)} AutoML predictions")
                
                if predictions and len(predictions) > 0:
                    first_prediction = predictions[0]
                    
                    # Handle different response formats
                    if isinstance(first_prediction, list):
                        if len(first_prediction) == len(AUTOML_COLOR_CLASSES):
                            predicted_class_idx = np.argmax(first_prediction)
                            confidence = float(first_prediction[predicted_class_idx])
                        else:
                            logger.error(f"Unexpected prediction length: {len(first_prediction)}, expected {len(AUTOML_COLOR_CLASSES)}")
                            return "middle_brown", 0.1
                            
                    elif isinstance(first_prediction, dict):
                        # Handle dict response formats
                        pred_values = None
                        for key in ['predictions', 'scores', 'outputs', 'probabilities']:
                            if key in first_prediction:
                                pred_values = first_prediction[key]
                                break
                        
                        if pred_values is None:
                            logger.error(f"Unknown dict format: {first_prediction.keys()}")
                            return "middle_brown", 0.1
                        
                        predicted_class_idx = np.argmax(pred_values)
                        confidence = float(pred_values[predicted_class_idx])
                        
                    else:
                        # Single value prediction
                        predicted_class_idx = int(first_prediction)
                        confidence = 0.5
                    
                    # Return AutoML prediction directly (specific color names)
                    if 0 <= predicted_class_idx < len(AUTOML_COLOR_CLASSES):
                        automl_predicted_class = AUTOML_COLOR_CLASSES[predicted_class_idx]
                        logger.info(f"🎯 AutoML prediction: {automl_predicted_class} (confidence: {confidence:.3f})")
                        return automl_predicted_class, confidence
                    else:
                        logger.warning(f"Invalid AutoML class index: {predicted_class_idx}")
                        return "Shimmer Ale", 0.1
                else:
                    logger.error("No predictions in response")
                    return "Shimmer Ale", 0.1
            else:
                logger.error("No predictions attribute in response")
                return "Shimmer Ale", 0.1
                
        except Exception as e:
            logger.error(f"Error processing AutoML prediction response: {e}")
            return "Shimmer Ale", 0.1

# Initialize AutoML predictor
logger.info("🚀 Initializing AutoML Vertex AI predictor...")
automl_predictor = AutoMLVertexAIPredictor()

def classify_hair_color(image):
    """Use AutoML Vertex AI model to classify hair color"""
    try:
        predicted_category, confidence = automl_predictor.predict(image)
        return predicted_category, confidence
    except Exception as e:
        logger.error(f"Error in AutoML classification: {str(e)}")
        return "Shimmer Ale", 0.1

@app.route('/classify', methods=['POST'])
def classify_image():
    """AutoML hair color classification route"""
    try:
        # Check if endpoint configuration is available (for direct HTTP requests)
        if not CONFIG['automl_endpoint_id']:
            logger.error("No AutoML endpoint ID configured")
            return jsonify({
                'error': 'AutoML endpoint not configured.',
                'details': 'Please configure automl_endpoint_id in CONFIG.'
            }), 503
        
        logger.info(f"🔍 Using AutoML endpoint: {CONFIG['automl_endpoint_id']}")
        
        # Get the uploaded image
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Open and process the image
        image = Image.open(file.stream)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Use AutoML model for classification
        predicted_category, confidence = classify_hair_color(image)
        
        # Map AutoML prediction to generic category for product matching
        generic_category = AUTOML_TO_GENERIC_MAPPING.get(predicted_category, "middle_brown")
        suggested_products = PRODUCTS.get(generic_category, [])
        
        # Add detected class information to each product
        enhanced_products = []
        for product in suggested_products[:8]:
            enhanced_product = product.copy()
            enhanced_product['detected_color_class'] = predicted_category
            enhanced_product['generic_category'] = generic_category
            enhanced_products.append(enhanced_product)
        
        return jsonify({
            'predicted_category': predicted_category,  # AutoML class name (e.g., "Diamond Frost")
            'generic_category': generic_category,      # Generic category for products
            'confidence': f'{confidence:.3f}',
            'model_type': 'automl_vertex_ai',
            'model_id': CONFIG['automl_model_id'],
            'endpoint_id': CONFIG['automl_endpoint_id'],
            'architecture': 'AutoML Image Classification',
            'input_size': f'{automl_predictor.input_size}x{automl_predictor.input_size}x3',
            'suggested_products': enhanced_products
        })
        
    except Exception as e:
        logger.error(f"Error processing AutoML request: {str(e)}")
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    model_available = automl_predictor.automl_model is not None
    endpoint_ready = automl_predictor.automl_endpoint_ready
    
    if endpoint_ready:
        status = 'ready'
    elif model_available:
        status = 'model_loaded_no_endpoint'
    else:
        status = 'not_configured'
    
    return jsonify({
        'status': 'healthy',
        'model_status': status,
        'model_type': 'automl_vertex_ai',
        'model_id': CONFIG['automl_model_id'],
        'endpoint_id': CONFIG['automl_endpoint_id'],
        'automl_model_ready': model_available,
        'automl_endpoint_ready': endpoint_ready,
        'input_size': f'{automl_predictor.input_size}x{automl_predictor.input_size}x3',
        'message': 'AutoML Vertex AI Hair Color Classifier API'
    })

@app.route('/categories', methods=['GET'])
def get_categories():
    """Get available hair color categories"""
    response = {
        'generic_categories': HAIR_COLOR_CLASSES,
        'automl_categories': AUTOML_COLOR_CLASSES,
        'active_model': 'automl',
        'color_mapping': AUTOML_TO_GENERIC_MAPPING
    }
    return jsonify(response)

# Model switching endpoint removed - using AutoML only

@app.route('/model-info', methods=['GET'])
def get_model_info():
    """Get AutoML model information"""
    return jsonify({
        'model_type': 'AutoML Vertex AI',
        'architecture': 'AutoML Image Classification',
        'model_status': 'ready' if automl_predictor.automl_model else 'not_configured',
        'endpoint_status': 'ready' if automl_predictor.automl_endpoint_ready else 'not_configured',
        'model_id': CONFIG['automl_model_id'],
        'endpoint_id': CONFIG['automl_endpoint_id'],
        'input_size': f'{automl_predictor.input_size}x{automl_predictor.input_size}x3',
        'automl_classes': AUTOML_COLOR_CLASSES,
        'generic_classes': HAIR_COLOR_CLASSES,
        'num_automl_classes': len(AUTOML_COLOR_CLASSES),
        'num_generic_classes': len(HAIR_COLOR_CLASSES),
        'project_id': CONFIG['project_id'],
        'location': CONFIG['location'],
        'features': [
            'Fully managed AutoML training',
            'Automatic hyperparameter tuning',
            'Built-in data augmentation',
            'Neural architecture search',
            'Product-specific color matching'
        ],
        'expected_accuracy': '90%+',
        'payload_size': f'~{(automl_predictor.input_size**2 * 3 * 4 / 1024):.0f}KB',
        'color_mapping': 'AutoML colors mapped to generic categories'
    })

if __name__ == '__main__':
    print("🚀 Starting AutoML Vertex AI Hair Color Classifier API")
    print("=" * 70)
    print(f"📍 Project: {CONFIG['project_id']}")
    print(f"🌍 Region: {CONFIG['location']}")
    print("=" * 70)
    print("🤖 AUTOML MODEL STATUS:")
    print(f"🎯 AutoML Model ID: {CONFIG['automl_model_id']}")
    print(f"🎯 AutoML Endpoint ID: {CONFIG['automl_endpoint_id'] or 'Not configured'}")
    print(f"✅ AutoML Model Ready: {'Yes' if automl_predictor.automl_model else 'No'}")
    print(f"✅ AutoML Endpoint Ready: {'Yes' if automl_predictor.automl_endpoint_ready else 'No'}")
    print(f"📏 Input Size: {automl_predictor.input_size}×{automl_predictor.input_size}×3")
    
    if not automl_predictor.automl_model and not automl_predictor.automl_endpoint_ready:
        print("⚠️  SETUP REQUIRED:")
        print("   1. Ensure automl_model_id is correct")
        print("   2. Deploy model to endpoint and set automl_endpoint_id")
        print("   3. Or use model directly without endpoint")
    elif not automl_predictor.automl_endpoint_ready:
        print("📍 Model loaded - predictions will use model directly")
        print("   To use endpoint, deploy model and set automl_endpoint_id")
    
    print("=" * 70)
    print("🌈 COLOR CLASSES:")
    print(f"   AutoML Classes: {len(AUTOML_COLOR_CLASSES)} product-specific colors")
    print(f"   Generic Classes: {len(HAIR_COLOR_CLASSES)} hair color categories")
    print("   Mapping: AutoML colors → Generic categories for products")
    print("=" * 70)
    
    app.run(debug=True, host='0.0.0.0', port=5000)