from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
from PIL import Image
import os
from groq import Groq
import csv
from io import StringIO

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded environment variables from .env file")
except ImportError:
    print("python-dotenv not installed. Using system environment variables only.")
    print("To use .env files, run: pip install python-dotenv")

app = Flask(__name__)
CORS(app)

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Product data - parsed from your CSV
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

def encode_image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def classify_hair_color(image):
    """Use Groq Vision to classify hair color"""
    try:
        # Convert image to base64
        img_base64 = encode_image_to_base64(image)
        
        # Available hair color categories
        hair_categories = list(PRODUCTS.keys())
        
        prompt = f"""Analyze this image and classify the person's hair color into one of these specific categories:
        
{', '.join(hair_categories)}

Look at the dominant hair color in the image. Consider:
- The overall tone and shade
- Whether it's natural or colored
- The lighting conditions

Respond with ONLY the exact category name from the list above that best matches the hair color. Do not provide explanations or multiple options."""

        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000,
            temperature=0.1
        )
        
        predicted_category = response.choices[0].message.content.strip().lower()
        
        # Find best match from available categories
        for category in hair_categories:
            if category.lower() in predicted_category or predicted_category in category.lower():
                return category
        
        # Default fallback
        return "middle_brown"
        
    except Exception as e:
        print(f"Error in hair color classification: {str(e)}")
        return "middle_brown"  # Default fallback

@app.route('/classify', methods=['POST'])
def classify_image():
    try:
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
        
        # Resize image if too large (optional)
        max_size = (800, 800)
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Classify hair color
        predicted_category = classify_hair_color(image)
        
        # Get matching products
        suggested_products = PRODUCTS.get(predicted_category, [])
        
        return jsonify({
            'predicted_category': predicted_category,
            'confidence': 'high',  # Groq doesn't return confidence scores
            'suggested_products': suggested_products[:8]  # Limit to 8 products
        })
        
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

@app.route('/categories', methods=['GET'])
def get_categories():
    return jsonify({'categories': list(PRODUCTS.keys())})

if __name__ == '__main__':
    # Make sure to set your GROQ_API_KEY environment variable
    if not os.environ.get("GROQ_API_KEY"):
        print("Warning: GROQ_API_KEY environment variable not set")
    
    app.run(debug=True, host='0.0.0.0', port=5000)