import os
import logging
import base64
import json
import sqlite3
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_file
from groq import Groq
import csv
import mimetypes
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Initialize Groq client
try:
    client = Groq(api_key="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")
    logging.info("Groq client initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize Groq client: {str(e)}")
    raise

# Global variables
image_path = None
csv_path = 'data.csv'
log_csv_path = 'refined_text_log.csv'
intake_log_path = 'intake_log.json'

# Database setup for ingredients
def init_db():
    conn = sqlite3.connect("ingredients.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ingredients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE
        )
    """)
    conn.commit()
    conn.close()

# Add ingredient to database
def add_ingredient(ingredient):
    try:
        conn = sqlite3.connect("ingredients.db")
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO ingredients (name) VALUES (?)", (ingredient.lower(),))
        conn.commit()
        conn.close()
        return True
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
        return False

# Fetch all ingredients from database
def get_ingredients():
    try:
        conn = sqlite3.connect("ingredients.db")
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM ingredients")
        ingredients = [row[0] for row in cursor.fetchall()]
        conn.close()
        return ingredients
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
        return []

# Original logic functions
def log_refined_text(refined_text):
    try:
        with open(log_csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([refined_text])
        logging.debug(f"Logged refined text to {log_csv_path}")
    except Exception as e:
        logging.error(f"Error logging refined text: {str(e)}")
        raise

def process_image_and_csv(image_path, csv_path):
    if not image_path:
        logging.error("No image path provided")
        return "Error: No image path provided!", "", ""
    try:
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type or not mime_type.startswith('image/'):
            mime_type = 'image/jpeg'
        logging.debug(f"Detected MIME type: {mime_type}")
        try:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        except IOError as e:
            logging.error(f"Failed to read image file: {str(e)}")
            raise ValueError(f"Cannot read image file: {str(e)}")
        image_data_url = f"data:{mime_type};base64,{base64_image}"
        logging.debug("Image encoded as base64 for Groq")
        response = client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract the Nutritional information from this Food pack label. Return the nutritional facts from the table, ingredients, and food name \n\nNutritional Facts:\n[Fact 1]\n[Fact 2]\n...\ ...\nnFood Name: [Name]\nDo not include any explanations other than the Nutritional facts and Ingredients"
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_url}
                        }
                    ]
                }
            ],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=False,
            stop=None
        )
        logging.debug(f"Groq response: {response}")
        refined_text = response.choices[0].message.content
        log_refined_text(refined_text)
        return refined_text, "", ""
    except Exception as e:
        logging.error(f"Error in process_image_and_csv: {str(e)}")
        return f"Error occurred: {str(e)}", "", ""

# Flask setup
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@app.route('/upload_nutritional', methods=['POST'])
def upload_nutritional():
    global image_path
    if 'file' not in request.files:
        logging.error("No file part in request")
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        logging.error("No file selected")
        return jsonify({"error": "No file selected"}), 400
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        logging.debug(f"Saving file to: {filepath}")
        file.save(filepath)
        if not os.path.exists(filepath):
            logging.error("File save failed")
            return jsonify({"error": "Failed to save the uploaded file"}), 500
        image_path = f"/{filepath}"
        refined_text, _, _ = process_image_and_csv(filepath, csv_path)
        if refined_text.startswith("Error occurred:"):
            logging.error(f"Processing failed: {refined_text}")
            return jsonify({"error": refined_text}), 500
        logging.info("Nutritional image processed successfully")
        return jsonify({"refined_text": refined_text, "image_url": image_path})
    except Exception as e:
        logging.error(f"Upload nutritional error: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/upload_medical', methods=['POST'])
def upload_medical():
    if 'file' not in request.files:
        logging.error("No file part in request")
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        logging.error("No file selected")
        return jsonify({"error": "No file selected"}), 400
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        logging.debug(f"Saving file to: {filepath}")
        file.save(filepath)
        if not os.path.exists(filepath):
            logging.error("File save failed")
            return jsonify({"error": "Failed to save the uploaded file"}), 500
        mime_type, _ = mimetypes.guess_type(filepath)
        if not mime_type or not mime_type.startswith('image/'):
            mime_type = 'image/jpeg'
        logging.debug(f"Detected MIME type: {mime_type}")
        try:
            with open(filepath, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        except IOError as e:
            logging.error(f"Failed to read image file: {str(e)}")
            raise ValueError(f"Cannot read image file: {str(e)}")
        image_data_url = f"data:{mime_type};base64,{base64_image}"
        logging.debug("Image encoded as base64 for Groq")
        response = client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract the important information from this medical report. Return only the important medical details and diagnosis (if available) in the following format:\n\nMedical Details:\n[Detail 1]\n[Detail 2]\n...\n\nDiagnosis: [Diagnosis]\n\nIf no diagnosis is present, omit the Diagnosis section. Do not include any explanations, steps, or additional text beyond this format."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_url}
                        }
                    ]
                }
            ],
            temperature=0.7,
            max_tokens=300,
            top_p=1,
            stream=False,
            stop=None
        )
        logging.debug(f"Groq response: {response}")
        refined_text = response.choices[0].message.content
        logging.info("Medical report processed successfully")
        return jsonify({"refined_text": refined_text})
    except Exception as e:
        logging.error(f"Upload medical error: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/evaluate_combined', methods=['POST'])
def evaluate_combined():
    data = request.json
    nutritional_text = data.get('nutritional_text', '').strip()
    medical_text = data.get('medical_text', '').strip()
    selected_model = data.get('model', 'llama-3.3-70b-versatile')
    selected_language = data.get('language', 'English')
    if not nutritional_text:
        logging.error("No nutritional text provided")
        return jsonify({"error": "Please analyze nutritional data first"}), 400
    if not medical_text:
        logging.error("No medical text provided")
        return jsonify({"error": "Please process medical report first"}), 400
    try:
        next_prompt = f"""
        Dear User,

        Based on the extracted text from your food pack labels: {nutritional_text},
        and the details from your medical report: {medical_text},
        please evaluate the ingredients for safety.
        Provide a short recommendation on whether the food is safe to consume,
        including the safe quantity for intake if applicable.
        If the food is not recommended, briefly explain why it should be avoided.

        Please provide the response in the following format:

        1. First, a short and clear recommendation in **English**.
        2. After that, a short and clear recommendation in **{selected_language}** that corresponds to the English response.
        """
        final_response = client.chat.completions.create(
            model=selected_model,
            messages=[{"role": "system", "content": "You are a professional medical advisor."},
                      {"role": "user", "content": next_prompt}],
            temperature=0.7,
            max_tokens=400,
            top_p=1,
            stream=False
        )
        logging.info("Combined evaluation completed successfully")
        return jsonify({"result": final_response.choices[0].message.content})
    except Exception as e:
        logging.error(f"Evaluate combined error: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/export_pdf', methods=['POST'])
def export_pdf():
    data = request.json
    result = data.get('result', '').strip()
    if not result:
        logging.error("No result provided for PDF export")
        return jsonify({"error": "No evaluation result to export"}), 400
    try:
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        story.append(Paragraph("Health and Nutrition Report", styles['Title']))
        story.append(Spacer(1, 12))
        for line in result.split('\n'):
            story.append(Paragraph(line, styles['Normal']))
            story.append(Spacer(1, 6))
        doc.build(story)
        logging.info(f"PDF generated: {pdf_path}")
        return send_file(pdf_path, as_attachment=True)
    except Exception as e:
        logging.error(f"PDF export error: {str(e)}")
        return jsonify({"error": f"Failed to generate PDF: {str(e)}"}), 500

@app.route('/confirm_intake', methods=['POST'])
def confirm_intake():
    data = request.json
    nutritional_text = data.get('nutritional_text', '').strip()
    if not nutritional_text:
        logging.error("No nutritional text provided for intake confirmation")
        return jsonify({"error": "No nutritional data to confirm"}), 400
    try:
        nutrients = {}
        for line in nutritional_text.split('\n'):
            if 'Carbohydrates' in line:
                nutrients['carbs'] = float(line.split()[-2]) if line.split()[-2].replace('.', '').isdigit() else 0
            elif 'Sugars' in line:
                nutrients['sugars'] = float(line.split()[-2]) if line.split()[-2].replace('.', '').isdigit() else 0
            elif 'Sodium' in line:
                nutrients['sodium'] = float(line.split()[-2]) if line.split()[-2].replace('.', '').isdigit() else 0
        intake_entry = {
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'nutrients': nutrients
        }
        if os.path.exists(intake_log_path):
            with open(intake_log_path, 'r') as f:
                intake_log = json.load(f)
        else:
            intake_log = []
        intake_log.append(intake_entry)
        with open(intake_log_path, 'w') as f:
            json.dump(intake_log, f, indent=2)
        logging.info("Food intake confirmed and logged")
        return jsonify({"message": "Intake confirmed and logged"})
    except Exception as e:
        logging.error(f"Intake confirmation error: {str(e)}")
        return jsonify({"error": f"Failed to log intake: {str(e)}"}), 500

@app.route('/dashboard', methods=['GET'])
def dashboard():
    try:
        if not os.path.exists(intake_log_path):
            return jsonify({"daily": {}, "weekly": {}})
        with open(intake_log_path, 'r') as f:
            intake_log = json.load(f)
        
        today = datetime.now().strftime('%Y-%m-%d')
        week_start = (datetime.now() - timedelta(days=datetime.now().weekday())).strftime('%Y-%m-%d')
        daily_totals = {'carbs': 0, 'sugars': 0, 'sodium': 0}
        weekly_totals = {'carbs': 0, 'sugars': 0, 'sodium': 0}
        
        for entry in intake_log:
            entry_date = entry['date'].split()[0]
            nutrients = entry['nutrients']
            if entry_date == today:
                for nutrient in daily_totals:
                    daily_totals[nutrient] += nutrients.get(nutrient, 0)
            if entry_date >= week_start:
                for nutrient in weekly_totals:
                    weekly_totals[nutrient] += nutrients.get(nutrient, 0)
        
        logging.info("Dashboard data calculated")
        return jsonify({"daily": daily_totals, "weekly": weekly_totals})
    except Exception as e:
        logging.error(f"Dashboard error: {str(e)}")
        return jsonify({"error": f"Failed to load dashboard: {str(e)}"}), 500

@app.route('/add_ingredient', methods=['POST'])
def add_ingredient_endpoint():
    data = request.json
    ingredient = data.get('ingredient', '').strip()
    if not ingredient:
        logging.error("No ingredient provided")
        return jsonify({"error": "Please provide an ingredient"}), 400
    if add_ingredient(ingredient):
        logging.info(f"Ingredient '{ingredient}' added successfully")
        return jsonify({"message": f"Added '{ingredient}' to the database"})
    else:
        logging.error("Failed to add ingredient")
        return jsonify({"error": "Failed to add ingredient"}), 500

@app.route('/get_ingredients', methods=['GET'])
def get_ingredients_endpoint():
    ingredients = get_ingredients()
    return jsonify({"ingredients": ingredients})

@app.route('/meal_plan', methods=['POST'])
def meal_plan():
    data = request.json
    medical_text = data.get('medical_text', '').strip()
    ingredients = data.get('ingredients', [])
    meal_plan_type = data.get('meal_plan_type', 'recipe')  # 'recipe' or 'weekly'
    
    if not ingredients:
        logging.error("No ingredients provided for meal plan")
        return jsonify({"error": "Please add some ingredients first"}), 400
    
    try:
        if not medical_text:
            prompt = (
                f"I have the following ingredients: {', '.join(ingredients)}. "
                f"Suggest a {'weekly meal plan' if meal_plan_type == 'weekly' else 'recipe'} based on these ingredients."
            )
        else:
            prompt = (
                f"Based on the following health conditions from a medical report: {medical_text}, "
                f"and the ingredients I have: {', '.join(ingredients)}, "
                f"suggest a {'weekly meal plan' if meal_plan_type == 'weekly' else 'recipe'} "
                f"that is safe and suitable for my health condition."
            )
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=False,
            stop=None
        )
        suggestion = response.choices[0].message.content.strip()
        logging.info("Meal plan generated successfully")
        return jsonify({"suggestion": suggestion})
    except Exception as e:
        logging.error(f"Meal plan error: {str(e)}")
        return jsonify({"error": f"Failed to generate meal plan: {str(e)}"}), 500

# Initialize app on startup
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
init_db()  # Initialize ingredients database
logging.info(f"Upload folder created/verified: {UPLOAD_FOLDER}")