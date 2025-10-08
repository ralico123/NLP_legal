from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import PyPDF2
import docx
import io
import os
import tempfile
from werkzeug.utils import secure_filename
import re
import string
import roman
from symspellpy.symspellpy import SymSpell, Verbosity
import pkg_resources

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize models dictionary
models = {}
tokenizers = {}

# Model configurations with token limits
MODEL_CONFIGS = {
    'bart': {
        'name': 'facebook/bart-base',
        'max_input_length': 1024,
        'max_target_length': 256,
        'max_tokens': 1024,
        'description': 'BART Base (1024 tokens)'
    },
    'led_base': {
        'name': 'nsi319/legal-led-base-16384',
        'max_input_length': 8192,
        'max_target_length': 256,
        'max_tokens': 8192,
        'description': 'LED Base Legal (8192 tokens)'
    },
    'led_large': {
        'name': '0-hero/led-large-legal-summary',
        'max_input_length': 16384,
        'max_target_length': 256,
        'max_tokens': 16384,
        'description': 'LED Large Legal Summary (16384 tokens)'
    },
    'longt5_base': {
        'name': 'google/long-t5-tglobal-base',
        'max_input_length': 4096,
        'max_target_length': 256,
        'max_tokens': 4096,
        'description': 'Pegasus XSum (4096 tokens)'
    }
}

# Initialize SymSpell for text correction
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# DEMO MODE - Skip model loading for now
DEMO_MODE = True

def load_model(model_key):
    """Load a specific model and tokenizer - DISABLED IN DEMO MODE"""
    if DEMO_MODE:
        print(f"DEMO MODE: Skipping model loading for {model_key}")
        return None, None
    
    if model_key not in models:
        config = MODEL_CONFIGS[model_key]
        print(f"Loading {config['name']}...")
        tokenizer = AutoTokenizer.from_pretrained(config['name'])
        model = AutoModelForSeq2SeqLM.from_pretrained(config['name']).to(device)
        model.eval()
        
        models[model_key] = model
        tokenizers[model_key] = tokenizer
        print(f"✅ {config['name']} loaded successfully")
    
    return models[model_key], tokenizers[model_key]

def count_tokens(text, model_key):
    """Count tokens for a given text using the model's tokenizer - DEMO MODE"""
    if DEMO_MODE:
        # Rough estimation: 1 token ≈ 0.75 characters
        return int(len(text) * 0.75)
    
    try:
        _, tokenizer = load_model(model_key)
        tokens = tokenizer.encode(text, add_special_tokens=True)
        return len(tokens)
    except Exception as e:
        print(f"Error counting tokens: {e}")
        # Fallback: rough estimation (1 token ≈ 0.75 characters)
        return int(len(text) * 0.75)

def get_available_models(text_length):
    """Get models that can handle the given text length"""
    available = []
    for model_key, config in MODEL_CONFIGS.items():
        if text_length <= config['max_tokens']:
            available.append({
                'key': model_key,
                'name': config['description'],
                'max_tokens': config['max_tokens']
            })
    return available

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    """Extract text from PDF file"""
    text = ""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error extracting PDF: {e}")
    return text

def extract_text_from_docx(file_path):
    """Extract text from DOCX file"""
    text = ""
    try:
        doc = docx.Document(file_path)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        print(f"Error extracting DOCX: {e}")
    return text

def extract_text_from_txt(file_path):
    """Extract text from TXT file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error extracting TXT: {e}")
        return ""

def clean_text_for_readability(text):
    """Clean text for readability analysis"""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove digits for readability check
    allowed_chars = string.ascii_lowercase + " "
    cleaned = "".join(ch if ch in allowed_chars else " " for ch in text)
    return " ".join(cleaned.split())

def convert_roman_numerals(text):
    """Convert Roman numerals to numbers"""
    words = text.split()
    converted = []
    for word in words:
        clean_word = word.strip(string.punctuation)
        try:
            num = roman.fromRoman(clean_word.upper())
            converted_word = word.replace(clean_word, str(num))
            converted.append(converted_word)
        except roman.InvalidRomanNumeralError:
            converted.append(word)
    return " ".join(converted)

def correct_with_symspell(text):
    """Correct spelling using SymSpell"""
    corrected_words = []
    for word in text.split():
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        if suggestions:
            corrected_words.append(suggestions[0].term)
        else:
            corrected_words.append(word)
    return " ".join(corrected_words)

def replace_acronyms(text):
    """Replace common legal acronyms with full forms"""
    known_acronyms = {
        "fda": "food and drug administration",
        "epa": "environmental protection agency",
        "pdmp": "prescription drug monitoring program",
        "medpac": "medicare payment advisory commission",
        "sec": "securities and exchange commission",
        "cfr": "code of federal regulations",
        "usc": "united states code"
    }
    for acronym, full in known_acronyms.items():
        text = re.sub(rf"\b{acronym}\b", full, text, flags=re.IGNORECASE)
    return text

def preprocess_text(text):
    """Apply all text preprocessing steps"""
    text = convert_roman_numerals(text)
    text = correct_with_symspell(text)
    text = replace_acronyms(text)
    return text

def generate_summary(text, model_key):
    """Generate summary using specified model - DEMO MODE"""
    if DEMO_MODE:
        # Return a mock summary for demo purposes
        words = text.split()
        summary_length = min(50, len(words) // 10)  # Roughly 10% of original length
        summary_words = words[:summary_length]
        
        mock_summary = " ".join(summary_words) + "... [DEMO SUMMARY - Model not loaded]"
        return mock_summary
    
    model, tokenizer = load_model(model_key)
    config = MODEL_CONFIGS[model_key]
    
    # Preprocess text
    processed_text = preprocess_text(text)
    
    # Tokenize input
    inputs = tokenizer(
        processed_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config['max_input_length']
    ).to(device)
    
    # Generate summary
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=config['max_target_length'],
            num_beams=4,
            early_stopping=True,
            do_sample=False
        )
    
    # Decode output
    summary = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return summary

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(STATIC_FOLDER, filename)

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available models"""
    return jsonify({
        'models': list(MODEL_CONFIGS.keys()),
        'model_info': MODEL_CONFIGS
    })

@app.route('/api/count-tokens', methods=['POST'])
def count_tokens_endpoint():
    """Count tokens for given text"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text.strip():
            return jsonify({'error': 'No text provided'}), 400
        
        # Count tokens using BART tokenizer as reference
        token_count = count_tokens(text, 'bart')
        
        # Get available models for this text length
        available_models = get_available_models(token_count)
        
        return jsonify({
            'token_count': token_count,
            'available_models': available_models
        })
        
    except Exception as e:
        print(f"Error counting tokens: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/summarize', methods=['POST'])
def summarize():
    """Summarize text or uploaded file"""
    try:
        data = request.get_json()
        model_key = data.get('model', 'led_large')  # Default to LED Large
        
        if model_key not in MODEL_CONFIGS:
            return jsonify({'error': 'Invalid model selected'}), 400
        
        # Check if text is provided directly
        if 'text' in data and data['text'].strip():
            text = data['text'].strip()
        else:
            return jsonify({'error': 'No text provided'}), 400
        
        # Count tokens
        token_count = count_tokens(text, model_key)
        
        # Check if text is too long for the selected model
        if token_count > MODEL_CONFIGS[model_key]['max_tokens']:
            return jsonify({
                'error': f'Text too long for selected model. Document has {token_count} tokens, but {model_key} can only handle {MODEL_CONFIGS[model_key]["max_tokens"]} tokens.'
            }), 400
        
        # Limit text length to prevent memory issues (use model's max length)
        max_chars = int(MODEL_CONFIGS[model_key]['max_tokens'] * 0.75)  # 1 token ≈ 0.75 characters
        if len(text) > max_chars:
            text = text[:max_chars]
            text += "... [truncated]"
        
        # Generate summary
        summary = generate_summary(text, model_key)
        
        return jsonify({
            'summary': summary,
            'model_used': model_key,
            'original_length': len(text),
            'summary_length': len(summary),
            'token_count': token_count
        })
        
    except Exception as e:
        print(f"Error in summarize: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload and extraction"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        model_key = request.form.get('model', 'led_large')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed. Please upload PDF, DOC, DOCX, or TXT files.'}), 400
        
        if model_key not in MODEL_CONFIGS:
            return jsonify({'error': 'Invalid model selected'}), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Extract text based on file type
        file_ext = filename.rsplit('.', 1)[1].lower()
        text = ""
        
        if file_ext == 'pdf':
            text = extract_text_from_pdf(file_path)
        elif file_ext in ['doc', 'docx']:
            text = extract_text_from_docx(file_path)
        elif file_ext == 'txt':
            text = extract_text_from_txt(file_path)
        
        # Clean up uploaded file
        os.remove(file_path)
        
        if not text.strip():
            return jsonify({'error': 'Could not extract text from file'}), 400
        
        # Count tokens
        token_count = count_tokens(text, model_key)
        
        # Check if text is too long for the selected model
        if token_count > MODEL_CONFIGS[model_key]['max_tokens']:
            return jsonify({
                'error': f'Text too long for selected model. Document has {token_count} tokens, but {model_key} can only handle {MODEL_CONFIGS[model_key]["max_tokens"]} tokens.'
            }), 400
        
        # Limit text length
        max_chars = int(MODEL_CONFIGS[model_key]['max_tokens'] * 0.75)  # Rough estimation
        if len(text) > max_chars:
            text = text[:max_chars]
            text += "... [truncated]"
        
        # Generate summary
        summary = generate_summary(text, model_key)
        
        return jsonify({
            'summary': summary,
            'model_used': model_key,
            'original_length': len(text),
            'summary_length': len(summary),
            'filename': filename,
            'token_count': token_count
        })
        
    except Exception as e:
        print(f"Error in upload_file: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Legal Document Summarization API...")
    print(f"Available models: {list(MODEL_CONFIGS.keys())}")
    if DEMO_MODE:
        print("⚠️  DEMO MODE: Models will not be loaded - mock summaries will be generated")
        print("   To enable real models, set DEMO_MODE = False in app.py")
    print("📁 Static files will be served from: /static/")
    print("📸 Place your JPG images in: static/images/")
    app.run(host='0.0.0.0', port=5000)
