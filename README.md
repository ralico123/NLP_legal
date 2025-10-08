# Legal Document Summarizer

A web application that uses advanced NLP models to summarize complex legal documents into clear, concise summaries. Built with Flask and Hugging Face Transformers.

## Features

- **Multiple AI Models**: Choose from 4 different specialized models:
  - Legal LED large
  - Legal LED base
  - Long T5 base
  - BART base
  

- **Flexible Input Methods**:
  - Paste text directly
  - Upload PDF, DOC, DOCX, or TXT files
  - Drag and drop file upload

- **Smart Text Processing**:
  - Roman numeral conversion
  - Spelling correction
  - Legal acronym expansion
  - Text preprocessing for optimal results

- **Modern Web Interface**:
  - Responsive design
  - Real-time character counting
  - Progress indicators
  - Summary statistics

## Installation

1. **Clone or download this repository**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```

4. **Open your browser** and go to `http://localhost:8080`

## Usage

### Text Input
1. Select your preferred AI model
2. Switch to the "Paste Text" tab
3. Paste your legal document text (max 10,000 characters)
4. Click "Generate Summary"

### File Upload
1. Select your preferred AI model
2. Switch to the "Upload File" tab
3. Upload a PDF, DOC, DOCX, or TXT file (max 16MB)
4. Click "Generate Summary"

## Supported File Types

- **PDF**: Extracts text using PyPDF2
- **DOC/DOCX**: Extracts text using python-docx
- **TXT**: Plain text files

## Models

### Legal LED large (Recommended)
- **Model**: `0-hero/led-large-legal-summary`
- **Best for**: Complex legal documents with large context
- **Context Window**: 16,384 tokens

### Legal LED base
- **Model**: `nsi319/legal-led-base-16384`
- **Best for**: General legal document summarization
- **Context Window**: 16,384 tokens

### Long T5 base
- **Model**: `google/long-t5-tglobal-base`
- **Best for**: Abstractive summarization for tasks in NLP among others
- **Context Window**: 4096 tokens

### BART base
- **Model**: `facebook/bart-base`
- **Best for**: General text summarization
- **Context Window**: 1024

## API Endpoints

### GET `/api/models`
Returns available models and their configurations.

### POST `/api/summarize`
Summarize text input.
```json
{
  "text": "Your legal document text...",
  "model": "led_large"
}
```

### POST `/api/upload`
Upload and summarize a file.
- **Content-Type**: `multipart/form-data`
- **Parameters**: `file`, `model`

## Technical Details

- **Backend**: Flask with CORS support
- **ML Framework**: PyTorch + Hugging Face Transformers
- **Text Processing**: SymSpell, Roman numeral conversion, acronym expansion
- **File Processing**: PyPDF2, python-docx
- **Frontend**: Bootstrap 5, vanilla JavaScript

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended for faster inference)
- 8GB+ RAM (for model loading)

## Performance Notes

- First model load may take 1-2 minutes
- Subsequent requests are faster due to model caching
- GPU acceleration significantly improves speed
- Text is limited to 10,000 characters for memory efficiency

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce text length or use a smaller model
2. **Model Loading Error**: Ensure you have sufficient RAM and disk space
3. **File Upload Error**: Check file size and format
4. **CUDA Error**: Install appropriate PyTorch version for your system

### Getting Help

If you encounter issues:
1. Check the console output for error messages
2. Ensure all dependencies are installed correctly
3. Verify your Python version (3.8+)
4. Check available system memory

## License

This project is for educational and research purposes. Please respect the terms of use for the underlying models and datasets.

## Acknowledgments

- Hugging Face for the Transformers library
- The creators of the legal summarization models
- The open-source community for various dependencies
