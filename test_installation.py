#!/usr/bin/env python3
"""
Test script to verify all dependencies are installed correctly
"""

import sys
import importlib

def test_import(module_name, package_name=None):
    """Test if a module can be imported"""
    try:
        if package_name:
            importlib.import_module(module_name, package_name)
        else:
            importlib.import_module(module_name)
        print(f"✅ {module_name}")
        return True
    except ImportError as e:
        print(f"❌ {module_name}: {e}")
        return False

def main():
    print("🧪 Testing Legal Document Summarizer Dependencies")
    print("=" * 50)
    
    # Core dependencies
    dependencies = [
        'flask',
        'flask_cors',
        'torch',
        'transformers',
        'PyPDF2',
        'docx',
        'werkzeug',
        'roman',
        'symspellpy',
        'textstat',
        'rouge_score',
        'bert_score',
        'datasets',
        'evaluate',
        'nltk',
        'wordfreq',
        'numpy',
        'pandas',
        'tqdm'
    ]
    
    failed_imports = []
    
    for dep in dependencies:
        if not test_import(dep):
            failed_imports.append(dep)
    
    print("\n" + "=" * 50)
    
    if failed_imports:
        print(f"❌ {len(failed_imports)} dependencies failed to import:")
        for dep in failed_imports:
            print(f"   - {dep}")
        print("\nPlease install missing dependencies with:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    else:
        print("✅ All dependencies are installed correctly!")
        print("\nYou can now run the application with:")
        print("python app.py")
        print("or")
        print("./start.sh")

if __name__ == "__main__":
    main()
