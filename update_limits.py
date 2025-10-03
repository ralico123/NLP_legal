#!/usr/bin/env python3

# Update the MODEL_CONFIGS with proper character limits
model_configs = {
    'bart': {
        'name': 'facebook/bart-base',
        'max_input_length': 1024,
        'max_target_length': 128,
        'max_tokens': 1024,
        'max_chars': 768,  # 1024 * 0.75
        'description': 'BART Base (1k tokens)'
    },
    'led_base': {
        'name': 'nsi319/legal-led-base-16384',
        'max_input_length': 16384,
        'max_target_length': 128,
        'max_tokens': 16384,
        'max_chars': 12288,  # 16384 * 0.75
        'description': 'LED Base Legal (16k tokens)'
    },
    'led_large': {
        'name': '0-hero/led-large-legal-summary',
        'max_input_length': 16384,
        'max_target_length': 128,
        'max_tokens': 16384,
        'max_chars': 12288,  # 16384 * 0.75
        'description': 'LED Large Legal Summary (16k tokens)'
    },
    'pegasus': {
        'name': 'nsi319/legal-pegasus',
        'max_input_length': 4096,
        'max_target_length': 128,
        'max_tokens': 4096,
        'max_chars': 3072,  # 4096 * 0.75
        'description': 'Legal Pegasus (4k tokens)'
    },
    'pegasus_xsum': {
        'name': 'google/pegasus-xsum',
        'max_input_length': 4096,
        'max_target_length': 128,
        'max_tokens': 4096,
        'max_chars': 3072,  # 4096 * 0.75
        'description': 'Pegasus XSum (4k tokens)'
    }
}

print("Updated character limits:")
for model, config in model_configs.items():
    print(f"{model}: {config['max_tokens']} tokens = {config['max_chars']} characters")
