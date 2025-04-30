import json
import os
import string
import re

def categorize_char(char):
    """Categorize a character into different linguistic/symbol groups"""
    code_point = ord(char)
    
    # Check for Japanese characters
    if any([
        0x3040 <= code_point <= 0x309F,  # Hiragana
        0x30A0 <= code_point <= 0x30FF,  # Katakana
        0x4E00 <= code_point <= 0x9FFF   # Kanji (Note: overlaps with Chinese)
    ]):
        return "Japanese"
    
    # Check for Chinese characters (that aren't shared with Japanese)
    # CJK Unified Ideographs Extension blocks
    if any([
        0x3400 <= code_point <= 0x4DBF,  # Extension A
        0x20000 <= code_point <= 0x2A6DF,  # Extension B
        0x2A700 <= code_point <= 0x2B73F,  # Extension C
        0x2B740 <= code_point <= 0x2B81F,  # Extension D
        0x2B820 <= code_point <= 0x2CEAF,  # Extension E
        0x2CEB0 <= code_point <= 0x2EBEF,  # Extension F
        # Add more ranges if needed
    ]):
        return "Chinese"
    
    # Check for English alphabet characters
    if char.isalpha() and char in string.ascii_letters:
        return "English"
    
    # Check for numbers
    if char.isdigit():
        return "Number"
    
    # Check for special characters/punctuation
    if char in string.punctuation or char.isspace():
        return "Special"
    
    # Everything else
    return "Other"

def analyze_text(text):
    """Analyze text and return character distribution by category"""
    total_chars = len(text)
    if total_chars == 0:
        return {"Japanese": 0, "Chinese": 0, "English": 0, "Number": 0, "Special": 0, "Other": 0}
    
    categories = {
        "Japanese": 0,
        "Chinese": 0,
        "English": 0,
        "Number": 0,
        "Special": 0,
        "Other": 0
    }
    
    for char in text:
        category = categorize_char(char)
        categories[category] += 1
    
    # Convert counts to percentages
    percentages = {category: (count / total_chars) * 100 for category, count in categories.items()}
    
    return percentages

def parse_file(filepath, model_id, models):
    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            try:
                data = json.loads(line)
                model_answer = data.get("ModelAnswer", "")
                
                if model_answer:
                    char_count = len(model_answer)
                    percentages = analyze_text(model_answer)
                    item = os.path.basename(filepath)

                    if model_id not in models:
                        models[model_id] = {'total': {'chars': 0}}
                        for category in percentages.keys():
                            models[model_id]['total'][category] = 0

                    if item not in models[model_id]:
                        models[model_id][item] = {'chars': 0}
                        for category in percentages.keys():
                            models[model_id][item][category] = 0

                    models[model_id][item]['chars'] += char_count
                    for category, percentage in percentages.items():
                        models[model_id][item][category] = percentage
                    models[model_id]['total']['chars'] += char_count
            except json.JSONDecodeError:
                print(f"Error parsing JSON in file: {filepath}")
                continue

def update_total_percentages(models):
    """Calculate weighted averages for total percentages across files"""
    for model_id in models:
        total_chars = models[model_id]['total']['chars']
        if total_chars > 0:
            for category in ["Japanese", "Chinese", "English", "Number", "Special", "Other"]:
                # Calculate weighted average based on character count in each file
                weighted_sum = sum(
                    models[model_id][item]['chars'] * models[model_id][item].get(category, 0) / 100
                    for item in models[model_id] if item != 'total'
                )
                models[model_id]['total'][category] = (weighted_sum / total_chars) * 100

def analyze_jsonl_files(base_directory):
    models = {}
    folders = [
        "elyza__ELYZA-tasks-100",
        "lightblue__tengu_bench",
        "shisa-ai__ja-mt-bench-1shot",
        "yuzuai__rakuda-questions"
    ]
    
    for folder in folders:
        directory = os.path.join(base_directory, folder)
        if not os.path.exists(directory):
            print(f"Directory not found: {directory}")
            continue
            
        for filename in os.listdir(directory):
            if filename.endswith(".json") or filename.endswith(".jsonl"):
                file_path = os.path.join(directory, filename)
                model_id = filename.rsplit('.', 1)[0]
                parse_file(file_path, model_id, models)
                
    update_total_percentages(models)

    # Print results
    for model_id, data in sorted(models.items()):
        print(f"\nModel: {model_id}, Total Chars: {data['total']['chars']}")
        print(f"Character Distribution:")
        for category in ["Japanese", "Chinese", "English", "Number", "Special", "Other"]:
            print(f"  - {category}: {data['total'].get(category, 0):.2f}%")

# Usage
base_directory = "./data/model_answers"
analyze_jsonl_files(base_directory)
