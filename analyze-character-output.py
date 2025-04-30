import json
import os
import string
import re
from collections import Counter

# --- Constants ---
CACHE_FILE = "analysis_cache.json"

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

def parse_file(file_path: str) -> dict | None:
    """Parses a single JSONL file and returns character analysis."""
    total_chars = 0
    japanese_chars = 0
    chinese_chars = 0
    english_chars = 0
    number_chars = 0
    special_chars = 0
    other_chars = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    model_answer = data.get("ModelAnswer", "")
                    
                    if model_answer:
                        for char in model_answer:
                            category = categorize_char(char)
                            if category == "Japanese":
                                japanese_chars += 1
                            elif category == "Chinese":
                                chinese_chars += 1
                            elif category == "English":
                                english_chars += 1
                            elif category == "Number":
                                number_chars += 1
                            elif category == "Special":
                                special_chars += 1
                            else:
                                other_chars += 1
                        total_chars += len(model_answer)
                except json.JSONDecodeError:
                    print(f"Error parsing JSON in file: {file_path}")
                    continue
    except IOError as e:
        print(f"Error opening or reading file {file_path}: {e}")
        return None

    if total_chars == 0:
        print(f"Warning: No characters found or parsed in {os.path.basename(file_path)}")
        # Return a structure indicating zero counts if no characters
        return {
            'chars': 0,
            'Japanese': 0,
            'Chinese': 0,
            'English': 0,
            'Number': 0,
            'Special': 0,
            'Other': 0
        }

    # Calculate percentages
    japanese_percent = (japanese_chars / total_chars) * 100
    chinese_percent = (chinese_chars / total_chars) * 100
    english_percent = (english_chars / total_chars) * 100
    number_percent = (number_chars / total_chars) * 100
    special_percent = (special_chars / total_chars) * 100
    other_percent = (other_chars / total_chars) * 100

    # Instead of modifying models dict directly, return the analysis result
    analysis_result = {
        'chars': total_chars,
        'Japanese': japanese_percent,
        'Chinese': chinese_percent,
        'English': english_percent,
        'Number': number_percent,
        'Special': special_percent,
        'Other': other_percent
    }
    return analysis_result

def update_total_percentages(models):
    """Calculates the weighted average percentage for each category across all files for each model."""
    for model_id in models:
        total_chars_model = 0
        # Sum total chars from all files for this model
        for filename, file_data in models[model_id].get('files', {}).items():
            total_chars_model += file_data.get('chars', 0)

        if total_chars_model == 0:
             models[model_id]['total'] = {'chars': 0}
             for category in ["Japanese", "Chinese", "English", "Number", "Special", "Other"]:
                 models[model_id]['total'][category] = 0
             continue

        models[model_id]['total']['chars'] = total_chars_model

        for category in ["Japanese", "Chinese", "English", "Number", "Special", "Other"]:
            weighted_sum = 0.0
            # Calculate weighted sum from files
            for filename, file_data in models[model_id].get('files', {}).items():
                file_chars = file_data.get('chars', 0)
                category_percent = file_data.get(category, 0.0)
                if file_chars > 0:
                    weighted_sum += file_chars * (category_percent / 100.0)

            # Calculate final percentage for the category
            models[model_id]['total'][category] = (weighted_sum / total_chars_model) * 100.0 if total_chars_model > 0 else 0.0

def analyze_jsonl_files(base_directory):
    models = {}
    cache_data = {}

    # Load cache if exists
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                cache_data = json.load(f)
            print(f"Loaded cache from {CACHE_FILE}")
        except json.JSONDecodeError:
            print(f"Warning: Cache file {CACHE_FILE} is corrupted. Starting with an empty cache.")
            cache_data = {}
        except IOError as e:
            print(f"Warning: Could not read cache file {CACHE_FILE}: {e}. Starting with an empty cache.")
            cache_data = {}

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

                try:
                    current_mtime = os.path.getmtime(file_path)
                except OSError as e:
                    print(f"Error getting mtime for {file_path}: {e}. Skipping file.")
                    continue

                cache_key = file_path
                cached_entry = cache_data.get(cache_key)
                analysis_result = None

                # Check cache
                if cached_entry and cached_entry.get('mtime') == current_mtime:
                    print(f"Using cache for {filename}")
                    analysis_result = cached_entry.get('analysis')
                    if analysis_result is None:
                         print(f"Warning: Cache entry for {filename} has no analysis data. Re-analyzing.")
                
                # Analyze if not cached, changed, or cache entry was invalid
                if analysis_result is None:
                    print(f"Analyzing {filename}")
                    analysis_result = parse_file(file_path)
                    if analysis_result:
                        # Update cache data immediately after successful analysis
                        cache_data[cache_key] = {
                            'mtime': current_mtime,
                            'analysis': analysis_result
                        }
                    else:
                        # Handle case where parse_file returns None (e.g., error)
                        print(f"Skipping {filename} due to parsing error or no characters.")
                        # Optionally remove bad cache entry if it exists
                        if cache_key in cache_data:
                             del cache_data[cache_key]
                        continue # Skip adding this file to models dictionary

                # Update models dictionary using analysis_result (either new or cached)
                if model_id not in models:
                    models[model_id] = {'files': {}, 'total': {'chars': 0}}
                
                models[model_id]['files'][filename] = analysis_result
                
    update_total_percentages(models)

    # Save updated cache
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache_data, f, indent=4)
        print(f"Saved updated cache to {CACHE_FILE}")
    except IOError as e:
        print(f"Error writing cache file {CACHE_FILE}: {e}")

    # Print results
    for model_id, data in sorted(models.items()):
        print(f"\nModel: {model_id}, Total Chars: {data['total']['chars']}")
        print(f"Character Distribution:")
        for category in ["Japanese", "Chinese", "English", "Number", "Special", "Other"]:
            print(f"  - {category}: {data['total'].get(category, 0):.2f}%")

# Usage
base_directory = "./data/model_answers"
analyze_jsonl_files(base_directory)
