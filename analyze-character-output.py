import json
import os
import re
import string
from tabulate import tabulate
from collections import Counter

# --- Constants ---
CACHE_FILE = "analysis_cache.json"

def categorize_char(char):
    """Categorize a character into different linguistic/symbol groups with enhanced detection"""
    code_point = ord(char)
    
    # Japanese-specific characters
    if any([
        0x3040 <= code_point <= 0x309F,  # Hiragana
        0x30A0 <= code_point <= 0x30FF,  # Katakana
    ]):
        return "Japanese"
    
    # CJK Unified Ideographs (Kanji/Hanzi) - Shared between Japanese and Chinese
    # We need to be careful here, as many of these characters are used in both languages
    # Let's consider basic CJK block (0x4E00-0x9FFF) as potentially shared
    if 0x4E00 <= code_point <= 0x9FFF:
        # To improve detection, we could implement frequency analysis or context-based 
        # classification here, but for now we'll categorize as CJK
        return "Japanese"  # Default to Japanese since analyzing Japanese text
        
    # Chinese-specific characters (extensions not commonly used in Japanese)
    if any([
        0x3400 <= code_point <= 0x4DBF,  # Extension A - some overlap with Japanese but less common
        0x20000 <= code_point <= 0x2A6DF,  # Extension B - rarely used in Japanese
        0x2A700 <= code_point <= 0x2B73F,  # Extension C - not commonly used in Japanese
        0x2B740 <= code_point <= 0x2B81F,  # Extension D - not commonly used in Japanese
        0x2B820 <= code_point <= 0x2CEAF,  # Extension E - not commonly used in Japanese
        0x2CEB0 <= code_point <= 0x2EBEF,  # Extension F - not commonly used in Japanese
        0xF900 <= code_point <= 0xFAFF,    # CJK Compatibility Ideographs
        0x2F800 <= code_point <= 0x2FA1F,  # CJK Compatibility Ideographs Supplement
    ]):
        return "Chinese"
        
    # Chinese-specific punctuation and symbols
    if any([
        0x3000 <= code_point <= 0x303F and code_point not in [0x3001, 0x3002, 0x300C, 0x300D, 0x3005],  # CJK punctuation excluding ones common in Japanese
        0xFE30 <= code_point <= 0xFE4F,    # CJK Compatibility Forms
    ]):
        return "Chinese"
        
    # Japanese-specific punctuation and symbols
    if any([
        code_point in [0x3001, 0x3002, 0x300C, 0x300D, 0x3005],  # Common Japanese punctuation
        code_point in [0x30FB, 0x30FC],    # Middle dot and prolonged sound mark
        0x3099 <= code_point <= 0x309C,    # Japanese combining marks
    ]):
        return "Japanese_Punct"
        
    # English alphabet characters
    if char.isalpha() and char in string.ascii_letters:
        return "English"
    
    # Numbers (both half-width and full-width)
    if char.isdigit() or (0xFF10 <= code_point <= 0xFF19):  # Include full-width digits
        return "Number"
    
    # Whitespace (separate from punctuation)
    if char.isspace():
        return "Whitespace"
    
    # Latin/English punctuation
    if char in string.punctuation:
        return "Latin_Punct"
        
    # Other Unicode punctuation and symbols
    if any([
        0x2000 <= code_point <= 0x206F,  # General Punctuation
        0x2E00 <= code_point <= 0x2E7F,  # Supplemental Punctuation
    ]):
        return "Other_Punct"
    
    # Everything else
    return "Other"

def analyze_text(text):
    """Analyze text and return character distribution by category"""
    total_chars = len(text)
    if total_chars == 0:
        return {"Japanese": 0, "Chinese": 0, "English": 0, "Number": 0, "Whitespace": 0, 
                "Japanese_Punct": 0, "Latin_Punct": 0, "Other_Punct": 0, "Other": 0}
    
    categories = {
        "Japanese": 0,
        "Chinese": 0,
        "English": 0,
        "Number": 0,
        "Whitespace": 0,
        "Japanese_Punct": 0,
        "Latin_Punct": 0,
        "Other_Punct": 0,
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
    whitespace_chars = 0
    japanese_punct_chars = 0
    latin_punct_chars = 0
    other_punct_chars = 0
    other_chars = 0

    # Regex to remove anything in various reasoning tags (including the tags)
    # Handle <think>, <thinking>, and <reason> tags
    reasoning_pattern = re.compile(r'<(think|thinking|reason)>.*?</(think|thinking|reason)>', re.DOTALL | re.IGNORECASE)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    model_answer = data.get("ModelAnswer", "")
                    
                    if model_answer:
                        # Strip out anything in reasoning tags
                        cleaned_answer = reasoning_pattern.sub('', model_answer)
                        
                        for char in cleaned_answer:
                            category = categorize_char(char)
                            if category == "Japanese":
                                japanese_chars += 1
                            elif category == "Chinese":
                                chinese_chars += 1
                            elif category == "English":
                                english_chars += 1
                            elif category == "Number":
                                number_chars += 1
                            elif category == "Whitespace":
                                whitespace_chars += 1
                            elif category == "Japanese_Punct":
                                japanese_punct_chars += 1
                            elif category == "Latin_Punct":
                                latin_punct_chars += 1
                            elif category == "Other_Punct":
                                other_punct_chars += 1
                            else:
                                other_chars += 1
                        total_chars += len(cleaned_answer)
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
            'Whitespace': 0,
            'Japanese_Punct': 0,
            'Latin_Punct': 0,
            'Other_Punct': 0,
            'Other': 0,
            'Special': 0
        }

    # Calculate percentages
    percentages = {
        "Japanese": (japanese_chars / total_chars) * 100 if total_chars else 0,
        "Chinese": (chinese_chars / total_chars) * 100 if total_chars else 0,
        "English": (english_chars / total_chars) * 100 if total_chars else 0,
        "Number": (number_chars / total_chars) * 100 if total_chars else 0,
        "Whitespace": (whitespace_chars / total_chars) * 100 if total_chars else 0,
        "Japanese_Punct": (japanese_punct_chars / total_chars) * 100 if total_chars else 0,
        "Latin_Punct": (latin_punct_chars / total_chars) * 100 if total_chars else 0,
        "Other_Punct": (other_punct_chars / total_chars) * 100 if total_chars else 0,
        "Other": (other_chars / total_chars) * 100 if total_chars else 0,
        "chars": total_chars
    }
    
    # Calculate a combined punctuation percentage for backward compatibility
    total_punct = japanese_punct_chars + latin_punct_chars + other_punct_chars
    percentages["Special"] = ((whitespace_chars + total_punct) / total_chars) * 100 if total_chars else 0
    return percentages

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

    # Print results using tabulate - main summary table
    headers = ["Model", "# Chars", "JA", "ZH", "EN", "#", "Special", "Other"]
    table_data = []
    
    '''
    for model_id, data in sorted(models.items()):
        row = [
            model_id,
            data['total']['chars'],
            f"{data['total'].get('Japanese', 0):.2f}%",
            f"{data['total'].get('Chinese', 0):.2f}%",
            f"{data['total'].get('English', 0):.2f}%",
            f"{data['total'].get('Number', 0):.2f}%",
            f"{data['total'].get('Special', 0):.2f}%",
            f"{data['total'].get('Other', 0):.2f}%"
        ]
        table_data.append(row)
    '''

    # Also create a detailed table with more granular breakdowns
    detail_headers = ["Model", "# Chars", "JA", "ZH", "EN", "#", "WS", "JP_P", "LAT_P", "OTH_P", "Other"]
    detail_table = []
    
    for model_id, data in sorted(models.items()):
        row = [
            model_id,
            data['total']['chars'],
            f"{data['total'].get('Japanese', 0):.2f}%",
            f"{data['total'].get('Chinese', 0):.2f}%",
            f"{data['total'].get('English', 0):.2f}%",
            f"{data['total'].get('Number', 0):.2f}%",
            f"{data['total'].get('Whitespace', 0):.2f}%",
            f"{data['total'].get('Japanese_Punct', 0):.2f}%",
            f"{data['total'].get('Latin_Punct', 0):.2f}%",
            f"{data['total'].get('Other_Punct', 0):.2f}%",
            f"{data['total'].get('Other', 0):.2f}%"
        ]
        detail_table.append(row)
    
    # print("\nCharacter Distribution Analysis:")
    # print(tabulate(table_data, headers=headers, tablefmt="pipe", stralign="right"))
    
    print("\nDetailed Character Distribution (WS=Whitespace, JP_P=Japanese Punctuation, LAT_P=Latin Punctuation, OTH_P=Other Punctuation):")
    print(tabulate(detail_table, headers=detail_headers, tablefmt="pipe", stralign="right"))

# Usage
base_directory = "./data/model_answers"
analyze_jsonl_files(base_directory)
