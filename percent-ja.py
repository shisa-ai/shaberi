import json
import os
import string

def is_japanese_char(char):
    # Check if the character falls within the Japanese Unicode codepoint ranges
    return any([
        0x3040 <= ord(char) <= 0x309F,  # Hiragana
        0x30A0 <= ord(char) <= 0x30FF,  # Katakana
        0x4E00 <= ord(char) <= 0x9FFF   # Kanji
    ])

def is_acceptable_char(char):
    # Check if the character is a common punctuation or numeral
    return char in string.punctuation or char.isdigit()

def calculate_non_japanese_percentage(text):
    total_chars = len(text)
    japanese_chars = sum(is_japanese_char(char) or is_acceptable_char(char) for char in text)
    non_japanese_chars = total_chars - japanese_chars
    percentage = (non_japanese_chars / total_chars) * 100 if total_chars > 0 else 0
    return percentage

def parse_file(filepath, model_id, models):
    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            model_answer = data.get("ModelAnswer", "")
            if model_answer:
                char_count = len(model_answer)
                percent_ja = 100 - calculate_non_japanese_percentage(model_answer)
                item = os.path.basename(filepath)

                if model_id not in models:
                    models[model_id] = {'total': {'chars': 0, 'percent_ja': 0}}

                if item not in models[model_id]:
                    models[model_id][item] = {'chars': 0, 'percent_ja': 0}

                models[model_id][item]['chars'] += char_count
                models[model_id][item]['percent_ja'] = percent_ja
                models[model_id]['total']['chars'] += char_count

def update_total_percentage(models):
    for model_id in models:
        total_chars = models[model_id]['total']['chars']
        if total_chars > 0:
            total_ja_chars = sum(models[model_id][item]['chars'] * models[model_id][item]['percent_ja'] / 100
                                 for item in models[model_id] if item != 'total')
            models[model_id]['total']['percent_ja'] = (total_ja_chars / total_chars) * 100

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
        for filename in os.listdir(directory):
            if filename.endswith(".json") or filename.endswith(".jsonl"):
                file_path = os.path.join(directory, filename)
                model_id = filename.rsplit('.', 1)[0]
                parse_file(file_path, model_id, models)
    update_total_percentage(models)

    for model_id, data in sorted(models.items()):
        for item, values in data.items():
            if item != 'total':
                # print(f"Model: {model_id}, File: {item}, Chars: {values['chars']}, Japanese Percentage: {values['percent_ja']:.2f}%")
                pass
        print(f"Model: {model_id}, Total Chars: {data['total']['chars']}, Total Japanese Percentage: {data['total']['percent_ja']:.2f}%")

# Usage
base_directory = "./data/model_answers"
analyze_jsonl_files(base_directory)

