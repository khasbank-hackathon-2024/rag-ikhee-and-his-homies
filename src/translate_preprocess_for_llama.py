import json
from google.cloud import translate_v2 as translate
import os

# Set up Google Cloud credentials (use your service account key file)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'gen-lang-client-0314852010-3b65ae414b7f.json'

# Initialize the Google Translate client
translate_client = translate.Client()

# Define paths for input and output files
input_file_path = 'dl.jsonl'  # Path to your input file
output_file_path = 'translated_d.jsonl'  # Path for the new translated file

# Function to translate text
def translate_text(text, target_language='en'):
    if isinstance(text, bytes):
        text = text.decode('utf-8')
    # Translate the text to the target language (default is English)
    translation = translate_client.translate(text, target_language=target_language)
    return translation['translatedText']

# Open the input file (data.jsonl) and the output file (translated_d.jsonl)
with open(input_file_path, 'r', encoding='utf-8') as infile, open(output_file_path, 'w', encoding='utf-8') as outfile:
    for line in infile:
        try:
            # Parse the JSON data from the input file
            data = json.loads(line.strip())

            # Extract the "input" and "output" fields to translate
            input_text = data.get("instruction", "")
            output_text = data.get("output", "")

            # Translate the "input" and "output" text (you can specify other languages)
            translated_input = translate_text(input_text, target_language='mn')  # Example: translating to English
            translated_output = translate_text(output_text, target_language='mn')  # Example: translating to English

            # Create a new dictionary with translated fields
            new_data = {
                "instruction": translated_input,
                "output": translated_output
            }

            # Write the new translated data to the output file
            json.dump(new_data, outfile, ensure_ascii=False)
            outfile.write('\n')  # Write a newline to separate entries

        except json.JSONDecodeError:
            print("Error decoding JSON, skipping line.")
        except Exception as e:
            print(f"Error processing line: {e}")

print(f"Translation completed. File saved at: {output_file_path}")
