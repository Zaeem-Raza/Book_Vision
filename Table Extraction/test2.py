import pdfplumber
import json


def extract_tables_to_json(pdf_path, output_json_file="extracted_tables.json"):
    with pdfplumber.open(pdf_path) as pdf:
        all_tables = []

        for page_num, page in enumerate(pdf.pages):
            table = page.extract_table()
            if table:
                page_data = {"page": page_num + 1, "table": []}
                for row in table:
                    page_data["table"].append(row)
                all_tables.append(page_data)

    with open(output_json_file, "w", encoding="utf-8") as json_file:
        json.dump(all_tables, json_file, indent=4, ensure_ascii=False)

    print(f"Data extracted and saved to {output_json_file}")


pdf_path = "physicschapter3.pdf"
if (not pdf_path):
    print("Please provide a valid pdf path")
output_json_file = "file_json_form.json"

extract_tables_to_json(pdf_path, output_json_file)
