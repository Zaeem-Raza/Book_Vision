import camelot
import json


def extract_tables_to_json(pdf_path, output_json_file="extracted_tables.json"):
    # Use 'stream' or 'lattice' depending on your PDF
    tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')

    all_tables = []
    for page_num, table in enumerate(tables):
        page_data = {"page": page_num + 1, "table": table.df.values.tolist()}
        all_tables.append(page_data)

    with open(output_json_file, "w", encoding="utf-8") as json_file:
        json.dump(all_tables, json_file, indent=4, ensure_ascii=False)

    print(f"Data extracted and saved to {output_json_file}")


pdf_path = "physicschapter3.pdf"
output_json_file = "file_json_form.json"
extract_tables_to_json(pdf_path, output_json_file)
