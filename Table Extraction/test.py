import pdfplumber


def extract_and_save_pdf_data(pdf_path, output_text_file="extracted_data.txt"):
    with pdfplumber.open(pdf_path) as pdf:
        with open(output_text_file, "w", encoding="utf-8") as output_file:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    output_file.write(f"--- Page {page_num + 1} Text ---\n")
                    output_file.write(text + "\n\n")
                table = page.extract_table()
                if table:
                    output_file.write(f"--- Page {page_num + 1} Table ---\n")
                    for row in table:
                        row = [
                            str(cell) if cell is not None else "" for cell in row]
                        output_file.write("\t".join(row) + "\n")
                    output_file.write("\n")

    print(f"Data extracted and saved to {output_text_file}")


pdf_path = "table.pdf"
output_text_file = "extracted_table.txt"

extract_and_save_pdf_data(pdf_path, output_text_file)
