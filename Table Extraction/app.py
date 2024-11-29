import tabula
import pandas as pd

def extract_tables_from_pdf(pdf_path):
    # Extract tables from PDF into a list of DataFrames
    tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)

    # Print and return all the tables found
    for i, table in enumerate(tables):
        print(f"Table {i + 1}:\n", table)
        print("\n" + "="*50 + "\n")

    return tables

# Path to the PDF file
pdf_path = 'table.pdf'

# Extract tables
tables = extract_tables_from_pdf(pdf_path)

# Optionally, save the tables to CSV files
for i, table in enumerate(tables):
    table.to_csv(f'table_{i + 1}.csv', index=False)

