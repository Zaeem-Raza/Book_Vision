import os
import openai
import PyPDF2
import panel as pn
import file
# Initialize Panel for file upload
pn.extension()


# Set up OpenAI API Key
openai.api_key = file.OPEN_AI_SECRET_KEY


class UpldFile:
    def __init__(self):
        self.widget_file_upload = pn.widgets.FileInput(
            accept='.pdf', multiple=False)
        self.widget_file_upload.param.watch(self.save_filename, 'filename')

    def save_filename(self, _):
        if len(self.widget_file_upload.value) > 2e6:
            print("File too large. 2 MB limit.")
        else:
            file_path = './example_files/' + self.widget_file_upload.filename
            with open(file_path, 'wb') as f:
                f.write(self.widget_file_upload.value)
            print(f"File saved: {file_path}")
            self.process_pdf(file_path)

    def process_pdf(self, file_path):
        # Extract text from the PDF
        extracted_text = self.extract_text_from_pdf(file_path)
        if extracted_text:
            # Use OpenAI to summarize the extracted text
            self.summarize_text(extracted_text)

    def extract_text_from_pdf(self, pdf_path):
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in range(len(reader.pages)):
                    text += reader.pages[page].extract_text()
                return text.strip()
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""

    def summarize_text(self, text):
        try:
            response = openai.Completion.create(
                model="gpt-3.5-turbo",
                prompt=f"Summarize the following text:\n\n{text}",
                max_tokens=150,
                temperature=0.5
            )
            summary = response.choices[0].text.strip()
            print(f"Summary: {summary}")
        except openai.error.OpenAIError as e:
            print(f"Error with OpenAI API: {e}")


# Instantiate file uploader and start the process
file_uploader = UpldFile()

# Display the file upload widget
file_uploader.widget_file_upload
