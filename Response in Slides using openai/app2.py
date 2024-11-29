import openai  # type: ignore
import secrets
import os
import subprocess
# import fitz  # PyMuPDF # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
import faiss  # type: ignore
import numpy as np  # type: ignore
import chunks
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
dimension = 384
index = faiss.IndexFlatL2(dimension)
embeddings = []
image_dir = "extracted_images"
os.makedirs(image_dir, exist_ok=True)
MyChunks = chunks.Mychunks


def embed_and_index_chunks():
    global embeddings
    if embeddings:
        print("Chunks already embedded and indexed.")
        return
    for chunk in MyChunks:
        text = chunk['text']
        embedding = embedding_model.encode(text)
        embeddings.append(embedding)
        index.add(np.array([embedding]).astype("float32"))


def retrieve_chunks(query, top_k=10):
    query_embedding = embedding_model.encode(query)
    distances, indices = index.search(
        np.array([query_embedding]).astype("float32"), top_k)
    return [MyChunks[i] for i in indices[0]]


def generate_answer(query, retrieved_chunks):
    context = "\n\n".join(
        [f"Page {chunk['page']}: {chunk['text']}" for chunk in retrieved_chunks])
    return f"Answer the question based on the following textbook content:\n\n{context}\n\nQuestion: {query}\nAnswer:"


def run_rag_pipeline(pdf_path, query):
    if not MyChunks:
        return "No data available."
    embed_and_index_chunks()
    retrieved_chunks = retrieve_chunks(query)
    return generate_answer(query, retrieved_chunks)


def load_openai_key():
    return secrets.OPEN_AI_SECRET_KEY


def call_openai_chat(prompt, model="gpt-3.5-turbo"):
    openai.api_key = load_openai_key()
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "system", "content": "You are an expert in LaTeX and Beamer presentations."},
                      {"role": "user", "content": prompt}],
            max_tokens=4000,
            temperature=0.7
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"An error occurred: {e}"


def save_to_file(filename, content):
    with open(filename, 'w') as f:
        f.write(content)


def compile_latex_to_pdf(tex_file):
    try:
        subprocess.run(["pdflatex", tex_file], check=True)
        pdf_file = tex_file.replace('.tex', '.pdf')
        return pdf_file if os.path.exists(pdf_file) else None
    except subprocess.CalledProcessError as e:
        print(f"Error compiling LaTeX: {e}")
        return None


def extract_latex_code(raw_response):
    lines = raw_response.split("\n")
    in_code_block = False
    cleaned_lines = []
    for line in lines:
        if line.strip().startswith("```latex"):
            in_code_block = True
            continue
        if line.strip() == "```":
            in_code_block = False
            continue
        if in_code_block:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


if __name__ == "__main__":
    # Step 1: Get the user's query
    user_prompt = input("Enter your prompt: ")

    # Step 2: Retrieve relevant chunks from the textbook
    pdf_path = "Physics 9.pdf"
    answer = run_rag_pipeline(pdf_path, user_prompt)

    # Step 3: Generate LaTeX Beamer code using OpenAI
    beamer_prompt = (
        f"Create a detailed LaTeX Beamer presentation on the following topic:\n\n"
        f"{answer}\n\n"
        "Include equations, bullet points, and TikZ-based diagrams to illustrate concepts. Use only TikZ to draw shapes or vectors "
        "instead of relying on external image files. Ensure the output is ready-to-compile Beamer code."
    )
    raw_beamer_code = call_openai_chat(beamer_prompt)

    # Step 4: Extract valid LaTeX code
    beamer_code = extract_latex_code(raw_beamer_code)

    # Step 5: Save LaTeX code to a .tex file
    tex_filename = "response.tex"
    save_to_file(tex_filename, beamer_code)

    # Step 6: Compile the .tex file into a PDF
    pdf_filename = compile_latex_to_pdf(tex_filename)
    if pdf_filename:
        print(f"PDF generated: {pdf_filename}")
    else:
        print("Failed to generate PDF.")
