import openai
import my_secrets
import os
import subprocess
from sentence_transformers import SentenceTransformer  
import faiss  
import numpy as np  
import chunks

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
dimension = 384
index = faiss.IndexFlatL2(dimension)
embeddings = []
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
    return my_secrets.OPEN_AI_SECRET_KEY

def call_openai_chat(prompt, model="gpt-3.5-turbo"):
    openai.api_key = load_openai_key()
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "system", "content": "You are an expert in physics."},
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

if __name__ == "__main__":

    # Step 1: Get the user's query
    user_prompt = input("Enter your prompt: ")

    # Step 2: Retrieve relevant chunks from the textbook
    pdf_path = "Physics 9.pdf"
    answer = run_rag_pipeline(pdf_path, user_prompt)

    # Step 3: Call OpenAI to refine the text
    refinement_prompt = (
        f"Read the following text and make the answer better:\n\n"
        f"{answer}"
    )
    refined_answer = call_openai_chat(refinement_prompt)

    # Step 4: Print and save the refined answer
    print("\nGenerated Answer:")
    print(refined_answer)

    output_filename = "text_response.txt"
    save_to_file(output_filename, refined_answer)

    print(f"\nRefined answer saved to {output_filename}.")
    
    
    # use the refined answer to create a presentation, call openai to generate slides
    # Step 5: Call OpenAI to generate slides
    slide_prompt = (
        f"Create a beamer presentation slides based on the following text:\n\n"
        f"{refined_answer}"
    )
    slide_content = call_openai_chat(slide_prompt)
    # Step 6: Save the slide content to a file
    slide_filename = "presentation.tex"
    save_to_file(slide_filename, slide_content)
    print(f"\nPresentation slides saved to {slide_filename}.")
