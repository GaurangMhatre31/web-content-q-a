from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM  # Updated import
from playwright.async_api import async_playwright
import asyncio

app = Flask(__name__)
CORS(app)

vector_store = None
OLLAMA_MODEL = "mistral"

async def scrape_website(url):
    async with async_playwright() as p:
        browser = await p.firefox.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url, wait_until="domcontentloaded")
        await asyncio.sleep(3)
        html_content = await page.content()
        await browser.close()
    
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    
    if not text.strip():
        return "Error: No readable text found on the page."
    
    print("Extracted Content:", text[:500])
    return text

def create_vector_store(content):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_text(content)
    
    if not docs:
        raise ValueError("No text chunks created. Ensure the content is not empty.")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    try:
        return FAISS.from_texts(docs, embeddings)
    except Exception as e:
        print(f"Error creating FAISS vector store: {e}")
        return None

def answer_question(query, vector_store):
    retriever = vector_store.as_retriever()
    docs = retriever.invoke(query)  # Updated method
    context = "\n".join([doc.page_content for doc in docs])
    
    llm = OllamaLLM(model=OLLAMA_MODEL, base_url="http://localhost:11434")  # Updated class
    return llm.invoke(f"Answer this question using the given context:\n{context}\n\n{query}")  # Updated method

@app.route('/scrape', methods=['POST'])
def scrape():
    global vector_store
    data = request.json
    url = data.get("url")
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    content = asyncio.run(scrape_website(url))
    
    if "Error" in content:
        return jsonify({"error": content}), 400

    vector_store = create_vector_store(content)
    
    if vector_store is None:
        return jsonify({"error": "Failed to create vector store. No valid content found."}), 500
    
    return jsonify({"message": "Content fetched and indexed successfully!"})

@app.route('/ask', methods=['POST'])
def ask():
    global vector_store
    if not vector_store:
        return jsonify({"error": "No content indexed. Please fetch content first."})
    data = request.json
    query = data.get("query")
    if not query:
        return jsonify({"error": "No query provided"}), 400
    answer = answer_question(query, vector_store)
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(debug=True)
