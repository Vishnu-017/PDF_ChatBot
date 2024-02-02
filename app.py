
from flask import Flask, render_template, request
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os

from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__,template_folder='template')

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

@app.route('/')
def home():
    
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_file():
    print("hello0")
    pdf = request.files['pdf']
    print(pdf)

    if pdf.filename != '':
        try:
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            print(text)

            char_text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
            text_chunks = char_text_splitter.split_text(text)

            embeddings = OpenAIEmbeddings()
            docsearch = FAISS.from_texts(text_chunks, embeddings)
            llm = OpenAI(model="gpt-3.5-turbo-instruct")
            chain = load_qa_chain(llm, chain_type="stuff")

            query = request.form['query']
            if query:
                docs = docsearch.similarity_search(query)
                response = chain.run(input_documents=docs, question=query)

                return render_template('index.html', response=response)
        
        except Exception as e:
            return render_template('index.html', error=f"Error reading PDF: {str(e)}")

    return render_template('index.html', error="Invalid file")

if __name__ == '__main__':
    app.run(debug=True)
