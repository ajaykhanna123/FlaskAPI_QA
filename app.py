
from flask import Flask, request, jsonify, session
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import CTransformers
import sys
import datetime
from langchain.chains.question_answering import load_qa_chain


@app.route('/', methods=['GET'])
def status():
    return jsonify({'result': "App is deployed"}), 200


def save_document_to_text_file(document_text):
    # Define the data to be written


    # Specify the file path
    file_path = "document_text.txt"

    # Open the file in write mode ('w')
    # This will create a new file or overwrite the existing one with the same name.
    # If you want to append data to an existing file, use 'a' mode instead of 'w'.
    try:
        with open(file_path, 'w') as file:
            # Write the data to the file
            file.write(document_text)
        print("document text saved to file - "+ file_path)
        return {"status":"document file saved , Call /build_model to get output"}
    except:
        exit(0)

    # The file will be automatically closed after the 'with' block


@app.route('/inferencing/question_response', methods=['POST'])
def qa_model():
    start_time = datetime.datetime.now()

    data = request.get_json()

    # Check if both inputs are provided
    if 'document_text' not in data:
        return jsonify({'error': 'Document text not passed'}), 400
    document_text=data['document_text']
    try:
        save_document_to_text_file(document_text)
    except:
        print("issue saving file")
        exit()
    loader = TextLoader("document_text.txt")
    documents = loader.load()

    if 'question' not in data:
        return jsonify({'error': 'question text not passed'}), 400

    question = data['question']


    llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q4_0.bin",
                        model_type="llama",
                        config={'max_new_tokens': 500,
                                'temperature': 0.1})

    start_time = datetime.datetime.now()
    chain = load_qa_chain(llm=llm, chain_type="stuff")


    result = chain.run(input_documents=documents, question=question)
    print(result)
    end_time = datetime.datetime.now()
    print("time taken for answering question- ", end_time - start_time)

    return jsonify({"result": result, "latency": str(end_time - start_time)}), 200

if __name__ == '__main__':
    app.run(debug=True)
