
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


@app.route('/', methods=['GET'])
def status():
    return jsonify({'result': "App is deployed"}), 200

@app.route('/inferencing/input', methods=['POST'])
def process_input():
    try:
        # Get the data from the POST request
        data = request.get_json()

        # Check if both inputs are provided
        if 'document_text' not in data :
            return jsonify({'error': 'Document text not passed'}), 400

        document_text = data['document_text']


        # Process your input data here
        # You can do whatever processing you want with input1 and input2

        # Example: Concatenate the inputs and return the result
        result=save_document_to_text_file(document_text)

        # Return the result as a JSON response
        return jsonify({'result': result}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20)
    text_chunks = text_splitter.split_documents(documents)
    print("text chunks")
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    # **Step 4: Convert the Text Chunks into Embeddings and Create a FAISS Vector Store***
    vector_store = FAISS.from_documents(text_chunks, embeddings)
    print("vector store", vector_store)




    if 'question' not in data:
        return jsonify({'error': 'question text not passed'}), 400

    question = data['question']


    llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q4_0.bin",
                        model_type="llama",
                        config={'max_new_tokens': 500,
                                'temperature': 0.1})
    template = """Use the following pieces of information to answer the user's question.
    If you dont know the answer just say you know, don't try to make up an answer.

    Context:{context}
    Question:{question}

    Only return the helpful answer below and nothing else
    Helpful answer
    """

    qa_prompt = PromptTemplate(template=template, input_variables=['context', 'question'])

    # start=timeit.default_timer()
    print("running question query --loading")
    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type='stuff',
                                        retriever=vector_store.as_retriever(search_kwargs={'k': 2}),
                                        return_source_documents=True,
                                        chain_type_kwargs={'prompt': qa_prompt})

    data = request.get_json()
    chat_history = []
    result = chain({'query': question, "chat_history": chat_history})
    end_time = datetime.datetime.now()
    print("time taken for answering question- ", end_time - start_time)

    return jsonify({"result":result['result'],"latency":str(end_time-start_time)}),200

if __name__ == '__main__':
    app.run(debug=True)
