# from langchain.document_loaders import CSVLoader
from langchain_community.document_loaders import CSVLoader
import chardet
from langchain_community.embeddings import HuggingFaceEmbeddings
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAI
import os
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
llm = GoogleGenerativeAI(model="gemini-pro")
def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read()
    return chardet.detect(raw_data)['encoding']



embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

e = embeddings.embed_query("What is the capital of France?")
print(e[:5])

def create_vector_db():
    # Detect the file encoding
    file_path = "edtech_faqs.csv.csv"
    detected_encoding = detect_encoding(file_path)
    print(f"Detected encoding: {detected_encoding}")

    try:
        # Initialize the CSVLoader with the detected encoding
        loader = CSVLoader(file_path=file_path, encoding=detected_encoding)
        
        # Load the data
        data = loader.load()
        
        print("Successfully loaded the CSV file.")
        # print(data)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    vectordb = FAISS.from_documents(data, embeddings)
    print("Vector DB created successfully")
    vectordb.save_local("faiss_index.json")
    print("Vector DB saved successfully")

def get_qa_chain():
    vectordb = FAISS.load_local("faiss_index.json", embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever()

    prompt_template = """
    Given the following context and a question, generate n answer based on the context. In the 
    try to give the most relevant answer. If the answer is not present in the context, 
    kindly say "I don't know". Don't try to make up an answer.

    Context: {context}

    Question: {question}
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(llm=llm, 
                                        chain_type="stuff", 
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": prompt})

    
    return qa_chain


if __name__ == "__main__":
    chain = get_qa_chain()
    print(chain.invoke({"query": "Do you have a course on Javascript?"}))
    