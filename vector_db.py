from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader,TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import * 

def faiss_vector_db():

    dir_loader = DirectoryLoader(
                            DATA_DIR_PATH,
                            glob='*.txt',
                            loader_cls=TextLoader
                        )
    docs = dir_loader.load()
    print("PDFs Loaded")
    
    txt_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=CHUNK_SIZE, 
                            chunk_overlap=CHUNK_OVERLAP
                        )
    inp_txt = txt_splitter.split_documents(docs)
    print("Data Chunks Created")

    hfembeddings = HuggingFaceEmbeddings(
                            model_name=EMBEDDER, 
                            model_kwargs={'device': 'cuda'}
                        )

    db = FAISS.from_documents(inp_txt, hfembeddings)
    db.save_local(VECTOR_DB_PATH)
    print("Vector Store Creation Completed")

if __name__ == "__main__":
    faiss_vector_db()