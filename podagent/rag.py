#---------------------------------------------------------------------------------------------------
# Dependencies 
#---------------------------------------------------------------------------------------------------

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from typing import Literal, Dict, List, Annotated, Optional

# local 
from pdf_processor import PDFProcessor
from configs import PodagentConfigs
from faiss_manager import setup_temp_faiss

# built in
from configs import QDRANT_CLIENT_URL, EMBEDDING_MODEL





# instance of pdf processor 
pdf_process = PDFProcessor()





#---------------------------------------------------------------------------------------------------
# Class scripts
#---------------------------------------------------------------------------------------------------

class ConversationalAgenticRAG:

    SUPPORTED_FILES = ["pdf"]



    def __init__(
            self,
            file_path: str 
        ):
        
        """ constructor """

        self.__file_path = file_path
        self.__chunk_size = 1000
        self.__overlap_size = 200
        self.__embedding_model = None
        self.__vector_store = None 
        self.__vector_store_manager = None 
        self.__vector_store_name = None

        self.__pdf_texts: List[Document] = None

        # setup everything here
        self.__indexing()



    def get_vector_store_manager(self):
        return self.__vector_store_manager
    

    def get_vector_store(self):
        return self.__vector_store_manager.get_vector_store(self.__vector_store_name)


        
    def __repr__(self):
        return "ConversationalAgenticRAG()"
    



    def __load_pdf_texts(self):
        """ It will load pdf content and update in the variable """
        self.__pdf_texts = pdf_process.process_pdf(pdf_path=PodagentConfigs.pdf_path)
        




    def __load_embedding_model(self):
        """ It will load embedding model """
        self.__embedding_model = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL
        )






    def __indexing(self):
        """ Indexing - A step in RAG """

        # loading embedding model
        self.__load_embedding_model()

        # loading content 
        self.__load_pdf_texts()

        # setting up faiss
        self.__setup_faiss()

        # updating vector database
        self.__vector_store.add_documents(self.__pdf_texts)

        print("\n\nALL SETUP !")









    def __setup_faiss(self):

        self.__vector_store, self.__vector_store_name, self.__vector_store_manager = setup_temp_faiss(
        embeddings=self.__embedding_model,
        store_name="vector_faiss",
        save_to_disk=True
    )






    def fetch_docs(
            self, 
            query: str,
            top_k: int = 3,
            conversational: bool = True,
            last_five_chat_messages: Optional[List[str]] = None
        ) -> List[Document]:

        # print("Enter in ConversationalAgenticRAG().fetch_docs()")

        docs = self.__vector_store.similarity_search(query=query, k=top_k)

        # print(f"Docs returned (from rag.py), returned docs -> {docs}")

        return docs











if __name__ == "__main__":

    rag = ConversationalAgenticRAG(file_path=PodagentConfigs.pdf_path)

    # manager = rag.get_vector_store_manager()

    # print(manager.list_stores())
    # print(manager.list_disk_stores())

    # store = rag.get_vector_store()

    # # print(type(store))
    # # print(dir(store))
    # print()

    while True:

        user = input("Your query: ")

        if user.strip().lower() == "exit" :
            break 

        docs = rag.fetch_docs(query=user, conversational=False)

        print(docs)




