#---------------------------------------------------------------------------------------------------
# Dependencies 
#---------------------------------------------------------------------------------------------------

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from typing import Literal, Dict, List, Annotated, Optional
from langchain_community.vectorstores import FAISS

# local 
from pdf_processor import PDFProcessor
from configs import PodagentConfigs
from faiss_manager import FAISSTempManager

# built in
from configs import EMBEDDING_MODEL








#---------------------------------------------------------------------------------------------------
# Class scripts
#---------------------------------------------------------------------------------------------------

class ConversationalAgenticRAG:

    SUPPORTED_FILES = ["pdf"]



    def __init__(
            self,
            file_path: str,
            chunk_size: int = 1000,
            chunk_overlap: int = 200,
            vector_store_name: Optional[str] = "vec_db_faiss"
        ):
        
        """ constructor """

        self.__file_path: str = file_path 
        self.__embedding_model: GoogleGenerativeAIEmbeddings = None

        self.__vector_store_manager = FAISSTempManager()
        self.__vector_store_name: str = vector_store_name
        self.__vector_store: FAISS = None 


        # instance of pdf processor 
        self.__pdf_processor = PDFProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.__pdf_documents: List[Document] = None





    def get_vector_store_manager(self):
        return self.__vector_store_manager
    

    def get_vector_store(self):
        return self.__vector_store_manager.get_vector_store(self.__vector_store_name)


        
    def __repr__(self):
        return "ConversationalAgenticRAG()"
    



    def __load_pdf_texts(self):
        """ It will load pdf content and update in the variable """
        extracted_docs = self.__pdf_processor.process_pdf(pdf_path=PodagentConfigs.pdf_path)

        self.__pdf_documents = extracted_docs[:99] # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Glue code due to free quota limit, learn more about it in local_dev.md
        




    def __load_embedding_model(self):
        """ It will load embedding model """
        self.__embedding_model = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL
        )






    def indexing(self):
        """ Indexing - A step in RAG """

        # loading embedding model
        self.__load_embedding_model()

        # loading content 
        self.__load_pdf_texts()

        # setting up faiss
        self.__setup_faiss()

        print(f"\n\nVector Store successfully saved `{self.__vector_store_name}`, along with {len(self.__vector_store.docstore._dict)} chunks.")






    def load_vector_store(
            self, 
            vector_store_name: Optional[str] = None 
        ):

        """
        Loading an existing vector store.
        
        :param self: Description
        :param vector_store_name: By default it will be `vec_db_faiss`
        :type vector_store_name: Optional[str]
        """

        if not vector_store_name:
            vector_store_name = self.__vector_store_name

        # loading embedding model
        self.__load_embedding_model()

        self.__vector_store = self.__vector_store_manager.load_vector_store(
            store_name=vector_store_name,
            embeddings=self.__embedding_model
        )

        print(f"\nVector store loaded ({self.__vector_store_name}) successfully along with {len(self.__vector_store.docstore._dict)}")

        






    def __setup_faiss(self):

        self.__vector_store, self.__vector_store_name = self.__vector_store_manager.create_vector_store(
            embeddings=self.__embedding_model,
            store_name=self.__vector_store_name,
            documents=self.__pdf_documents,
            save_to_disk=True 
        )





    def fetch_docs(
            self, 
            query: str,
            top_k: int = 3,
            conversational: bool = True,
            last_five_chat_messages: Optional[List[str]] = None
        ) -> List[Document]:

        if not self.__vector_store:
            raise(AttributeError("No vector store! Create a new or load an existing FAISS vector store before calling `ConversationalAgentRAG.fetch_docs()`"))

        docs = self.__vector_store.similarity_search(query=query, k=top_k)

        # print(f"Docs returned (from rag.py), returned docs -> {docs}")

        return docs











if __name__ == "__main__":

    from dotenv import load_dotenv

    load_dotenv()

    rag = ConversationalAgenticRAG(file_path=PodagentConfigs.pdf_path)

    # rag.indexing()
    rag.load_vector_store()

    print(rag.get_vector_store_manager().list_disk_stores())

    # while True:

    #     user = input("Your query: ")

    #     if user.strip().lower() == "exit" :
    #         break 

    #     docs = rag.fetch_docs(query=user, conversational=False)

    #     print(docs)




