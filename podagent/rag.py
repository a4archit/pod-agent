#---------------------------------------------------------------------------------------------------
# Dependencies 
#---------------------------------------------------------------------------------------------------


from typing import Literal, Dict, List, Annotated, Optional





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
        self.__embedding_model = ""
        self.__qdrant_collection_name = "conv_agentic_rag_db"




        
    def __repr__(self):
        return "ConversationalAgenticRAG()"
    


    def __load_pdf(self):
        """ It will load pdf content and update in the variable """

        pass 


    def __load_embedding_model(self):
        """ It will load embedding model """
        pass 



    def __indexing(self):
        pass



    def __create_qdrant_vector_db(self):
        pass 



    def fetch_docs(
            self, 
            query: str,
            tok_k: int = 3,
            conversational: bool = True,
            last_five_chat_messages: Optional[List[str]] = None
        ) -> List[str]:

        pass 







