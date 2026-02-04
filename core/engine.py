from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.chains import RetrievalQA

class RAGEngine:
    def __init__(self, openai_key: str, pinecone_key: str, index_name: str):
        # 1. Configurar Embeddings (con la API Key del usuario)
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
        
        # 2. Conectar con el índice de Pinecone
        self.vector_store = PineconeVectorStore(
            index_name=index_name,
            embedding=self.embeddings,
            pinecone_api_key=pinecone_key
        )
        
        # 3. Configurar el LLM
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview", 
            temperature=0, 
            openai_api_key=openai_key
        )

    def ask(self, question: str, k: int = 3):
        # Creamos la cadena de QA (Pregunta-Respuesta)
        # 'stuff' significa que mete todo el contexto recuperado en el prompt
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": k})
        )
        
        response = qa_chain.invoke(question)
        return response["result"]