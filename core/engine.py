from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain, create_stuff_documents_chain   
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

class RAGEngine:
    def __init__(self, openai_key: str, pinecone_key: str, index_name: str):
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
        self.vector_store = PineconeVectorStore(
            index_name=index_name,
            embedding=self.embeddings,
            pinecone_api_key=pinecone_key
        )
        self.llm = ChatOpenAI(model="gpt-4-turbo-preview", openai_api_key=openai_key)

    def ask(self, question: str, k: int = 3):
        # 1. Define the System Prompt (Modern way to "Stuff" context)
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "\n\n"
            "{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        # 2. Create the 'Combine Documents' chain (the prompt + LLM part)
        combine_docs_chain = create_stuff_documents_chain(self.llm, prompt)

        # 3. Create the final 'Retrieval' chain (linking the DB to the prompt)
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

        # 4. Invoke using the new standard syntax
        response = rag_chain.invoke({"input": question})
        
        # The answer now lives in the "answer" key instead of "result"
        return response["answer"]
    
    def ingest_file(self, file_path: str):
        """
        Carga un archivo, lo divide en fragmentos y lo sube a la base de datos vectorial.
        """
        
        ext = os.path.splitext(file_path)[-1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".txt":
            loader = TextLoader(file_path)
        elif ext == ".md":
            loader = UnstructuredMarkdownLoader(file_path)
        else:
            raise ValueError(f"Extensión de archivo '{ext}' no soportada.")

        # 2. Extraer el contenido del documento
        documents = loader.load()

        # 3. Fragmentación (Chunking)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=150  # Solapamiento para mantener contexto entre fragmentos
        )
        final_chunks = text_splitter.split_documents(documents)

        # 4. Ingesta en Pinecone
        # Esto genera los embeddings y los guarda automáticamente
        self.vector_store.add_documents(final_chunks)

        return len(final_chunks)