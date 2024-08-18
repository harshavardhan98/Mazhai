from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import (Language,RecursiveCharacterTextSplitter)
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_google_vertexai import VertexAI
from langchain_google_vertexai import VertexAIEmbeddings
import vertexai 
from pprint import pprint
from langchain.vectorstores import FAISS
import faiss
import numpy as np
from langchain.docstore.in_memory import InMemoryDocstore


class ChatPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        self.model = ChatOllama(model="llama3.1")
        PROJECT_ID = "llm-test-432914"  # @param {type:"string"}
        REGION = "us-central1"  # @param {type:"string"}
        vertexai.init(project=PROJECT_ID, location=REGION)
        self.model = VertexAI(model_name="gemini-1.5-flash")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.code_splitter = RecursiveCharacterTextSplitter.from_language(Language.PYTHON, chunk_size=1024, chunk_overlap=5)
        self.prompt = PromptTemplate.from_template(
            """
            You are an assistant for answering questions. Use the following context to respond to the question. If you don't know the answer, simply say you don't know. Use a maximum of three sentences and be concise in your response.
            Question: {question}
            Context: {context}
            Answer:"

            Let me know if you need further help!
            """
        )

    def ingest_folder(self, folder_path: str):
        chunks = []
        print("Folder path:" + str(folder_path))
        loader = GenericLoader.from_filesystem(folder_path, glob="*", suffixes=[".py"], parser=LanguageParser())
        docs = loader.load()
        for document in docs:
            pprint(document.metadata)

        chunks = self.code_splitter.split_documents(docs)
        # vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
        embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@003").embed_documents([chunk.page_content for chunk in chunks])
        dimension = len(embeddings[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings))
        docstore = InMemoryDocstore(dict((i, chunk) for i, chunk in enumerate(chunks)))
        index_to_docstore_id = {i: i for i in range(len(chunks))}

        # vector_store = Chroma.from_documents(documents=chunks, embedding=VertexAIEmbeddings(model_name="textembedding-gecko@003"))
        vector_store = FAISS(embedding_function=VertexAIEmbeddings(model_name="textembedding-gecko@003"),
                                  index=index,
                                  docstore=docstore,
                                  index_to_docstore_id=index_to_docstore_id)
        self.retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5,
            },
        )

        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())


    def ingest(self, file_path: str, ext: str):
        chunks = []
        print("\n\nfilepath: " + file_path + "str: " + ext + "\n\n")
        if ext.endswith("pdf"):
            docs = PyPDFLoader(file_path=file_path).load()
            chunks = self.text_splitter.split_documents(docs)
            print(chunks)
            chunks = filter_complex_metadata(chunks)
            print(chunks)
        elif ext.endswith("java"):
            print("\n\nharsha 1\n\n")
            with open(file_path, 'r') as file:
                code = file.read()
                print("\n\nharsha2 - code: " + str(code) + "\n\n")
                # docs = [{"content": code, "metadata": {"source": file_path}}]
                chunks = self.code_splitter.create_documents([code])            
                print("\n\nharsha: " + str(chunks) + "\n\n")

        print("\n\nharsha 2\n\n")    
        vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
        # docs = PyPDFLoader(file_path=pdf_file_path).load()
        self.retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5,
            },
        )

        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())

    def ask(self, query: str):
        if not self.chain:
            return "Please, add a PDF document first."

        return self.chain.invoke(query)

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None