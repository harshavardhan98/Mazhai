from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import (Language,RecursiveCharacterTextSplitter)
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata


class ChatPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        self.model = ChatOllama(model="llama3.1")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.code_splitter = RecursiveCharacterTextSplitter.from_language(Language.JAVA, chunk_size=1024, chunk_overlap=5)
        self.prompt = PromptTemplate.from_template(
            """
            You are an assistant for answering questions. Use the following context to respond to the question. If you don't know the answer, simply say you don't know. Use a maximum of three sentences and be concise in your response.
            Question: {question}
            Context: {context}
            Answer:"

            Let me know if you need further help!
            """
        )

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