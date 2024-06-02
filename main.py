import os
import requests
import dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Weaviate
import weaviate
from weaviate.embedded import EmbeddedOptions
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

debug = False

def downloadDocs():
    # download state of the union address dataset from the url, if the file doesn't exit locally
    file = './state_of_the_union.txt'
    if not os.path.isfile(file):
        print('Downloading dataset file...')
        url = "https://raw.githubusercontent.com/hwchase17/chroma-langchain/master/state_of_the_union.txt"
        res = requests.get(url)
        with open("state_of_the_union.txt", "w") as f:
            f.write(res.text)

    # use langchain's TextLoader to load the documents
    loader = TextLoader(file)
    documents = loader.load()

    if debug == True:
        print(documents)

    return documents

def chunkDocs(documents):
    # chunk docs using langchain's CharacterTextSplitter with appropriate chunk size and overlap window to preserve continuity
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    if debug == True:    
        # Print the chunked texts
        for i, text in enumerate(chunks):
            print(f"Chunk {i+1}:\n{text.page_content}\n")

    return chunks

def embedChunks(chunks):
    vectorstore = None

    client = weaviate.Client(
        embedded_options = EmbeddedOptions()
    )

    # if not client.data_object:
    vectorstore = Weaviate.from_documents(
        client = client,
        documents = chunks,
        embedding = OpenAIEmbeddings(),
        by_text = False
    )
    # else:
        # vectorstore = Weaviate(client, index_name='main', text_key='text')

    return vectorstore

def rag(question, vectorstore):
    retriever = vectorstore.as_retriever()

    template = """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise.
    Question: {question} 
    Context: {context} 
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    print(prompt)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    output = rag_chain.invoke(question)

    print(f"""
    ***
    
          
    {output}
    
    
    ***
    """)


if __name__ == "__main__":
    # load env variables
    dotenv.load_dotenv()

    # Ingest docs
    documents = downloadDocs()
    chunks = chunkDocs(documents)
    vectorstore = embedChunks(chunks)

    # RAG on ingested docs - (R)etrieve docs, (A)ugment with prompt, (G)enerate response using llm)
    question = "What did the president say about Vladimir Putin?"
    rag(question, vectorstore)

