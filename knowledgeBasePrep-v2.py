# creates the vectors from chuncks of an input data source and saves them in a local knowledge base

# to reference envinronment variables
from dotenv import load_dotenv
# to split input data source into chuncks
from langchain.text_splitter import CharacterTextSplitter
# to load pdf
from PyPDF2 import PdfReader
# to convert text chunks into embeddings
from langchain.embeddings.openai import OpenAIEmbeddings
# for vector store of embeddings and chunks
from langchain.vectorstores import FAISS
# for UX
import streamlit as st




def main():

    load_dotenv()

    st.set_page_config(page_title='Load transcripts')
    st.header('Load your transcript')

    company = st.text_input('Company')
    pdf = st.file_uploader("Upload your PDF", type = "pdf")

    filepathprefix = 'C:\\MIDS\\'


    if pdf is not None:
        # extract text and create corpus
        st.write(f'Company name is {company}')
        pdf_reader = PdfReader(pdf)
        corpus = ''
        for page in pdf_reader.pages:
                corpus += page.extract_text()
        
        st.write(f'Corpus length: {len(corpus)}')
        
        # splitting corpus into chunks
        text_splitter = CharacterTextSplitter(separator='\n'
                                        , chunk_size = 1000
                                        , chunk_overlap = 200
                                        , length_function = len
                                        )
        chunks = text_splitter.split_text(corpus)
        st.write(f'Chunks length: {len(chunks)}')
        
        embeddings = OpenAIEmbeddings(model='text-embedding-ada-002', deployment='embeddings',chunk_size=1)

        # save just the embedding into a text file
        # saveEmbeddings = embeddings.embed_documents(chunks)
        # with open(f'{filepathprefix}{company}-embeddings.txt', 'w') as file:
        #         for embedding in saveEmbeddings:
        #                 file.write(' '.join(str(value) for value in embedding))
        #                 file.write('\n')
        
        # create the vector db of embeddings and corresponding chunks
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # make local copy of the knowledgeBase
        knowledge_base.save_local(f'{filepathprefix}localFAISSkb', index_name=company)
        st.write("Load done")


main()

    
