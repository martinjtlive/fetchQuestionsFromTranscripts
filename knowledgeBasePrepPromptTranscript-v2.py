# creates the vectors from chuncks of an input data source and saves them in a local knowledge base
# Then using a prompt template to prompt the corpus to get response

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
# for file access
import os
# pandas to make list into a series
import pandas as pd
# create prompt template
from langchain import PromptTemplate
# get response from  Azure OpenAI
from langchain.llms import AzureOpenAI
from langchain.chains import LLMChain





def main():

    load_dotenv()

    st.set_page_config(page_title='Load transcripts', layout='wide')
    st.header('Load your transcript')

    filepathprefix = 'C:\\MIDS\\'
    filepathlocalFAISS = 'C:\\MIDS\\localFAISSkb\\'

    # layout with tabs

    tab1, tab2 = st.tabs(['KB Creation', 'Questions in Corpus'])

    with tab1:

        company = st.text_input('Company')
        pdf = st.file_uploader("Upload your PDF", type = "pdf")

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
                saveEmbeddings = embeddings.embed_documents(chunks)
                with open(f'{filepathprefix}\CompanyEmbeddings\{company}-embeddings.txt', 'w') as file:
                        for embedding in saveEmbeddings:
                                file.write(' '.join(str(value) for value in embedding))
                                file.write('\n')
                
                # create the vector db of embeddings and corresponding chunks
                knowledge_base = FAISS.from_texts(chunks, embeddings)

                # make local copy of the knowledgeBase
                knowledge_base.save_local(f'{filepathprefix}localFAISSkb', index_name=company)
                st.write("Load done")
        
        with tab2:
                # get company name from local FAISS
                files = os.listdir(filepathlocalFAISS)
                company = [file.removesuffix('.faiss') for file in files if file.endswith('.faiss')]
                
                # make list into pandas series as that is acceptable argument for 
                # streamlist select box
                companySequence = pd.Series(company) # ', '.join(company)
                
                # add 'None' as top item in companySequence
                companySequenceWNone = pd.concat([pd.Series(['None']), companySequence])

                # header and subheader details within this tab
                st.header('Top Questions')
                st.subheader('Know the important questions raised by analysts during Company Earnings calls')

                companySelected = st.selectbox(
                       'Select the Company', companySequenceWNone    )

                questionAreas = ['Financial results', 'Partnerships', 'Market conditions', 'Product announcement']
                questionAreasAsStringSequence = ', '.join(questionAreas)
                template = ''' For this {companyOfChoice} transcript - {docSubset}, what are the most interesting/important questions asked by analysts 
                on following topics: {topics}. Asssume unknown attendees also as analysts. 
                If there are no questions for a topic, respond with 'None' for the corresponding topic list item. 
                Ensure line break after each topic and questions to it. The questions are to be formatted as numbered list
                '''

                prompt = PromptTemplate(
                input_variables=["companyOfChoice", "docSubset","topics"],
                template=template
                )

                if companySelected!='None':
                       embeddings = OpenAIEmbeddings(model='text-embedding-ada-002', deployment='embeddings',chunk_size=1)
                                                     
                       # load knowledge base from local
                       current_kb = FAISS.load_local(filepathlocalFAISS, embeddings, index_name=companySelected)
                       docs = current_kb.similarity_search (f'For this {companySelected} transcript, what are the most interesting/important questions asked by analysts \
                                                        on following topics: {questionAreasAsStringSequence}. Asssume unknown attendees also as analysts.')
                       #st.write(docs)

                       # set llm
                       llm = AzureOpenAI(deployment_name='gpt-3')

                       chain = LLMChain(llm=llm, prompt=prompt)
                       response = chain.run({'companyOfChoice':companySelected, 'docSubset':docs,'topics':questionAreas})
                       st.write(response)
                       



if __name__ == '__main__': main()

    
