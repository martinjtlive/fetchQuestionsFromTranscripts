# prompt template execution to find questions by analysts 

# to reference envinronment variables
from dotenv import load_dotenv
# to load knowledge base vector sort
from langchain.vectorstores import FAISS
# to convert text chunks into embeddings
from langchain.embeddings.openai import OpenAIEmbeddings
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

    
    filepathlocalFAISS = 'C:\\MIDS\\localFAISSkb\\'

    # get company name from local FAISS
    files = os.listdir(filepathlocalFAISS)
    company = [file.removesuffix('.faiss') for file in files if file.endswith('.faiss')]
    
    # make list into pandas series as that is acceptable argument for 
    # streamlist select box
    companySequence = pd.Series(company) # ', '.join(company)
    
        


    st.set_page_config(page_title='Top questions')
    st.header('Know the important questions raised by analysts during Company Earnings calls')

    companySelected = st.selectbox(
    'Select the Company',
    companySequence
    )

    questionAreas = ['Financial results', 'Partnerships', 'Market conditions', 'Product announcement']
    questionAreasAsStringSequence = ', '.join(questionAreas)
    template = ''' For this {companyOfChoice} transcript, what are the most interesting/important questions asked by analysts 
    on following topics: {topics}. Asssume unknown attendees also as analysts. 
    If there are no questions for a topic, respond with 'None' for the corresponding topic list item.
    '''

    prompt = PromptTemplate(
    input_variables=["companyOfChoice", "topics"],
    template=template
    )

    if companySelected!='':
        embeddings = OpenAIEmbeddings(model='text-embedding-ada-002', deployment='embeddings',chunk_size=1)

        # load knowledge base from local
        current_kb = FAISS.load_local(filepathlocalFAISS, embeddings, index_name=companySelected)
        docs = current_kb.similarity_search (f'For this {companySelected} transcript, what are the most interesting/important questions asked by analysts \
                                             on following topics: {questionAreasAsStringSequence}. Asssume unknown attendees also as analysts.')
        #st.write(docs)

        # set llm
        llm = AzureOpenAI(deployment_name='gpt-3')
        
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run({'companyOfChoice':companySelected,  'topics':questionAreas})
        st.write(response)







    # # Load embeddings from the text file
    # embeddings = OpenAIEmbeddings()
    
    
    # current_kb = FAISS.load_local('localFAISSkb', embeddings, index_name='Tesla')
     

    # # print(f'current_kb type: {type(current_kb)}') # current_kb type: <class 'langchain.vectorstores.faiss.FAISS'>
    # # print(f'Dictionary store for kb: {current_kb.docstore.__dict__}')

    # # start here: prompt template

if __name__=='__main__': main()
