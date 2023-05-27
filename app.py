from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.vectorstores import Chroma
import pickle
import os
import openai
load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']


class QA:
    def __init__(self) -> None:
        load_dotenv()
        self.embeddings = pickle.load(open('embeddings.pkl', 'rb'))#load
        self.texts = None
        self.query =None
        self.context = None
        self.docsearch = None
        return
    def input_file(self):
        st.set_page_config(page_title="Upload file")
        st.header("Upload file")
        pdf = st.file_uploader("Upload your file", type='pdf')
        text = ""
        if pdf is None:
            return
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
        self.texts = text.split('.')
        if len(text)<1:
            return
        st.write('Building database\n')
        self.docsearch = Chroma.from_texts(self.texts, self.embeddings, metadatas=[{"source": f"{i}-pl"} for i in range(len(self.texts))])
        return 
    def process_query(self):
        self.query = st.text_input('Query')
        if not self.query:
            return
        result = self.docsearch.similarity_search_with_score(self.query)
        self.context = ''
        for res in result:
            self.context+=res[0].page_content
        return
    def inference(self):
        if not self.query:
            return
        st.write('Inferencing\n')
        response = openai.ChatCompletion.create(model='gpt-3.5-turbo',
                         messages=[{"role": "user", "content": f"{self.query} strictly based on this context - {self.context}"}])
        answer = response.choices[0].message.content
        st.write(answer)
        return 

if __name__=='__main__':
    inp = QA()
    inp.input_file()        
    inp.process_query()
    inp.inference()
        
    
