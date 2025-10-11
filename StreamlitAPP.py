import os
import json
import pandas as pd
import traceback
from dotenv import load_dotenv

from src.mcqgenerator.utils import read_file, get_table_data
from src.mcqgenerator.logger import logging

import streamlit as st
from langchain.callbacks import get_openai_callback
from src.mcqgenerator.MCQGenerator import generate_evaluate_chain

#load json file
with open(r'D:\GAI-Project\mcqgen\Response.json','r') as file:
    RESPONSE_JSON = json.load(file)

st.write("Streamlit version:", st.__version__)

#create a title for the app
st.title("MCQ Creator Application with Langchain and OpenAI")

#Create a form using st.form
with st.form("user_inputs"):
    #file upload
    uploaded_file=st.file_uploader("upload a PDF or txt file")

    #input fields
    mcq_count = st.number_input("No. of MCQs", min_value=3, max_value=50)

    #subject
    subject=st.text_input("Instert Subject", max_chars=20)

    #Quiz tone
    tone=st.text_input("Complexity level of Questions", max_chars=20, placeholder="Simple")

    #add button
    button = st.form_submit_button("Create MCQs")

    #check if the button is clicked and all fields have input

    if button and uploaded_file is not None and mcq_count and subject and tone:
        with st.spinner("Loading....."):
            try:
                text=read_file(uploaded_file)
                #count token and the cost of API call
                with get_openai_callback() as cb:
                    response = generate_evaluate_chain(
                        {
                            "text":text,
                            "number":mcq_count,
                            "subject":subject,
                            "tone":tone,
                            "response_json":json.dumps(RESPONSE_JSON)
                        }
                    )
                # st.write(response)
            except Exception as e:
                traceback.print_exception(type(e),e,e.__traceback__)
                st.error("error")
            
            else:
                print(f"Total Tokens: {cb.total_tokens}")
                print(f"Prompt Tokens: {cb.prompt_tokens}")
                print(f"Completion Tokens: {cb.completion_tokens}")
                print(f"Total Cost: {cb.total_cost}")
                if isinstance(response, dict):
                    quiz = response.get("quiz",None)
                    if quiz is not None:
                        table_data = get_table_data(quiz)
                        if table_data is not None:
                            df=pd.DataFrame(table_data)
                            df.index = df.index+1
                            st.table(df)
                            st.text_area(label="Review", value=response["review"])
                        else:
                            st.error("Error in the table data")
                    else:
                        st.error("Quiz is none")
                else:
                    st.write(response)