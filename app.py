import streamlit as st

from src.get_config import get_config
from src.preprocess_data import SingleInferenceDataPreprocessor
from transformers import BertForSequenceClassification
from src.classifier import SingleInferenceClassifier


#main body
st.title("Text classification of firm business activities")
    
MainTab, InfoTab = st.tabs(["Main", "Info"])

with MainTab:
    st.write("""
        This text classifier predicts the SSIC code for a firm based on free-text descriptions of their business activities.
        It is trained for usage specifically on firms in the sea transport industry in Singapore. 
    """)
    st.write('')
    text_input = st.text_input("Enter firm business activity description.")

    if st.button('Submit'):
        config = get_config(path='/config/main.yaml')
        
        inf_preprocessor = SingleInferenceDataPreprocessor(config=config, text=text_input)
        inf_texts = inf_preprocessor.process()
        
        ensembled_model = BertForSequenceClassification.from_pretrained(config['model_ensemble']['ensemble_model_directory'])
        
        ssic_classifier = SingleInferenceClassifier(config=config, tokenized_texts_for_pred=inf_texts, model=ensembled_model)
        preds_df, top_subsector = ssic_classifier.get_predictions(top_n=5)
        
        #display results
        st.dataframe(preds_df)
        st.write(f'The firm likely falls under the following sea transport subsector: {top_subsector}.')

with InfoTab:
    st.subheader('Project Motivation')
    st.write("""
        In Singapore, all firms are required to register with the Accounting and Corporate Regulatory Authority (ACRA) during firm creation. During this registration, firms self-declare their Singapore Standard Industrial Classification (SSIC) code, which is the national standard for classifying economic activities, based on their planned business activities to be undertaken. 

        However, 2 scenarios are common:
        - Firms may not select the most appropriate SSIC code at the point of registration.
        - Firms may subsequently change their business activities and may not inform ACRA about this change.
        As a result, many firms' SSIC codes do not correctly reflect their business activities.

        This is a problem because the government relies on accurate firm SSIC codes for various monitoring and development purposes.
        
        Previously, officers manually read through each firm's business activity descriptions that are periodically collected over time through surveys, to determine if each firm's SSIC code is still reflective of its current business activities. If not, officers manually re-assign a new SSIC code to the firm.

        However, this requires:
        - A significant amount of man hours for reading thousands of texts.
        - Officers to have a good understanding of all SSIC codes, in order to correctly re-assign correct SSIC codes to firms. This is difficult as there are thousands of different SSIC codes in existence.

        To resolve the above problems, this project builds a text classifier to automatically classify firms into correct SSIC codes based on free-text descriptions of their business activities.
    """)


#sidebar
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write("App created by [Sheila Teo](https://sg.linkedin.com/in/sheila-teo) using [Streamlit](https://streamlit.io/) 🎈 and HuggingFace 🤗's [bert-base-uncased](https://huggingface.co/bert-base-uncased) model, ensembled under a [model soup](https://arxiv.org/abs/2203.05482) methodology.")
st.sidebar.markdown("---")
st.sidebar.write("Project codes open sourced [here](https://github.com/sheilateozy/ssic-classifier).")