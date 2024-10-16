import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the alpineai/cosql model and tokenizer
model_name = "alpineai/cosql"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def generate_sql_query(user_input):
    inputs = tokenizer(user_input, return_tensors="pt")
    outputs = model.generate(**inputs)
    query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return query

# Streamlit app interface
st.title("Chat-to-SQL Generator")

user_input = st.text_input("Ask your query:")
if user_input:
    sql_query = generate_sql_query(user_input)
    st.subheader("Generated SQL Query")
    st.code(sql_query)
