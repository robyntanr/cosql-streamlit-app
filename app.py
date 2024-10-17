import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the model and tokenizer with error handling
try:
    model_name = "alpineai/cosql"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.gradient_checkpointing_enable()  # Ensure it aligns with the new API
except ImportError as e:
    st.error(f"An error occurred while loading the model: {e}. Please check your dependencies.")
except Exception as e:
    st.error(f"Unexpected error: {e}")

# Function to generate SQL query
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
