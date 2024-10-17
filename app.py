import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the model and tokenizer with error handling
try:
    model_name = "alpineai/cosql"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
except ImportError as e:
    st.error(f"Import Error: {e}. Make sure PyTorch and Transformers are installed.")
    st.stop()  # Stop the app if model fails to load
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
    st.stop()

# Function to generate SQL query
def generate_sql_query(user_input):
    try:
        with st.spinner("Generating SQL query..."):  # Add a loading spinner
            inputs = tokenizer(user_input, return_tensors="pt")
            outputs = model.generate(**inputs)
            query = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return query
    except Exception as e:
        st.error(f"Error generating SQL: {e}")
        return None

# Streamlit app interface
st.title("Chat-to-SQL Generator")

user_input = st.text_input("Ask your query:")
if user_input:
    sql_query = generate_sql_query(user_input)
    if sql_query:
        st.subheader("Generated SQL Query")
        st.code(sql_query)
