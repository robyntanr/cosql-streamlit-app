import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the model and tokenizer with error handling
try:
    model_name = "alpineai/cosql"
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Safely enable gradient checkpointing if supported
    try:
        model.gradient_checkpointing_enable()
    except AttributeError:
        st.warning("This model does not support gradient checkpointing.")

except ImportError as e:
    st.error(f"Import Error: {e}. Please ensure all dependencies are installed.")
except OSError as e:
    st.error(f"Model loading error: {e}. Check your internet connection or model availability.")
except Exception as e:
    st.error(f"Unexpected error: {e}")

# Function to generate SQL query from user input
def generate_sql_query(user_input):
    try:
        inputs = tokenizer(user_input, return_tensors="pt")
        outputs = model.generate(**inputs)
        query = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return query
    except Exception as e:
        st.error(f"Failed to generate SQL query: {e}")
        return ""

# Streamlit app interface
st.title("Chat-to-SQL Generator")

user_input = st.text_input("Ask your query:")
if user_input:
    sql_query = generate_sql_query(user_input)
    if sql_query:
        st.subheader("Generated SQL Query")
        st.code(sql_query)
