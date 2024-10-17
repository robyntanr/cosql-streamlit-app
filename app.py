import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Helper function to initialize the model and tokenizer
@st.cache_resource  # Cache to prevent reloading on every run
def load_model():
    try:
        model_name = "alpineai/cosql"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Enable gradient checkpointing if supported
        try:
            model.gradient_checkpointing_enable()
        except AttributeError:
            st.warning("This model does not support gradient checkpointing.")
        
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None, None

# Initialize the tokenizer and model
tokenizer, model = load_model()

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

if tokenizer and model:  # Ensure the model loaded successfully
    user_input = st.text_input("Ask your query:")
    if user_input:
        sql_query = generate_sql_query(user_input)
        if sql_query:
            st.subheader("Generated SQL Query")
            st.code(sql_query)
else:
    st.error("The model failed to load. Please check the logs for more details.")
