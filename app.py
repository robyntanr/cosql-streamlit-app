import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Function to load the model and tokenizer
@st.cache_resource  # Cache to prevent reloading on every run
def load_model():
    try:
        model_name = "mrm8488/t5-small-finetuned-wikiSQL"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None, None

# Function to generate SQL query from user input
def generate_sql_query(user_input):
    # Load the model only when needed
    with st.spinner('Loading model...'):
        tokenizer, model = load_model()
    if tokenizer and model:
        try:
            inputs = tokenizer.encode(user_input, return_tensors="pt", max_length=512, truncation=True)
            outputs = model.generate(inputs, max_length=150)
            query = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return query
        except Exception as e:
            st.error(f"Failed to generate SQL query: {e}")
            return ""
    else:
        st.error("The model failed to load. Please check the logs for more details.")
        return ""

# Streamlit app interface
st.title("Chat-to-SQL Generator")

user_input = st.text_input("Ask your query:")
if user_input:
    sql_query = generate_sql_query(user_input)
    if sql_query:
        st.subheader("Generated SQL Query")
        st.code(sql_query)
