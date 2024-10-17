import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Function to load the model and tokenizer
@st.cache_resource  # Cache to prevent reloading on every run
def load_model():
    try:
        model_name = "mrm8488/t5-small-finetuned-wikiSQL"
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
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
            # Add the necessary prefix
            input_text = "translate English to SQL: " + user_input
            input_ids = tokenizer.encode(input_text, return_tensors="pt")
            outputs = model.generate(input_ids)
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
