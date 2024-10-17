import streamlit as st
import openai

# Set up OpenAI API key
openai.api_key = st.secrets["openai_api_key"]

# Function to generate SQL query from user input
def generate_sql_query(user_input):
    try:
        # Define the prompt for the model
        prompt = f"Translate the following natural language query into SQL:\n\n\"{user_input}\"\n\nSQL Query:"
        
        # Call the OpenAI API
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150,
            temperature=0,
            n=1,
            stop=None
        )
        # Extract the SQL query from the response
        sql_query = response.choices[0].text.strip()
        return sql_query
    except Exception as e:
        st.error(f"Failed to generate SQL query: {e}")
        return ""

# Streamlit app interface
st.title("Chat-to-SQL Generator")

user_input = st.text_input("Enter your query in natural language:")
if user_input:
    with st.spinner('Generating SQL query...'):
        sql_query = generate_sql_query(user_input)
    if sql_query:
        st.subheader("Generated SQL Query")
        st.code(sql_query, language='sql')
