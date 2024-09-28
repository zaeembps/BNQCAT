import openai
import streamlit as st

# Set the OpenAI API key from Streamlit Secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Function to query GPT-3.5 for product description matching
def query_gpt(product_description):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a product matching assistant."},
                {"role": "user", "content": f"Match the following product description to the most suitable product type: {product_description}"}
            ]
        )
        return response['choices'][0]['message']['content'].strip()
    except openai.error.AuthenticationError:
        return "Authentication Error: Check your API key."
    except openai.error.APIConnectionError:
        return "Connection Error: Unable to reach OpenAI's API."
    except openai.error.RateLimitError:
        return "Rate Limit Exceeded: You have hit your rate limit."
    except openai.error.InvalidRequestError as e:
        return f"Invalid Request: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

# Example use of the query_gpt function in your Streamlit app
product_description = "A description of a product"
gpt_suggestion = query_gpt(product_description)
st.write(f"GPT-3.5's Suggested Match: {gpt_suggestion}")
