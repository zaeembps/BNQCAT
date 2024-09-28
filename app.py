import openai
import streamlit as st
import pandas as pd
from rapidfuzz import fuzz, process
import re

# Set your OpenAI API key directly from Streamlit Secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Function to query GPT-3.5 for product description matching
def query_gpt(product_description):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Match the following product description: {product_description}"}
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

# Function to clean the product description for fuzzy matching
def clean_description(description):
    description = description.lower()
    description = re.sub(r'[^\w\s]', '', description)  # Remove special characters
    return description

# Function to find the best matching product type and category codes using fuzzy matching
def get_best_token_match(product_description, df, column_to_match, code_column, threshold=80):
    cleaned_description = clean_description(product_description)
    best_match = process.extractOne(cleaned_description, df[column_to_match], scorer=fuzz.token_set_ratio)
    if best_match and best_match[1] >= threshold:
        best_value = best_match[0]
        code = df[df[column_to_match] == best_value][code_column].values[0]
        return best_value, code
    return None, None

# Load the product type and category data (cached for performance)
@st.cache_data
def load_data():
    product_type_df = pd.read_csv("B&Q Product Type Codes.csv")
    category_df = pd.read_csv("B&Q Category Codes.csv")
    return product_type_df, category_df

# App title and instructions
st.title("AI-assisted Product Code Finder")
st.write("Enter a product description to find the most suitable product type and category codes.")

# Load data
product_type_df, category_df = load_data()

# Input field for product description
product_description = st.text_input("Enter the Product Description")

# Process the input if a description is provided
if product_description:
    st.write(f"Searching for: **{product_description}**")

    # GPT-3.5 suggestion for product type match
    gpt_suggestion = query_gpt(product_description)
    st.write(f"**GPT-3.5's Suggested Match**: {gpt_suggestion}")

    # Fuzzy matching for product type and category codes
    product_type_match, product_type_code = get_best_token_match(
        product_description,
        product_type_df,
        "Allowed Value to Be Mapped",
        "Allowed Value as Sent to Channel"
    )
    
    category_match, category_code = get_best_token_match(
        product_description,
        category_df,
        "Allowed Value to Be Mapped",
        "Allowed Value as Sent to Channel"
    )

    # Display the fuzzy matching results
    if product_type_match and category_match:
        st.write(f"**Fuzzy Match for Product Type**: {product_type_match} | **Code**: {product_type_code}")
        st.write(f"**Fuzzy Match for Category**: {category_match} | **Code**: {category_code}")
    else:
        st.write("No match found. Please refine your product description.")

# Footer
st.write("---")
st.write("This tool uses GPT-3.5 and fuzzy matching to suggest the best product and category codes.")
