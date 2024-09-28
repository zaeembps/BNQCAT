import streamlit as st
import pandas as pd
import openai
from rapidfuzz import fuzz, process
import re

# Set your OpenAI API key here (or use environment variables)
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Function to query GPT-3.5 using the Chat API
def query_gpt(product_description):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Switched to GPT-3.5
        messages=[
            {"role": "system", "content": "You are a product matching assistant."},
            {"role": "user", "content": f"Match the following product description to the most suitable product type: {product_description}"}
        ]
    )
    # Extract GPT's response from the messages
    return response['choices'][0]['message']['content'].strip()

# Clean and preprocess the description
def clean_description(description):
    description = description.lower()
    description = re.sub(r'[^\w\s]', '', description)  # Remove special characters
    return description

# Function to match product description to product type and category codes
def get_best_token_match(product_description, df, column_to_match, code_column, threshold=80):
    product_description = clean_description(product_description)
    # Use token set ratio for improved matching
    best_match = process.extractOne(product_description, df[column_to_match], scorer=fuzz.token_set_ratio)
    if best_match and best_match[1] >= threshold:
        best_value = best_match[0]
        code = df[df[column_to_match] == best_value][code_column].values[0]
        return best_value, code
    return None, None

# Title of the app
st.title("AI-assisted Product Code Finder with GPT-3.5")

# Load the CSV data
@st.cache_data
def load_data():
    product_type_df = pd.read_csv("B&Q Product Type Codes.csv")
    category_df = pd.read_csv("B&Q Category Codes.csv")
    return product_type_df, category_df

product_type_df, category_df = load_data()

# Input from the user
product_description = st.text_input("Enter the Product Description")

# If user has provided input
if product_description:
    st.write(f"Searching for: **{product_description}**")

    # Try GPT-3.5 to get the best product type match
    gpt_match = query_gpt(product_description)
    st.write(f"GPT-3.5's Suggested Match: **{gpt_match}**")

    # Product Type Matching using fuzzy matching as a fallback
    product_type_match, product_type_code = get_best_token_match(
        product_description,
        product_type_df,
        "Allowed Value to Be Mapped",
        "Allowed Value as Sent to Channel"
    )
    
    # Category Matching
    category_match, category_code = get_best_token_match(
        product_description,
        category_df,
        "Allowed Value to Be Mapped",
        "Allowed Value as Sent to Channel"
    )

    # Display Results
    if product_type_match and category_match:
        st.write(f"**Product Type Match (Fuzzy)**: {product_type_match} | **Code**: {product_type_code}")
        st.write(f"**Category Match (Fuzzy)**: {category_match} | **Code**: {category_code}")
    else:
        st.write("No match found. Please refine your product description.")

# Footer
st.write("---")
st.write("AI-powered product and category code finder with GPT-3.5.")
