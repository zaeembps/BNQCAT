import streamlit as st
import pandas as pd
import openai
from rapidfuzz import fuzz, process
import re

# Set your OpenAI API key here (or use environment variables)
openai.api_key = "your-openai-api-key"

# Function to query GPT for better matching
def query_gpt(product_description):
    response = openai.Completion.create(
        engine="text-davinci-003",  # You can change this to other models like "gpt-4" if you have access
        prompt=f"Match the following product description to the most suitable product type: {product_description}",
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.5
    )
    # Extract GPT's response
    return response.choices[0].text.strip()

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
st.title("AI-assisted Product Code Finder with GPT")

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

    # Try GPT to get the best product type match
    gpt_match = query_gpt(product_description)
    st.write(f"GPT's Suggested Match: **{gpt_match}**")

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
st.write("AI-powered product and category code finder with GPT.")
