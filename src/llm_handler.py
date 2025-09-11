import json
import os

import google.generativeai as genai

from src.config import GEMINI_MODEL_NAME


def get_multi_variable_extraction(context: str, variables_to_extract: list):
    """
    Sends retrieved context to the Gemini API to extract multiple variables
    and return a structured JSON object.

    Args:
        context (str): The combined text from the retrieved document chunks.
        variables_to_extract (list): A list of the variable names to find.

    Returns:
        dict: A dictionary with extracted values.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found. Please set it in the .env file.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL_NAME)

    json_template = {key: "value" for key in variables_to_extract}

    prompt = f"""
You are a highly precise data extraction system. Your task is to analyze the provided text and extract specific financial data points.

**Context:**
---
{context}
---

**Instructions:**

1.  **Primary Goal:** From the context above, identify and extract all values for the following variables: {", ".join(variables_to_extract)}.

2.  **Output Format:** Your response **MUST** be a single, valid JSON object and nothing else. Do not include explanations or any text outside the JSON.

3.  **Value Requirements:**
    *   The value for each variable in the JSON **MUST be a LIST of strings**.
    *   For numerical variables like `Share-based compensation` and `Net investment income`, extract **ONLY the monetary values** exactly as they appear in the text (e.g., '$60 million', '1.1 billion'). Do not include surrounding text.
    *   For categorical variables like `Product Line`, extract the names of the products lines.
    *   If no values are found for a variable, its value must be an empty list `[]`.

**Example of the required JSON format:**
```json
{json.dumps(json_template, indent=2)}
```
Provide only the JSON object in your response.
"""

    try:
        print("\nSending context to Gemini for multi-variable extraction...")
        response = model.generate_content(prompt)
        cleaned_response = (
            response.text.strip().replace("```json", "").replace("```", "")
        )
        return json.loads(cleaned_response)

    except json.JSONDecodeError:
        print("Error: Failed to decode JSON from the LLM response.")
        print(f"Raw response: {response.text}")
        return {key: "JSON Decode Error" for key in variables_to_extract}
    except Exception as e:
        print(f"An error occurred while calling the Gemini API: {e}")
        return {key: "API Error" for key in variables_to_extract}
