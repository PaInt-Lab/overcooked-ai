"""OpenAI API client for LLM queries."""

import os
from openai import OpenAI

DEFAULT_MODEL = "gpt-4o"

def query_openai(prompt: str, model: str = None, temperature: float = 0.0):
    """Query the OpenAI API with the given prompt.

    Returns:
        tuple: (response_text: str, usage: dict) where usage has keys
               'prompt_tokens', 'completion_tokens', 'total_tokens'.
               On error, returns ("", {}).
    """
    if model is None:
        model = DEFAULT_MODEL

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set.")

    client = OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=256,
        )

        content = response.choices[0].message.content or ""
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        return content.strip(), usage

    except Exception as e:
        print(f"Error querying OpenAI: {e}")
        return "", {}

