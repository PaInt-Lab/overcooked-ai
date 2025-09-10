import ollama

# Tell the Ollama client to connect to the 'ollama-model' container on port 11434
client = ollama.Client(host="http://ollama-model:11434")

def query_ollama(model: str, prompt: str) -> str:
    """
    Send `prompt` to the specified Ollama model and return a plain string.
    """
    gen = client.generate(model=model, prompt=prompt)

    # Ollama’s Python client returns a GenerateResponse object, which has a .response attribute:
    # e.g. GenerateResponse(response="1. …\n2. …", …)
    if hasattr(gen, "response"):
        return gen.response

    # Fallback: if it’s already a string (or some other shape), just str() it:
    return str(gen)