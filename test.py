import requests
import json
import re

def call_llm(prompt: str) -> str:
    """
    Call local LLM (Ollama) to summarize text into JSON.
    """

    model = "gemma3:4b"
    url = "http://localhost:11434/api/chat"
    fields = [str]

    # Auto-generate schema from list of types
    schema = {f"field{i+1}": t.__name__ if hasattr(t, "__name__") else str(t)
              for i, t in enumerate(fields)}

    # Prepare messages for LLM
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant. Always respond with ONLY valid JSON. "
                           "No markdown, no code fences, no explanations."
            },
            {
                "role": "user",
                "content": f"""Schema:
{json.dumps(schema, indent=2)}

Task: Summarize the following text and put the summary into field1.

Input:
{prompt}

Output JSON:"""
            }
        ],
        "stream": False
    }

    print("Payload sent to LLM:", json.dumps(payload, indent=2))

    response = requests.post(url, json=payload)
    data = response.json()

    raw = data["message"]["content"].strip()

    # Clean code fences if present
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    # Parse JSON
    json_data = json.loads(raw)

    if not isinstance(json_data, dict) or not all(key in json_data for key in schema):
        raise ValueError("Generated JSON does not match the expected schema")

    # Extract values
    values = [str(json_data[f"field{i+1}"]) for i in range(len(fields))]
    return " ".join(values)


if __name__ == "__main__":
    text = """© 2025 -Privacy-Terms
    Advertising Business Solutions About Google Google.co.in
    Google Search I'm Feeling Lucky"""

    summary = call_llm(f"Summarize this text:\n{text}")
    print("\n=== Final Summary ===")
    print(summary)
