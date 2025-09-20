import requests
import json
import re

from functions import function_registry


call_llm = function_registry['call_llm']
open_note = function_registry['open_notepad']


print(open_note["output_type"])



plan = {
  "execution_plan": [
    {
      "instruction": "write a poem",
      "input": None,
      "output": "output1",
      "func_name" : "call_llm"

    },
    {
      "instruction": "put it in notepad",
      "input": "output1",
      "output": "output2: Processed output1",
      "func_name" : "open_notepad"
    }
  ]
}







import requests
import json
import re
from typing import Any, List


def generate_json(prompt: str, fields: List[Any], model: str = "gemma3:4b", url: str = "http://localhost:11434/api/chat"):
    """
    Generate JSON from a local LLM given a schema and a prompt.

    Args:
        prompt (str): User's input prompt (e.g., "write a poem").
        fields (List[Any]): List of Python types (e.g., [str, str, int]).
        model (str): LLM model name (default: "gemma3:4b").
        url (str): Ollama/LLM API URL.

    Returns:
        dict: Parsed JSON object from the model.
    """

    # Auto-generate schema from list of types
    schema = {f"field{i+1}": t.__name__ if hasattr(t, "__name__") else str(t)
              for i, t in enumerate(fields)}

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a JSON generator. Always respond with ONLY valid JSON. "
                           "No markdown, no code fences, no explanations."
            },
            {
                "role": "user",
                "content": f"""Schema:
{json.dumps(schema, indent=2)}

Input:
{prompt}

Output JSON:"""
            }
        ],
        "stream": False
    }

    response = requests.post(url, json=payload)
    data = response.json()

    raw = data["message"]["content"].strip()

    # # Clean code fences if present
    # if raw.startswith("```"):
    #     raw = re.sub(r"^```(?:json)?\s*", "", raw)
    #     raw = re.sub(r"\s*```$", "", raw)

    # try:
    #     return json.loads(raw)
    # except json.JSONDecodeError:
    #     raise ValueError(f"Model did not return valid JSON: {raw}")
    return raw

# Example usage
if __name__ == "__main__":
    
    def funnc(data):
      for value in data.values():
          yield value


    data = {"text": "hey there", "number": 10}

    print(list(funnc(data)))
    # Define schema via a list of types


    # field_types = [int , int]
    # field_types = [open_note["output_type"]]

    # result = generate_json(prompt="write a poem", fields=field_types)
    # print(result)
