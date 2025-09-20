import json
import re
import subprocess
import tempfile
import os
from time import sleep
import webbrowser
import urllib.parse
import platform
from typing import Type, Union, List, Dict, Any

import requests  # For type annotations



from bs4 import BeautifulSoup
import requests
from lxml import html



# Global registry for functions and their metadata
function_registry = {}

# Decorator to register functions with description, param_types, and output_type
def register_function(description: str, param_types: List[str] = None, output_type: Type = None):
    """
    Decorator to register a function with its description, parameter types, and output type.
    
    Args:
        description (str): Description of the function.
        param_types (List[str], optional): List of parameter types (e.g., ["TEXT", "URL"]).
        output_type (Type, optional): Expected return type of the function (e.g., str, list).
    """
    if param_types is None:
        param_types = []
    
    def decorator(func):
        function_registry[func.__name__] = {
            "function": func,
            "description": description,
            "required_params": getattr(func, "__code__").co_varnames[:func.__code__.co_argcount],
            "param_types": param_types,
            "output_type": output_type.__name__ if output_type else "Any"  # Store type name as string
        }
        return func
    return decorator

# Example functions (unchanged except for the updated decorator usage)
@register_function("Write text to a text editor", param_types=["TEXT"], output_type=str)
def open_notepad(text: str) -> str:
    """
    Write text to a temporary file and open it in the default text editor.
    """
    if not text:
        return "Error: No text provided to write to Notepad"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as tmp:
        tmp.write(str(text))
        tmp_path = tmp.name

    try:
        if platform.system() == "Windows":
            subprocess.Popen(['notepad.exe', tmp_path])
        else:
            subprocess.Popen(['xdg-open' if platform.system() == "Linux" else 'open', tmp_path])
        return f"Text written to editor: {tmp_path}"
    except FileNotFoundError:
        return "Error: Text editor not found"

@register_function("Open a website in the default browser", param_types=["URL"], output_type=str)
def open_browser(website: str) -> str:
    """
    Open a website in the default browser.
    """
    if not website:
        return "Error: No website URL provided"
    
    if not website.startswith(("http://", "https://")):
        website = "https://" + website
    
    try:
        webbrowser.open(website)
        return f"URL {website} opened in browser"
    except Exception as e:
        return f"Error opening browser: {str(e)}"

@register_function("Search a query on Google in the default browser", param_types=["TEXT"], output_type=str)
def search_browser_google(query: str) -> str:
    """
    Search a query on Google in the default browser.
    """
    if not query:
        return "Error: No search query provided"
    
    safe_query = urllib.parse.quote(str(query))
    search_url = f"https://www.google.com/search?q={safe_query}"
    
    try:
        webbrowser.open(search_url)
        return f"Search for '{query}' opened in browser"
    except Exception as e:
        return f"Error opening search: {str(e)}"

@register_function("Take a screenshot of the current screen", param_types=[], output_type=str)
def take_screenshot() -> str:
    sleep(3)
    """
    Take a screenshot and open it in the default image viewer.
    """
    try:
        from PIL import ImageGrab

    except ImportError:
        return "Error: PIL module not found. Please install Pillow to use this function."

    try:
        screenshot = ImageGrab.grab()
        screenshot_path = os.path.join(tempfile.gettempdir(), f"screenshot_{os.urandom(4).hex()}.png")
        screenshot.save(screenshot_path)
        
        if platform.system() == "Windows":
            os.startfile(screenshot_path)
        else:
            subprocess.Popen(['xdg-open' if platform.system() == "Linux" else 'open', screenshot_path])
        
        return f"Screenshot taken and saved to {screenshot_path}"
    except Exception as e:
        return f"Error taking screenshot: {str(e)}"

@register_function("Open Telegram web in Firefox", param_types=[], output_type=str)
def open_telegram() -> str:
    """
    Open Telegram web in Firefox browser.
    """
    try:
        webbrowser.open('https://web.telegram.org/a/#2083633131')
        return "Telegram web opened in Firefox"
    except webbrowser.Error:
        return "Error: Firefox browser not found"

@register_function("Run a program at Windows startup takes program path as input", param_types=["FILE_PATH"], output_type=str)
def run_program_at_startup(program_path: str) -> str:
    """
    Add a program to the system startup.
    """
    program_path = str(program_path).replace(" ", '')
    if not os.path.isfile(program_path):
        return "Error: Program path is invalid"
    
    try:
        if platform.system() == "Windows":
            startup_folder = os.path.join(os.getenv('APPDATA'), 'Microsoft', 'Windows', 'Start Menu', 'Programs', 'Startup')
            shortcut_path = os.path.join(startup_folder, os.path.basename(program_path) + '.lnk')
            with open(shortcut_path, 'w') as shortcut:
                shortcut.write(f'[InternetShortcut]\nURL=file:///{program_path}\n')
            return f"Program {program_path} added to startup"
        else:
            return "Error: This function is only implemented for Windows"
    except Exception as e:
        return f"Error adding program to startup: {str(e)}"

@register_function("Launch an application or program by its name on the computer", param_types=["PROGRAM"], output_type=str)
def open_application_by_name(application_name: str) -> str:
    """
    Open an application by its name using the system's default method.
    """
    application_name = str(application_name) + ".exe"
    if not application_name:
        return "Error: No application name provided"
    
    try:
        if platform.system() == "Windows":
            os.startfile(application_name)
        elif platform.system() == "Darwin":
            subprocess.Popen(['open', '-a', application_name])
        else:
            subprocess.Popen([application_name])
        return f"Application '{application_name}' opened"
    except FileNotFoundError:
        return f"Error: Application '{application_name}' not found"
    except Exception as e:
        return f"Error opening application: {str(e)}"

@register_function("Checks me in to the emp monitor", param_types=None, output_type=str)
def checkin() -> str:
    # if ckeckin_to_emp():
    #     return "Checked in to emp monitor"
    return "Error in checking in to emp monitor"

@register_function("Checks me out of the emp monitor", param_types=None, output_type=str)
def checkout() -> str:
    # if ckeckout_to_emp():
    #     return "Checked out to emp monitor"
    return "Error in checking out to emp monitor"

@register_function("Uses LLM to answer questions", param_types=["PROMPT"], output_type=str)
def call_llm(prompt: str) -> str:

    print("Prompt received in call_llm:", prompt)
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

    model = "gemma3:4b"
    url = "http://localhost:11434/api/chat"
    fields = [str]

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

    print(raw)

    # Clean code fences if present
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    json_data = json.loads(raw)



    if not isinstance(json_data, dict) or not all(key in json_data for key in schema):
        raise ValueError("Generated JSON does not match the expected schema")

    # Extract values and join as a space-separated string
    values = [str(json_data[f"field{i+1}"]) for i in range(len(fields))]
    return " ".join(values)

    # try:
    #     return json.loads(raw)
    # except json.JSONDecodeError:
    #     raise ValueError(f"Model did not return valid JSON: {raw}")
    # return raw

# Example: Print the function registry to verify


@register_function("scrape html from a site and gives the output", param_types=["TEXT"], output_type=str)
def scrape_website(url: str) -> str:
    """
    Scrape HTML content from a website and extract all <h1> and <p> text using lxml.

    Args:
        url (str): The URL of the website to scrape.
    """

    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    try:
        response = requests.get(url)
        response.raise_for_status()
        tree = html.fromstring(response.content)

        h1_texts = [el.text_content().strip() for el in tree.xpath("//h1")]
        p_texts = [el.text_content().strip() for el in tree.xpath("//p")]

        result =  "\n".join(h1_texts) + "\n" + "\n".join(p_texts)
        return result
    except requests.RequestException as e:
        return f"Error fetching {url}: {str(e)}"
    except Exception as e:
        return f"Error parsing HTML: {str(e)}"




if __name__ == "__main__":
    for func_name, metadata in function_registry.items():
        print(f"Function: {func_name}")
        print(f"  Description: {metadata['description']}")
        print(f"  Parameters: {metadata['required_params']}")
        print(f"  Parameter Types: {metadata['param_types']}")
        print(f"  Output Type: {metadata['output_type']}")
        print()