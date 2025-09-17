import subprocess
import tempfile
import os
from time import sleep
import webbrowser
import urllib.parse
import platform
from typing import Type, Union, List, Dict, Any  # For type annotations

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
    """
    Uses an LLM to generate a response to the given prompt.
    """
    # Placeholder response for demonstration
    return "Roses are red, violets are blue, I'm an LLM, answering for you!"

# Example: Print the function registry to verify
if __name__ == "__main__":
    for func_name, metadata in function_registry.items():
        print(f"Function: {func_name}")
        print(f"  Description: {metadata['description']}")
        print(f"  Parameters: {metadata['required_params']}")
        print(f"  Parameter Types: {metadata['param_types']}")
        print(f"  Output Type: {metadata['output_type']}")
        print()