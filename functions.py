import subprocess
import tempfile
import os
from time import sleep
import webbrowser
import urllib.parse
import platform

# Global registry for functions and their descriptions
function_registry = {}

# Decorator to register functions with descriptions
def register_function(description):
    def decorator(func):
        function_registry[func.__name__] = {
            "function": func,
            "description": description,
            "required_params": getattr(func, "__code__").co_varnames[:func.__code__.co_argcount]
        }
        return func
    return decorator

@register_function("Write text to a text editor")
def open_notepad(text):
    """
    Write text to a temporary file and open it in the default text editor.
    """
    if not text:
        return "Error: No text provided to write to Notepad"
    
    # Create a temporary file with the text
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as tmp:
        tmp.write(str(text))
        tmp_path = tmp.name

    # Open the file in the default text editor
    try:
        if platform.system() == "Windows":
            subprocess.Popen(['notepad.exe', tmp_path])
        else:
            # Use xdg-open on Linux or open on macOS
            subprocess.Popen(['xdg-open' if platform.system() == "Linux" else 'open', tmp_path])
        return f"Text written to editor: {tmp_path}"
    except FileNotFoundError:
        return "Error: Text editor not found"

@register_function("Open a website in the default browser")
def open_browser(website):
    """
    Open a website in the default browser.
    """
    if not website:
        return "Error: No website URL provided"
    
    # Ensure the URL has a protocol
    if not website.startswith(("http://", "https://")):
        website = "https://" + website
    
    try:
        webbrowser.open(website)
        return f"URL {website} opened in browser"
    except Exception as e:
        return f"Error opening browser: {str(e)}"

@register_function("Search a query on Google in the default browser")
def search_browser_google(query):
    """
    Search a query on Google in the default browser.
    """
    if not query:
        return "Error: No search query provided"
    
    # Sanitize the query for URL
    safe_query = urllib.parse.quote(str(query))
    search_url = f"https://www.google.com/search?q={safe_query}"
    
    try:
        webbrowser.open(search_url)
        return f"Search for '{query}' opened in browser"
    except Exception as e:
        return f"Error opening search: {str(e)}"

@register_function("Take a screenshot of the current screen")
def take_screenshot():
    sleep(1)  # Give user a moment to prepare
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
        
        # Open the screenshot with the default image viewer
        if platform.system() == "Windows":
            os.startfile(screenshot_path)
        else:
            subprocess.Popen(['xdg-open' if platform.system() == "Linux" else 'open', screenshot_path])
        
        return f"Screenshot taken and saved to {screenshot_path}"
    except Exception as e:
        return f"Error taking screenshot: {str(e)}"