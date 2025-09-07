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

@register_function("Search a query on Google in the default browser takes search query as input")
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
    sleep(3)  # Give user a moment to prepare
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
    

@register_function("opens telegram on firefox")
def open_telegram():
    """
    Open Telegram web in Firefox browser.
    """
    try:
        webbrowser.open('https://web.telegram.org/a/#2083633131')
        return "Telegram web opened in Firefox"
    except webbrowser.Error:
        return "Error: Firefox browser not found"
    

@register_function("run a program at window startup takes program path as input")
def run_program_at_startup(program_path):
    """
    Add a program to the system startup.
    """
    program_path = str(program_path).replace(" " , '')
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
    


    
@register_function("Open an application by its name takes program name as input")
def open_application_by_name(application_name):
    """
    Open an application by its name using the system's default method.
    """
    if not application_name:
        return "Error: No application name provided"
    
    try:
        if platform.system() == "Windows":
            # Use 'start' to open the application by name on Windows
            subprocess.Popen(['start', '', application_name], shell=True)
        else:
            # On Unix-like systems, try to open the application directly
            subprocess.Popen([application_name])
        return f"Application '{application_name}' opened"
    except FileNotFoundError:
        return f"Error: Application '{application_name}' not found"
    except Exception as e:
        return f"Error opening application: {str(e)}"
