import inspect
import json

# A global registry of functions
FUNCTION_REGISTRY = {}

def funx(func):
    """
    Decorator that registers a function for ML-based function prediction.
    Stores metadata like name, signature, and docstring.
    """
    sig = str(inspect.signature(func))
    doc = inspect.getdoc(func)
    
    FUNCTION_REGISTRY[func.__name__] = {
        "name": func.__name__,
        "signature": sig,
        "doc": doc,
        # don’t store the function itself in JSON
        "function": func  # keep in registry (runtime use only, not JSON)
    }

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper

# Example function with @funx
@funx
def open_notepad_and_write(text: str) -> str:
    """Opens Notepad (or default text editor) and writes the given text into it."""
    import os
    import subprocess
    import platform
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode='w', encoding='utf-8') as tmp:
        tmp.write(text)
        tmp_path = tmp.name

    system = platform.system()
    try:
        if system == "Windows":
            os.startfile(tmp_path)
        elif system == "Darwin":  # macOS
            subprocess.Popen(['open', '-a', 'TextEdit', tmp_path])
        elif system == "Linux":
            subprocess.Popen(['gedit', tmp_path])
        else:
            raise OSError("Unsupported operating system")
    except Exception as e:
        return f"Failed to open editor: {e}"

    return "Notepad opened with text."


exportable_registry = {
    name: {k: v for k, v in data.items() if k != "function"}
    for name, data in FUNCTION_REGISTRY.items()
}

# print(json.dumps(exportable_registry, indent=2))


def call_function(name: str, *args, **kwargs):
    """
    Calls a registered function by name with given arguments.
    """
    if name not in FUNCTION_REGISTRY:
        raise ValueError(f"Function '{name}' not found in registry.")
    func = FUNCTION_REGISTRY[name]["function"]
    return func(*args, **kwargs)


# # ML model predicts function name = "open_notepad_and_write"
# predicted_function = "open_notepad_and_write"

# # Call it with args
# result = call_function(predicted_function, "Hello world from ML model!")
# print(result)
