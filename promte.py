import json
import random

# Sample function templates
functions = [
    {"function": "play_music(song: str, artist: str)", "params": ["song", "artist"], "context": "Plays a song by the specified artist in the music player."},
    {"function": "set_alarm(time: str)", "params": ["time"], "context": "Sets an alarm for the given time."},
    {"function": "open_notepad_with_text(text: str)", "params": ["text"], "context": "Opens notepad and writes text."},
    {"function": "send_email(to: str, subject: str, body: str)", "params": ["to", "subject", "body"], "context": "Sends an email to the recipient with subject and body."},
    {"function": "open_firefox(website: str)", "params": ["website"], "context": "Opens Firefox and navigates to specified website."},
]

# Sample words / values for parameters
songs = ["Shape of You", "Blinding Lights", "Levitating", "Bad Habits"]
artists = ["Ed Sheeran", "The Weeknd", "Dua Lipa", "Justin Bieber"]
times = ["7:00 AM", "8:30 AM", "6:45 PM", "12:00 PM"]
texts = ["hello world", "buy milk", "write a report", "meeting notes"]
people = ["John", "Alice", "Bob", "Sarah"]
subjects = ["Hello", "Project Update", "Meeting", "Greetings"]
bodies = ["Please review the report.", "See you tomorrow.", "Happy Birthday!", ""]
websites = ["google.com", "github.com", "stackoverflow.com", "youtube.com"]

data = []

for i in range(500):
    func = random.choice(functions)
    prompt = ""
    args = {}
    param_annotations = {}

    if func["function"].startswith("play_music"):
        song = random.choice(songs)
        artist = random.choice(artists)
        prompt = f"play the song '{song}' by {artist}"
        args = {"song": song, "artist": artist}
        param_annotations = {"song": [song], "artist": [artist]}

    elif func["function"].startswith("set_alarm"):
        time = random.choice(times)
        prompt = f"set an alarm for {time}"
        args = {"time": time}
        param_annotations = {"time": [time]}

    elif func["function"].startswith("open_notepad_with_text"):
        text = random.choice(texts)
        prompt = f"open notepad and write '{text}'"
        args = {"text": text}
        param_annotations = {"text": [text]}

    elif func["function"].startswith("send_email"):
        to = random.choice(people)
        subject = random.choice(subjects)
        body = random.choice(bodies)
        prompt = f"send an email to {to} with subject {subject}"
        if body:
            prompt += f" and body {body}"
        args = {"to": to, "subject": subject, "body": body}
        param_annotations = {"to": [to], "subject": [subject], "body": [body]}

    elif func["function"].startswith("open_firefox"):
        website = random.choice(websites)
        prompt = f"open firefox and go to {website}"
        args = {"website": website}
        param_annotations = {"website": [website]}

    data.append({
        "prompt": prompt,
        "context": func["context"],
        "function_signature": func["function"],
        "param_annotations": param_annotations,
        "args": args
    })

# Save to JSONL
with open("function_training_data.jsonl", "w", encoding="utf-8") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")

print("Generated 500-function training examples in 'function_training_data.jsonl'.")
