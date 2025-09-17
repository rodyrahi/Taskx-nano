from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

fixing_data = {
    "Write text to a text editor": {
        "positive": ["write note in notepad", "open text editor to write", "edit text in notepad"],
        "negative": ["search on google", "open website", "take screenshot", "open steam" , "write a poem"]
    },
    "Open a website in the default browser": {
        "positive": ["open chrome", "launch browser", "go to website"],
        "negative": ["search videos", "write text", "check in employee", "open google and search for cat videos", "open steam"]
    },
    "Search a query on Google in the default browser": {
        "positive": ["search dog videos on google", "google cat pictures", "find articles on google", "open google and search for cat videos"],
        "negative": ["open notepad", "launch telegram", "take screenshot", "open steam"]
    },
    "Take a screenshot of the current screen": {
        "positive": ["capture screen", "take a screenshot", "screenshot desktop"],
        "negative": ["search google", "open browser", "write text", "open steam"]
    },
    "Open Telegram web in Firefox": {
        "positive": ["open telegram in firefox", "launch telegram web", "start telegram"],
        "negative": ["search on google", "write note", "check out employee", "open steam"]
    },
    "Run a program at Windows startup takes program path as input": {
        "positive": ["set program to start on boot", "run app at startup", "launch program on windows start"],
        "negative": ["search videos", "open telegram", "take screenshot" , "write a poem"]
    },
    "Launch an application or program by its name on the computer": {
        "positive": ["launch notepad", "open chrome by name", "start application", "open steam"],
        "negative": ["search google", "check in", "take screenshot" , "write a poem"]
    },
    "This function checks me in the emp monitor ex. check me in": {
        "positive": ["check me in employee monitor", "log in to emp monitor", "employee check in"],
        "negative": ["check me out employee monitor", "log out of emp monitor", "employee check out", "open steam"]
    },
    "This function checks me out the emp monitor ex. check me out": {
        "positive": ["check me out employee monitor", "log out of emp monitor", "employee check out"],
        "negative": ["check me in employee monitor", "log in to emp monitor", "employee check in", "open steam"]
    },

    "this function will use llm to answer questions": {
        "positive": ["write a poem", "draft an email", "create a story", "compose a message"],
        "negative": ["open notepad", "search google", "take screenshot", "open steam"]  
        }
}

def rerank_functions(data, query, fixing_data=fixing_data, desc_weight=0.3, positive_weight=0.7, negative_weight=0.2, model_name='all-MiniLM-L6-v2'):
    # Convert data to list of dictionaries
    functions = [{"name": name, "description": desc, "original_score": score} for name, desc, score in data]

    # Load embedding model
    model = SentenceTransformer(model_name)

    # Collect all texts for encoding
    all_texts = [query] + [f["description"] for f in functions]
    for desc in fixing_data:
        all_texts.extend(fixing_data[desc]["positive"])
        all_texts.extend(fixing_data[desc]["negative"])

    # Remove duplicates while preserving order
    all_texts = list(dict.fromkeys(all_texts))

    # Compute embeddings
    embeddings = model.encode(all_texts, normalize_embeddings=True)

    # Split embeddings
    query_emb = embeddings[0:1]  # Query embedding
    desc_embs = embeddings[1:len(functions) + 1]  # Function description embeddings

    # Compute adjusted scores
    adjusted_scores = []
    for i, func in enumerate(functions):
        func_desc = func["description"]
        original_score = func["original_score"]
        
        # Compute description similarity
        desc_sim = cosine_similarity(query_emb, desc_embs[i:i+1])[0][0]
        
        # Get positive and negative examples for the function's description
        pos_examples = fixing_data.get(func_desc, {}).get("positive", [])
        neg_examples = fixing_data.get(func_desc, {}).get("negative", [])
        
        # Compute positive similarity
        pos_indices = [all_texts.index(p) for p in pos_examples if p in all_texts]
        pos_embs = embeddings[pos_indices] if pos_indices else np.array([])
        pos_sim = np.mean(cosine_similarity(query_emb, pos_embs)[0]) if pos_indices else 0.0
        
        # Compute negative similarity
        neg_indices = [all_texts.index(n) for n in neg_examples if n in all_texts]
        neg_embs = embeddings[neg_indices] if neg_indices else np.array([])
        neg_sim = np.mean(cosine_similarity(query_emb, neg_embs)[0]) if neg_indices else 0.0
        
        # Adjusted score: original_score + desc_weight * desc_sim + positive_weight * pos_sim - negative_weight * neg_sim
        adjusted_score = original_score + desc_weight * desc_sim + positive_weight * pos_sim - negative_weight * neg_sim
        
        # Debug output for similarity values
        # print(f"Function: {func['name']}, Desc Sim: {desc_sim:.4f}, Pos Sim: {pos_sim:.4f}, Neg Sim: {neg_sim:.4f}, Original Score: {original_score:.4f}, Adjusted Score: {adjusted_score:.4f}")
        
        adjusted_scores.append({
            "name": func["name"],
            "description": func_desc,
            "adjusted_score": adjusted_score
        })

    # Sort by adjusted score
    adjusted_scores.sort(key=lambda x: x["adjusted_score"], reverse=True)

    print( "\n")
    for item in adjusted_scores:
       
        print(f"Func: {item['name']}, Adjusted Score: {item['adjusted_score']:.2f}")
    print( "\n")
    return adjusted_scores

if __name__ == "__main__":
    data = [
        ('open_notepad', 'Write text to a text editor', 0.028776612281799313),
        ('open_browser', 'Open a website in the default browser', 0.19340458154678344),
        ('search_browser_google', 'Search a query on Google in the default browser', 0.08325648322701454),
        ('take_screenshot', 'Take a screenshot of the current screen', 0.26811617590487),
        ('open_telegram', 'Open Telegram web in Firefox', 0.36174831330776214),
        ('run_program_at_startup', 'Run a program at Windows startup takes program path as input', 0.13820195212960243),
        ('open_application_by_name', 'Launch an application or program by its name on the computer', 0.24628954768180847),
        ('checkin', 'this function checkes me in the emp monitor ex. check me in ', 0.29309606343507766),
        ('checkout', 'this function checkes me out the emp monitor ex. check me out', 0.2679979947954416)
    ]
    query = "open steam"
    
    try:
        reranked = rerank_functions(data, query)
        print("\nReranked Functions:")
        for func in reranked:
            print(f"{func['name']}: {func['adjusted_score']:.4f}")
    except Exception as e:
        print(f"Error: {str(e)}")