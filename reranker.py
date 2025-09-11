from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import uuid
import os



fixing_data = {
        "open_notepad": {
        "positive": ["write note in notepad", "open text editor to write", "edit text in notepad"],
        "negative": ["search on google", "open website", "take screenshot"]
        },
        
        "open_browser": {
        "positive": ["open chrome", "launch browser", "go to website"],
        "negative": ["search videos", "write text", "check in employee" , "open google and search for cat videos"]
        },


        "search_browser_google": {
        "positive": ["search dog videos on google", "google cat pictures", "find articles on google" , "open google and search for cat videos"],
        "negative": ["open notepad", "launch telegram", "take screenshot"]
        },




        "take_screenshot": {
        "positive": ["capture screen", "take a screenshot", "screenshot desktop"],
        "negative": ["search google", "open browser", "write text"]
        },
        "open_telegram": {
        "positive": ["open telegram in firefox", "launch telegram web", "start telegram"],
        "negative": ["search on google", "write note", "check out employee"]
        },
        "run_program_at_startup": {
        "positive": ["set program to start on boot", "run app at startup", "launch program on windows start"],
        "negative": ["search videos", "open telegram", "take screenshot"]
        },
        "open_application_by_name": {
        "positive": ["launch notepad", "open chrome by name", "start application"],
        "negative": ["search google", "check in", "take screenshot"]
        },
        "checkin": {
        "positive": ["check me in employee monitor", "log in to emp monitor", "employee check in"],
        "negative": ["check me out employee monitor", "log out of emp monitor", "employee check out"]
        },
        "checkout": {
        "positive": ["check me out employee monitor", "log out of emp monitor", "employee check out"],
        "negative": ["check me in employee monitor", "log in to emp monitor", "employee check in"]
        }
        }

def rerank_functions(data, query, fixing_data=fixing_data, positive_weight=0.5, negative_weight=0.3, model_name='all-MiniLM-L6-v2'):
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    # # Input JSON-like data
    # data = [('open_notepad', 'Write text to a text editor', 0.23478991985321046), 
    #         ('open_browser', 'Open a website in the default browser', 0.8678413510322571), 
    #         ('search_browser_google', 'Search a query on Google in the default browser', 0.31010807752609254),
    #         ('take_screenshot', 'Take a screenshot of the current screen', 0.7039841103553772), 
    #         ('open_telegram', 'Open Telegram web in Firefox', 0.659984736442566), 
    #         ('run_program_at_startup', 'Run a program at Windows startup takes program path as input', 0.1714404046535492), 
    #         ('open_application_by_name', 'Launch an application or program by its name on the computer', 0.26438517570495607),
    #         ('checkin', 'this function checkes me in the emp monitor ex. check me in ', 0.8478660869598389),
    #         ('checkout', 'this function checkes me out the emp monitor ex. check me out', 0.9088648843765259)]

    # # Query
    # query = "open notepad to write some text"

    # # Synthetic fixing data (positive and negative examples for each function)
    # fixing_data = {
    #     "open_notepad": {
    #         "positive": ["write note in notepad", "open text editor to write", "edit text in notepad"],
    #         "negative": ["search on google", "open website", "take screenshot"]
    #     },
    #     "open_browser": {
    #         "positive": ["open chrome", "launch browser", "go to website"],
    #         "negative": ["search videos", "write text", "check in employee"]
    #     },
    #     "search_browser_google": {
    #         "positive": ["search dog videos on google", "google cat pictures", "find articles on google"],
    #         "negative": ["open notepad", "launch telegram", "take screenshot"]
    #     },
    #     "take_screenshot": {
    #         "positive": ["capture screen", "take a screenshot", "screenshot desktop"],
    #         "negative": ["search google", "open browser", "write text"]
    #     },
    #     "open_telegram": {
    #         "positive": ["open telegram in firefox", "launch telegram web", "start telegram"],
    #         "negative": ["search on google", "write note", "check out employee"]
    #     },
    #     "run_program_at_startup": {
    #         "positive": ["set program to start on boot", "run app at startup", "launch program on windows start"],
    #         "negative": ["search videos", "open telegram", "take screenshot"]
    #     },
    #     "open_application_by_name": {
    #         "positive": ["launch notepad", "open chrome by name", "start application"],
    #         "negative": ["search google", "check in", "take screenshot"]
    #     },
    #     # "checkin": {
    #     #     "positive": ["check me in employee monitor", "log in to emp monitor", "employee check in"],
    #     #     "negative": ["search videos", "open browser", "write text"]
    #     # },

    #     "checkin": {
    #         "positive": ["check me in employee monitor", "log in to emp monitor", "employee check in"],
    #         "negative": ["check me out employee monitor", "log out of emp monitor", "employee check out"]
    #     },

    #     "checkout": {
    #         "positive": ["check me out employee monitor", "log out of emp monitor", "employee check out"],
    #         "negative": ["check me in employee monitor", "log in to emp monitor", "employee check in"]
    #     }
    # }

    # Convert data to list of dictionaries
    functions = [{"name": name, "description": desc, "original_score": score} for name, desc, score in data]

    # Load embedding model
    model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')

    # Collect all texts for encoding
    all_texts = [query] + [f["description"] for f in functions]
    for func in fixing_data:
        all_texts.extend(fixing_data[func]["positive"])
        all_texts.extend(fixing_data[func]["negative"])

    # Compute embeddings
    embeddings = model.encode(all_texts, normalize_embeddings=True)

    # Split embeddings
    query_emb = embeddings[0:1]  # Query embedding
    desc_embs = embeddings[1:len(functions) + 1]  # Function description embeddings
    offset = len(functions) + 1

    # Compute adjusted scores
    adjusted_scores = []
    for i, func in enumerate(functions[:5]):
        func_name = func["name"]
        original_score = func["original_score"]
        
        # Get positive and negative example indices
        pos_examples = fixing_data[func_name]["positive"]
        neg_examples = fixing_data[func_name]["negative"]
        
        # Compute positive similarity
        pos_indices = [all_texts.index(p) for p in pos_examples]
        pos_embs = embeddings[pos_indices]
        pos_sim = np.mean(cosine_similarity(query_emb, pos_embs)[0]) if pos_indices else 0.0
        
        # Compute negative similarity
        neg_indices = [all_texts.index(n) for n in neg_examples]
        neg_embs = embeddings[neg_indices]
        neg_sim = np.mean(cosine_similarity(query_emb, neg_embs)[0]) if neg_indices else 0.0
        
        # Adjusted score: original_sim + 0.5 * positive_sim - 0.3 * negative_sim
        adjusted_score = original_score + 0.5 * pos_sim - 0.3 * neg_sim
        adjusted_scores.append({
            "name": func_name,
            "description": func["description"],
            "adjusted_score": adjusted_score
        })

    # Sort by adjusted score
    adjusted_scores.sort(key=lambda x: x["adjusted_score"], reverse=True)

    print( "adjusted_scores\n"  ,  adjusted_scores)
    
    return adjusted_scores

    # # Print reranked results
    # print("Reranked Functions:")
    # print("| Function Name              | Description                                                                 | Adjusted Score |")
    # print("|----------------------------|-----------------------------------------------------------------------------|----------------|")
    # for func in adjusted_scores:
    #     print(f"| {func['name']:<25} | {func['description']:<75} | {func['adjusted_score']:.4f} |")

if __name__ == "__main__":
    data = [('open_notepad', 'Write text to a text editor', 0.23478991985321046), ('open_browser', 'Open a website in the default browser', 0.8678413510322571), ('search_browser_google', 'Search a query on Google in the default browser', 0.31010807752609254), ('take_screenshot', 'Take a screenshot of the current screen', 0.7039841103553772), ('open_telegram', 'Open Telegram web in Firefox', 0.659984736442566), ('run_program_at_startup', 'Run a program at Windows startup takes program path as input', 0.1714404046535492), ('open_application_by_name', 'Launch an application or program by its name on the computer', 0.26438517570495607), ('checkin', 'this function checkes me in the emp monitor ex. check me in ', 0.8478660869598389), ('checkout', 'this function checkes me out the emp monitor ex. check me out', 0.9088648843765259)]
    query = "check me in employee monitor"
   
    try:
        reranked= rerank_functions(data, query, fixing_data)
        print("Reranked Functions:")
        # for func in reranked:
        #     print(f"{func['name']}: {func['adjusted_score']:.4f}")
    except Exception as e:
        print(f"Error: {str(e)}")