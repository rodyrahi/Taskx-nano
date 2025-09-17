import spacy

def split_into_actions(doc):
    actions = []
    current_action = []
    for sent in doc.sents:
        # Identify verbs that could start a new action
        action_verbs = [token for token in sent if token.pos_ == "VERB" and token.dep_ in ("ROOT", "conj", "advcl")]
        
        for i, token in enumerate(sent):
            current_action.append(token)
            
            # Check if the current token is a verb that starts a new action
            if token in action_verbs and i < len(sent) - 1:
                next_token = sent[i + 1]
                # Split if the next token is a conjunction, adverb, or part of a new verb phrase
                if next_token.pos_ in ("CCONJ", "SCONJ", "ADV") or next_token.dep_ in ("conj", "advcl"):
                    # Clean up current action
                    while current_action and current_action[-1].pos_ in ("CCONJ", "SCONJ", "ADV", "PUNCT"):
                        current_action.pop()
                    if current_action:
                        actions.append(" ".join(t.text for t in current_action))
                    current_action = []
            
            # Also split if the current token is a conjunction followed by a verb
            elif token.pos_ in ("CCONJ", "SCONJ", "ADV") and i < len(sent) - 1:
                next_token = sent[i + 1]
                if next_token.pos_ == "VERB":
                    # Clean up current action
                    while current_action and current_action[-1].pos_ in ("CCONJ", "SCONJ", "ADV", "PUNCT"):
                        current_action.pop()
                    if current_action:
                        actions.append(" ".join(t.text for t in current_action))
                    current_action = []
        
        # Handle the last action in the sentence
        if current_action:
            while current_action and current_action[0].pos_ in ("CCONJ", "SCONJ", "ADV", "PUNCT"):
                current_action.pop(0)
            while current_action and current_action[-1].pos_ in ("CCONJ", "SCONJ", "ADV", "PUNCT"):
                current_action.pop()
            if current_action:
                actions.append(" ".join(t.text for t in current_action))
            current_action = []
    
    return actions

# nlp = spacy.load("en_core_web_lg")
# doc = nlp("write a poem and then put it in notepad")
# print(split_into_actions(doc))