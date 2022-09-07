import os
import torch
import pandas as pd
from FISHClass import models

def get_top_model(base):
    
    state_dict_paths = [os.path.join(base, x) for x in os.listdir(base) if ".pt" in x.lower()]
    state_dict_dict = []
    for state_dict_path in state_dict_paths:
        
        state_dict = torch.load(state_dict_path)
        state_dict_dict.append({"validation_loss": state_dict["validation_loss"],
                                "path": state_dict_path,
                                "accuracy": state_dict["accuracy"]})
        
    state_dict_df = pd.DataFrame(state_dict_dict)
    
    state_dict_df = state_dict_df.sort_values(by="validation_loss", ascending=False)

    best_model = state_dict_df.iloc[-1]
    
    return model_from_file(best_model["path"])


def model_from_file(path):
    
    state_dict = torch.load(path) 

    try:
        model = getattr(models, state_dict["model_type"])(**state_dict["model_kwargs"])
    except:
        model = getattr(models, state_dict["model_type"])()

    model.load_state_dict(state_dict["model_state_dict"])
    
    print(f"Loaded {state_dict['model_type']}")
    
    return model 
    
