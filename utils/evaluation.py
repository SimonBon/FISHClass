import os
import torch
import pandas as pd
import FISHClass.models

def get_top_model(base):
    
    state_dict_paths = [os.path.join(base, x) for x in os.listdir(base) if ".pt" in x.lower()]
    state_dict_dict = []
    for state_dict_path in state_dict_paths:
        
        state_dict = torch.load(state_dict_path, map_location="cpu")
        state_dict_dict.append({"validation_loss": state_dict["validation_loss"],
                                "path": state_dict_path})
        
    state_dict_df = pd.DataFrame(state_dict_dict)
    state_dict_df = state_dict_df[state_dict_df['validation_loss'].notna()]
    state_dict_df = state_dict_df.sort_values(by="validation_loss", ascending=False)

    best_model = state_dict_df.iloc[-1]
    return best_model["path"]
    
    #return model_from_file(best_model["path"])


def model_from_file(path):
    
    state_dict = torch.load(path, map_location="cpu") 

    if "model_kwargs" in state_dict.keys():
        model = getattr(FISHClass.models, state_dict["model_type"])(**state_dict["model_kwargs"])
    else:
        model = getattr(FISHClass.models, state_dict["model_type"])()

    model.load_state_dict(state_dict["model_state_dict"])
    
    print(f"Loaded {state_dict['model_type']} from {path}")
    
    return model 
    
