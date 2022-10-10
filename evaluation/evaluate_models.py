import yaml
import argparse
import FISHClass
from baseline import models
from FISHClass.evaluation import predict_dilution as dil
from FISHClass.utils.device import best_gpu
from FISHClass.utils.evaluation import get_top_model
import os

import torch

def save2yaml(results, path):
    
    with open(path, 'w') as file:
        yaml.dump(results, file)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yamlfile", type=str, default="/home/simon_g/src/FISHClass/evaluation/model_evaluation.yaml")
    parser.add_argument("-d", "--dataset", type=str, default="/data_isilon_main/isilon_images/10_MetaSystems/MetaSystemsData/MYCN_SpikeIn/CELLINES")
    parser.add_argument("-o", "--out_path", required=True, type=str)
    parser.add_argument("-s", "--save_h5", action='store_true')
    args = parser.parse_args()
    
    
    with open(args.yamlfile) as f:
    
        yaml_data = yaml.load(f, Loader=yaml.FullLoader)["trained_model_paths"]


    for model_name, model_items in yaml_data.items():

        sample = None
        n = None
        
        if not isinstance(model_items, str):
            if model_items["model_type"] == "AreaModel":
                model = getattr(models, model_items["model_type"])(**model_items["AreaModel_kwargs"])
                results = dil.predict_mixture_baseline(model, args.dataset, dataset_kwargs={"norm_type": None, "transform": None}, n=n, sample=sample, save2h5=args.save_h5, save_path=os.path.join(args.out_path, f"{model_name}_results.h5"))
                save2yaml(results, os.path.join(args.out_path, f"{model_name}_results.yaml"))
        
        
            elif model_items["model_type"] == "SpotdetectionModel":

                model = getattr(models, model_items["model_type"])(**model_items["SpotdetectionModel_kwargs"])
                results = dil.predict_mixture_baseline(model, args.dataset, dataset_kwargs={"norm_type": None, "transform": None}, n=n, sample=sample, save2h5=args.save_h5, save_path=os.path.join(args.out_path, f"{model_name}_results.h5"))
                save2yaml(results, os.path.join(args.out_path, f"{model_name}_results.yaml"))
            
        else:
            try:
                model = torch.load(get_top_model(model_items))["model"]
            except:
                model = torch.load(get_top_model(model_items))

            print(type(model))
            results= dil.predict_mixture(model, args.dataset, device=best_gpu(), batch_size=16, dataset_kwargs={"norm_type": model.norm_type, "channels": model.channels, "mask": model.mask, "transform": None, "double_return": isinstance(model, (FISHClass.ModelZoo.FeaturespaceClassifier.FeaturespaceClassifier, FISHClass.ModelZoo.WeightedFeaturespaceClassifier.WeightedFeaturespaceClassifier))}, n=n, sample=sample, save2h5=args.save_h5, save_path=os.path.join(args.out_path, f"{model_name}_results.h5"))
            save2yaml(results, os.path.join(args.out_path, f"{model_name}_results.yaml"))   