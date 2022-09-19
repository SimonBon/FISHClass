import yaml
import argparse
from FISHClass.models import ClassificationCNN, CombinedModel
from FISHClass.utils.evaluation import get_top_model
from baseline import models
from FISHClass.evaluation import predict_dilution as dil
import pickle as pkl
from FISHClass.utils.device import best_gpu
import os

def save2yaml(results, path):
    
    with open(path, 'w') as file:
        yaml.dump(results, file)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yamlfile", type=str, default="/home/simon_g/src/FISHClass/evaluation/model_evaluation.yaml")
    parser.add_argument("-d", "--dataset", type=str, default="/data_isilon_main/isilon_images/10_MetaSystems/MetaSystemsData/MYCN_SpikeIn/results/h5/dilutions_cleaned.h5")
    parser.add_argument("-o", "--out_path", required=True, type=str)
    parser.add_argument("-s", "--save_h5", action='store_true')
    args = parser.parse_args()
    
    print(args)
    
    with open(args.yamlfile) as f:
    
        yaml_data = yaml.load(f, Loader=yaml.FullLoader)

    trained_model_paths = yaml_data["trained_model_paths"]
    
    models_to_test = ["AreaModel", "CombinedModel", "ClassificationCNN", "FeaturespaceClassifier", "WeightedFeaturespaceClassifier", "SpotdetectionModel"]
    models_to_test = ["WeightedFeaturespaceClassifier", "CombinedModel", "ClassificationCNN"]
    combined_classifier_to_test = ["LSTMClassifier", "BasicClassifier"]
    
    for test_model in models_to_test:

        sample = ["S2"]
        trials = 1
        n = 500

        if test_model == "AreaModel":
            model = getattr(models, test_model)(**yaml_data["AreaModel_kwargs"])
            results = dil.predict_mixture_baseline(model, args.dataset, dataset_kwargs={"norm_type": None, "transform": None}, trials=trials, n=n, sample=sample, save2h5=args.save_h5, save_path=os.path.join(args.out_path, f"{test_model}_results.h5"))
            save2yaml(results, os.path.join(args.out_path, f"{test_model}_results.yaml"))
    
        elif test_model == "SpotdetectionModel":

            model = getattr(models, test_model)(**yaml_data["SpotdetectionModel_kwargs"])
            results = dil.predict_mixture_baseline(model, args.dataset, dataset_kwargs={"norm_type": None, "transform": None}, trials=trials, n=n, sample=sample, save2h5=args.save_h5, save_path=os.path.join(args.out_path, f"{test_model}_results.h5"))
            save2yaml(results, os.path.join(args.out_path, f"{test_model}_results.yaml"))
            
        elif test_model == "CombinedModel":
            
            for classifier in combined_classifier_to_test:
            
                model = CombinedModel(fasterrcnn_path=trained_model_paths["FasterRCNNModel"], classification_path=trained_model_paths[classifier])
                results= dil.predict_mixture(model, args.dataset, device=best_gpu(), batch_size=4, dataset_kwargs={"norm_type": None, "transform": None}, trials=trials, n=n, sample=sample, save2h5=args.save_h5, save_path=os.path.join(args.out_path, f"{test_model}{classifier}_results.h5"))
                save2yaml(results, os.path.join(args.out_path, f"{test_model}{classifier}_results.yaml"))   
                        
        elif test_model == "FeaturespaceClassifier" or test_model == "WeightedFeaturespaceClassifier":
            
            model = get_top_model(trained_model_paths[test_model])
            results= dil.predict_mixture(model, args.dataset, batch_size=16, device=best_gpu(), dataset_kwargs={"norm_type": model.cnn_model.norm_type, "transform": None, "double_return": True}, trials=trials, n=n, sample=sample, save2h5=args.save_h5, save_path=os.path.join(args.out_path, f"{test_model}_results.h5"))
            save2yaml(results, os.path.join(args.out_path, f"{test_model}_results.yaml"))

        else:
            
            model = get_top_model(trained_model_paths[test_model])
            results= dil.predict_mixture(model, args.dataset, device=best_gpu(), dataset_kwargs={"norm_type": model.norm_type, "transform": None, "double_return": False}, trials=trials, n=n, sample=sample, save2h5=args.save_h5, save_path=os.path.join(args.out_path, f"{test_model}_results.h5"))
            save2yaml(results, os.path.join(args.out_path, f"{test_model}_results.yaml"))