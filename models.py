from FISHClass.ModelZoo.LSTMModel import LSTMClassifier 
from FISHClass.ModelZoo.BasicModel import BasicClassifier 
from FISHClass.ModelZoo.ClassificationCNN import ClassificationCNN as __cnn
from FISHClass.ModelZoo.DoubleConvCNN import DoubleConvCNN as __doublecnn
from FISHClass.ModelZoo.ResNet50 import AdaptedResNet50 as __resnet
from FISHClass.ModelZoo.CombinedModel import CombinedModel as __combined
from FISHClass.ModelZoo.FeaturespaceClassifier import FeaturespaceClassifier as __feature
from FISHClass.ModelZoo.WeightedFeaturespaceClassifier import WeightedFeaturespaceClassifier as __weight_feature
from FISHClass.ModelZoo.CellAE import CellAE

from FISHClass.ModelZoo.FasterRCNNModel import train_fn, validation_fn
from torchvision.models.detection import fasterrcnn_resnet50_fpn 
from FISHClass.ModelZoo import _CNNModel_fns as CNN_fns
from FISHClass.ModelZoo import _FeaturespaceModel_fns as Featurespace_fns

from types import MethodType
from typing import Union
from pathlib import Path

def FasterRCNN(pretrained=True, weights="DEFAULT"):
    
    model = fasterrcnn_resnet50_fpn(pretrained=pretrained, weights=weights)

    model.train_fn = MethodType(train_fn, model)
    model.validation_fn = MethodType(validation_fn, model)

    return model


def ClassificationCNN(layers=[3,16,32,64,128], in_shape=[128, 128], drop_p=0.5, norm_type="dataset"):
    
    model = __cnn(layers=layers, in_shape=in_shape, norm_type=norm_type)
    
    model.train_fn = MethodType(CNN_fns.train_fn, model)
    model.validation_fn = MethodType(CNN_fns.validation_fn, model)
    
    return model


def DoubleConvCNN(layers=[3,16,32,64,128], in_shape=[128, 128], drop_p=0.5, norm_type="dataset"):
    
    model = __doublecnn(layers=layers, in_shape=in_shape, norm_type=norm_type)
    
    model.train_fn = MethodType(CNN_fns.train_fn, model)
    model.validation_fn = MethodType(CNN_fns.validation_fn, model)
    
    return model


def AdaptedResNet50(weights=None, drop_p=0.5):
    
    model = __resnet(weights=weights, drop_p=drop_p )
    
    model.train_fn = MethodType(CNN_fns.train_fn, model)
    model.validation_fn = MethodType(CNN_fns.validation_fn, model)
    
    return model


def FeaturespaceClassifier(cnnmodel_path, boxmodel_path, device="cuda", out_channel=32, box_featurespace_size=600, drop_p=0.5):
    
    model = __feature(cnnmodel_path, boxmodel_path, device=device, out_channel=out_channel, box_featurespace_size=box_featurespace_size, drop_p=drop_p)
    
    model.train_fn = MethodType(Featurespace_fns.train_fn, model)
    model.validation_fn = MethodType(Featurespace_fns.validation_fn, model)
    
    return model


def WeightedFeaturespaceClassifier(cnnmodel_path, boxmodel_path, classifiermodel_path, device="cuda", out_channel=32, box_featurespace_size=600, drop_p=0.5):
    
    model = __weight_feature(cnnmodel_path, boxmodel_path, classifiermodel_path, device=device, out_channel=out_channel, box_featurespace_size=box_featurespace_size, drop_p=drop_p)
    
    model.train_fn = MethodType(Featurespace_fns.train_fn, model)
    model.validation_fn = MethodType(Featurespace_fns.validation_fn, model)
    
    return model


def CombinedModel(fasterrcnn_path: Union[Path, str], classification_path: Union[Path, str]=None):
    
    model = __combined(fasterrcnn_path=fasterrcnn_path, classification_path=classification_path)
    
    return model

