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

def FasterRCNN(*args, **kwargs):
    
    model = fasterrcnn_resnet50_fpn(*args, **kwargs)

    model.train_fn = MethodType(train_fn, model)
    model.validation_fn = MethodType(validation_fn, model)

    return model


def ClassificationCNN(*args, **kwargs):
    
    model = __cnn(*args, **kwargs)
    
    model.train_fn = MethodType(CNN_fns.train_fn, model)
    model.validation_fn = MethodType(CNN_fns.validation_fn, model)
    
    return model


def DoubleConvCNN(*args, **kwargs):
    
    model = __doublecnn(*args, **kwargs)
    
    model.train_fn = MethodType(CNN_fns.train_fn, model)
    model.validation_fn = MethodType(CNN_fns.validation_fn, model)
    
    return model


def AdaptedResNet50(*args, **kwargs):
    
    model = __resnet(*args, **kwargs)
    
    model.train_fn = MethodType(CNN_fns.train_fn, model)
    model.validation_fn = MethodType(CNN_fns.validation_fn, model)
    
    return model


def FeaturespaceClassifier(*args, **kwargs):
    
    model = __feature(*args, **kwargs)
    
    model.train_fn = MethodType(Featurespace_fns.train_fn, model)
    model.validation_fn = MethodType(Featurespace_fns.validation_fn, model)
    
    return model


def WeightedFeaturespaceClassifier(*args, **kwargs):
    
    model = __weight_feature(*args, **kwargs)
    
    model.train_fn = MethodType(Featurespace_fns.train_fn, model)
    model.validation_fn = MethodType(Featurespace_fns.validation_fn, model)
    
    return model


def CombinedModel(*args, **kwargs):
    
    model = __combined(*args, **kwargs)
    
    return model

