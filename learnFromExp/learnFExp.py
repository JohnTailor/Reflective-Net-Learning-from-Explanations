import torch
from torch.utils.data import Dataset,TensorDataset
import clModel,dutils
import numpy as np

def trainOne():
   allGains=[]
   while True:
    dummy=False
    #dummy = True
    cfg={ 'ds': ('Ci10', 10),  #Dataset either  ('Ci100', 100) or ('Ci10', 10)
          'batchSize': 128, 'opt': ('S', 1 if dummy else 120, 0.1, 0.0005), #optimizer settings

          'compExpTar': [1], #Layer to explain - here we have only two layers in the middle, ie. use either 0 or 1
          'nSplit': 16, #depth of explanation, ie. number of features of explanation more is generally better
          'maxRan': 0.7, #Newly added: if choose a class randomly, choose it among the fraction X of most likely classes according to the models' prediction, ie. for value 0.3 and a dataset with 100 classes, the top 30 classes are used
                         #1.0 is best for CIFAR-10, ~0.5 is better for CIFAR-100 (higher accuracy and less variance across runs); in original paper this would be 1
          #These are internal settings
          'expRed': [1, 2],  'nin': 1,   'exps': ['C', 'R', '1'],
           'netSi': 0.25 if dummy else 1.0,'ntrain': 500 if dummy else 50000}
    print("Executing config",cfg)
    cfg["num_classes"]=cfg["ds"][1]
    #Get Data
    print("Get dataset")
    train_dataset, val_dataset,oX,norm=dutils.getFullDS(cfg)

    # Train and save non-reflective Model
    model, lcfg, loaded = clModel.getclassifier(cfg,  train_dataset, val_dataset, None,norm=norm)

    # Create explanations on model
    train_dataset, val_dataset, oX,_ = dutils.getFullDS(cfg)
    trd,ted= clModel.getExps(model, cfg, train_dataset, val_dataset,norm)
    trd,ted=clModel.selectLays(trd,cfg),clModel.selectLays(ted,cfg)

    # Train new reflecitve model
    gds = lambda dataset: torch.utils.data.DataLoader(TensorDataset(*[torch.from_numpy(x) for x in dataset]), batch_size=cfg["batchSize"])
    train_dataset, val_dataset = gds(trd), gds(ted)
    emodel, ecfg = clModel.getExpClassifier(cfg,  train_dataset, val_dataset, None, trainedNetSelf=model )

    print("Outcomes: ")
    print("Accuracies non-reflective Classifier", lcfg)
    print("Accuracies reflective Classifier", ecfg)
    allGains.append(ecfg["testAccPred"]- lcfg["testAcc"])
    print("Gain of reflective Classifier if use its predictions in [%] for this run", np.round(allGains[-1],4)*100)
    print("... for all runs",  "#runs: ", len(allGains), "Mean Gain", np.round(np.mean(np.array(allGains)), 4) * 100, " Std Gain",np.round(np.std(np.array(allGains)), 4) * 100)
    print("\n\n")


if __name__ == '__main__':
    trainOne()
