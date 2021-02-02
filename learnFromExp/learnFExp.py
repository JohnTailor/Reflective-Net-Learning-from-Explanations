import numpy as np,pickle
import torch
from torch.utils.data import Dataset,TensorDataset
import learnFromExp.clModel as clModel
import learnFromExp.dutils as dutils
import benchHelp as bhelp
import learnFromExp.learnCfg as learnCfg

def trainOne():
    cfg["num_classes"]=cfg["ds"][1]
    if cfg["onlyExpCl"]==0: bhelp.getPaths(cfg,longName=False) #learnCfg.bname,l
    train_dataset, val_dataset,oX,norm=dutils.getFullDS(cfg)
    # Train and save a simple model on cifar10
    model, lcfg, loaded = clModel.getclassifier(cfg,  train_dataset, val_dataset, learnCfg.resFolder,norm=norm)
    # Create explanations on model
    train_dataset, val_dataset, oX,_ = dutils.getFullDS(cfg)
    trd,ted= clModel.getExps(loaded, cfg, train_dataset, val_dataset,norm)
    trd,ted=clModel.selectLays(trd,cfg),clModel.selectLays(ted,cfg)
    su=np.abs(np.sum(trd[2], axis=(-1, -2)))
    su0 = np.sum(su==0)
    su1 = np.sum(su > 0)
    print("% 0 exp",np.round(su0/(su0+su1),3), "Tr Dat shape",[t.shape for t in trd] if not trd is None else None," Te",[t.shape for t in ted] if not ted is None else None)
    # Train new model
    gds = lambda dataset: torch.utils.data.DataLoader(TensorDataset(*[torch.from_numpy(x) for x in dataset]), batch_size=cfg["batchSize"])
    train_dataset, val_dataset = gds(trd), gds(ted)
    emodel, ecfg = clModel.getExpClassifier(cfg,  train_dataset, val_dataset, learnCfg.resFolder, trainedNetSelf=model ) #,val_datasetMa, learnCfg.resFolder


if __name__ == '__main__':
    trainOne()
