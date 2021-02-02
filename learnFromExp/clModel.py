import numpy as np,copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as tca
import learnFromExp.lutils as lutils
import learnFromExp.learnCfg as learnCfg
from learnFromExp.models import ExpNet

niter = 1e10

def expRun(dataset, getExp, nexp, exps,cfg,grad_cam,tarLays,norm,actModel,lrpModel,netClaDec,netclact):
    #Data consists of: X,y,all exps, estimated y of exps, logit y, normed logits
    #All Exps: List of Input Ex, Mid Ex (where Input Ex is always one entry, and Mid Ex has one list entry per layer)
    # Mid Ex for one layer: batchSize,ClassForWhichExp,splits,h,w
    # Input Ex: batchSize,ClassForWhichExp,layers,h,w
    ox, oy, expx, rawexpx, masks,aids,alogs,anlogs = [], [], [], [], [], [],[],[]
    for i, data in enumerate(dataset):
        normX = (data[0].cuda() - norm[0]) / norm[1]
        ox.append(normX.cpu().numpy().astype(np.float16))
        oy.append(data[1].numpy().astype(np.int16))
        bx, bx2,clids,logs,nlogs = lutils.batchExp(data,normX, exps, cfg, grad_cam,actModel,lrpModel,netClaDec,netclact,getExp=getExp, tarLays=tarLays)
        expx.append(bx)
        rawexpx.append(bx2)
        aids.append(clids)
        alogs.append(logs)
        anlogs.append(nlogs)
        if len(oy) * data[0].shape[0] > nexp: break
        if len(oy) % 10 == 0: print("Nex", len(oy) * data[0].shape[0], nexp)
    oy = np.concatenate(oy).astype(np.int16)
    ox = np.concatenate(ox, axis=0).astype(np.float16)
    ex = np.concatenate(expx, axis=0).astype(np.float16)
    aids=np.concatenate(aids,axis=0).astype(np.int16)
    alogs = np.concatenate(alogs, axis=0).astype(np.float16)
    anlogs = np.concatenate(anlogs, axis=0).astype(np.float16)
    sta = lambda i: np.concatenate([r[i] for r in rawexpx], axis=0)
    exr = [sta(i) for i in range(len(tarLays))]  #np.concatenate(rawexpx,axis=0).astype(np.float16)
    return ox, oy, [ex] + exr,aids,alogs,anlogs


def selectLays(ds, cfg): #Returns X,Y, Exp(SalMaps),classes,logits (if used later)
    tl = np.array(cfg['usedExpTar']) - 1-cfg["compExpOff"]
    return ds[:2] + [ds[3 + t] for t in tl] + [ds[-3]]

def getExps(loaded,cfg,train_dataset,val_dataset,norm):
    #Data consists of: X,y,all exps, estimated y of exps, logit y, normed logits
    #All Exps: List of Input Ex, Mid Ex (where Input Ex is always one entry, and Mid Ex has one list entry per layer)
    # Mid Ex for one layer: batchSize,ClassForWhichExp,splits,h,w
    # Input Ex: batchSize,ClassForWhichExp,layers,h,w
    #Exp have shape Upsampled: batchSize,#targetlays,#targetclasses,#features/splits,imgheight,imgwid
    ### Exp have shape for nonexp: batchSize,#targetclasses,#targetlays,#splits,imgheight,imgwid
    modcfg=copy.deepcopy(cfg)    #ecfg = modcfg["clcfgForExp"]  modcfg["clcfg"]=ecfg
    model, lcfg, loadedExp = getclassifier(modcfg,  train_dataset, val_dataset, learnCfg.resFolder,forceLoad=True)  # if "trainCl" in cfg: return
    grad_cam, actModel, lrpModel,netClaDec,netclact = None, None, None,None,None
    grad_cam=lutils.getGradcam(cfg,model,cfg["compExpTar"])
    d = expRun(train_dataset,True,cfg["nexp"],cfg["exps"],cfg,grad_cam,cfg["compExpTar"],norm,actModel,lrpModel,netClaDec,netclact)#"CMSTR"
    vd = expRun(val_dataset, True, cfg["nexp"] // 2, cfg["exps"],cfg,grad_cam,cfg["compExpTar"],norm,actModel,lrpModel,netClaDec,netclact) #"CMSTRRRRRR"
    return d,vd

def decay(ccf,epoch,optimizerCl):
    if ccf["opt"][0] == "S" and (epoch + 1) % (ccf["opt"][1] // 3+ccf["opt"][1]//10+2 ) == 0:
        for p in optimizerCl.param_groups: p['lr'] *= 0.1
        print("  D", np.round(optimizerCl.param_groups[0]['lr'],5))

def getSingleAcc(net, dsx, labels, pool=None):
  with tca.autocast():
    outputs = net(dsx)
    if type(outputs) is tuple: outputs=outputs[1] #for attention net use second output
    _, predicted = torch.max(outputs.data, 1)
    correct = torch.eq(predicted,labels).sum().item()
    return correct

def getEAcc(net, dataset, iexp,  niter=10000, pool=None, zeroExp=1, cfg=None):
    correct,total = 0,0
    net.eval()
    with torch.no_grad():
        for i,data in enumerate(dataset):
            labels = data[1].cuda()
            edat=[d.clone().numpy() for d in data[2:]]
            nd,_=getexp(edat, cfg, iexp, zeroExp=zeroExp, isTrain=False)
            xgpu=data[0].cuda()
            ndgpu=[torch.from_numpy(x).cuda() for x in nd]
            correct += getSingleAcc(net, (xgpu, ndgpu), labels, pool=pool)
            total += labels.size(0)
            if i>=niter: break
    return correct/total


def getAcc(net, dataset,  niter=10000,norm=None):
    correct,total = 0,0
    net.eval()
    with torch.no_grad():
        for cit,data in enumerate(dataset):
            with tca.autocast():
                dsx,dsy = data[0].cuda(),data[1].cuda()
                dsx = (dsx - norm[0])/norm[1]
                total += dsy.size(0)
                outputs = net(dsx.float())
                _, predicted = torch.max(outputs.data, 1)
                correct += torch.eq(predicted, dsy).sum().item()
                if cit>=niter: break
    return correct/total

def getexp(data,cfg,ranOpt,zeroExp,isTrain=True):
    #Mid Ex for one layer:
    #Return for Mid Ex: List of Exp: one entry for each layer (+ if aexp one entry containing all classes -> this works only for nin==1)
    #One entry per layer: batchSize,ClassForWhichExp (==1 for nin=1),splits,h,w
    inds=None
    if isTrain:
        inds = np.random.choice(ranOpt, data[0].shape[0])  # expx = np.copy(data)[:, :1] for d in data: print(d.shape,"ds",ranOpt)
        lex = [np.expand_dims(d[np.arange(d.shape[0]), inds], axis=1) for d in data]
    else:
        lex = [np.expand_dims(d[:, ranOpt[0]], axis=1) for d in data]
    return lex,inds

def getxdat(xdat,zeroExp,aug,cfg,ranOpt):
    rdat = [x.clone().numpy() for x in xdat[1:]] #Selected explanations - don't change original input at 0
    ex,inds = getexp(rdat, cfg, ranOpt, zeroExp=zeroExp)
    return [xdat[0].cuda()]+[torch.from_numpy(d).cuda() for d in ex],inds

def getOut(ndgpu,netCl,cfg):
    dropCl = np.random.random() < cfg["dropCl"]
    dropA =  np.random.random() < cfg["dropA"]
    output = netCl((ndgpu[0], ndgpu[1:],dropCl,dropA))
    return output

def getNet(cfg,ccf,isExp):
    NETWORK = ExpNet
    netCl = NETWORK(cfg, cfg["num_classes"],isExp).cuda()
    return netCl

def getExpClassifier(cfg,  train_dataset, val_dataset, resFolder, trainedNetSelf=None): #Co,val_datasetMa,resFolder
    ccf=cfg["expcfg"]
    netCl=getNet(cfg,ccf,True)
    #print(netCl)  #print(trainedNetSelf)
    aep,asp= list(netCl.parameters()), trainedNetSelf.parameters()
    for iep,sp in enumerate(asp):
        ep=aep[iep]
        #print(ep.shape,sp.shape,"sha", len(aep), len(list(trainedNetSelf.parameters())))
        if sum(list(sp.data.shape))!=sum(list(ep.data.shape)):
            if len(sp.shape)>1: ep.data[:sp.data.shape[0],:sp.data.shape[1]]=sp.data.clone()
            else: ep.data[:sp.data.shape[0]]=sp.data.clone()
        else: ep.data.copy_(sp.data)
    if ccf["opt"][0] == "S": optimizerCl = optim.SGD(netCl.parameters(), lr=ccf["opt"][2], momentum=0.9, weight_decay=ccf["opt"][3]) #elif ccf["opt"][0] == "A": optimizerCl = optim.Adam(netCl.parameters(), ccf["opt"][2], weight_decay=ccf["opt"][3])
    else: "Error opt not found"
    closs, trep, loss = 0,  ccf["opt"][1], nn.CrossEntropyLoss()
    print("Train ExpCL")
    scaler = tca.GradScaler()
    ranOpt=np.sort(np.array(cfg["netExp"])) #sorting is very important
    emateAccs,etrAccs=[],[]
    iexp = list(np.arange(len(cfg["exps"])))
    icorr=iexp[:1]+iexp[2:]
    imax=iexp[2:]
    for epoch in range(trep):
        netCl.train()
        for i, data in enumerate(train_dataset):
            with tca.autocast():
                optimizerCl.zero_grad()
                dsy = data[1].cuda()
                ndgpu,inds = getxdat([data[0]] + list(data[2:]),cfg["expB"],ccf["aug"],cfg,ranOpt)
                output=getOut(ndgpu,netCl,cfg)
                errD_real = loss(output, dsy.long())
                scaler.scale(errD_real).backward()
                scaler.step(optimizerCl)
                scaler.update()
                closs = 0.97 * closs + 0.03 * errD_real.item() if i > 20 else 0.8 * closs + 0.2 * errD_real.item()
        decay(ccf,epoch,optimizerCl)
        netCl.eval()
        emateAccs.append(getEAcc(netCl, val_dataset, imax,  niter=niter, cfg=cfg) if len(imax) else -1)
        if cfg['getTrAcc']: etrAccs.append(getEAcc(netCl, train_dataset, imax,  niter=niter, cfg=cfg) if len(imax) else -1)
        if (epoch % 2 == 0 and epoch<=10) or (epoch % 10==0 and epoch>10) :
            cacc=getEAcc(netCl, val_dataset, iexp[1:],  niter=niter, cfg=cfg)
            print(epoch, np.round(np.array([closs, cacc, getEAcc(netCl, train_dataset, icorr,  niter=niter, cfg=cfg)]), 5), cfg["pr"])
            if np.isnan(closs):
                print("Failed!!!")
                return None,None
    netCl.eval()
    lcfg = {"teAccECo": getEAcc(netCl, val_dataset, [0],  niter=niter, cfg=cfg),
                "teAccEMa": getEAcc(netCl, val_dataset, [2] ,  niter=niter, cfg=cfg) if len(iexp)>2 else -1,
                "teAccER": getEAcc(netCl, val_dataset, [1],  niter=niter, cfg=cfg),
                "teAccsEMa": emateAccs,"trAccsEMa": etrAccs}
    setEval(netCl)
    return netCl, lcfg

def setEval(netCl):
        netCl.eval()
        for name, module in netCl.named_modules():
            if isinstance(module, nn.Dropout): module.p = 0
            elif isinstance(module, nn.LSTM): module.dropout = 0 #print("zero lstm drop") #print("zero drop")
            elif isinstance(module, nn.GRU): module.dropout = 0

def getLo(model):
    reg_loss = 0
    for name,param in model.named_parameters():
        if 'bn' not in name:
             reg_loss += torch.norm(param)
    #loss = cls_loss + args.weight_decay*reg_loss
    return reg_loss


def getclassifier(cfg,train_dataset,val_dataset,resFolder,forceLoad=False,norm=None):
    ccf=cfg["clcfg"]
    netCl=getNet(cfg,ccf,False)
    optimizerCl = optim.SGD(netCl.parameters(), lr=ccf["opt"][2], momentum=0.9, weight_decay=ccf["opt"][3])
    closs,teaccs,trep,loss,clr = 0,[],ccf["opt"][1],nn.CrossEntropyLoss(), ccf["opt"][2]
    print("Train CL",ccf)
    scaler = tca.GradScaler()
    teAccs,trAccs=[],[]
    clAcc = lambda dataset: getAcc(netCl, dataset,  niter=niter,norm=norm)
    for epoch in range(trep):
        netCl.train()
        for i, data in enumerate(train_dataset):
          with tca.autocast():
            optimizerCl.zero_grad()
            dsx = data[0]
            dsx,dsy = dsx.cuda(),data[1].cuda()
            dsx=(dsx-norm[0])/norm[1]
            output = netCl(dsx.float())  # if useAtt:                #     errD_real = loss(output[0], dsy.long())+loss(output[1], dsy.long())                #     output=output[1] #prediction outputs                # else:
            errD_real = loss(output, dsy.long())
            scaler.scale(errD_real).backward()
            scaler.step(optimizerCl)
            scaler.update()
            closs = 0.97 * closs + 0.03 * errD_real.item() if i > 20 else 0.8 * closs + 0.2 * errD_real.item()
        decay(ccf,epoch,optimizerCl)
        netCl.eval()
        teAccs.append(clAcc(val_dataset))
        if cfg['getTrAcc']: trAccs.append(clAcc(train_dataset))
        if (epoch % 2 == 0 and epoch<=10) or (epoch % 10==0 and epoch>10):
            print(epoch, np.round(np.array([closs, teAccs[-1], clAcc(train_dataset),max(teAccs)]), 5), cfg["pr"])
            if np.isnan(closs):
                print("Failed!!!")
                return None,None

    lcfg = {"ClteAcc": clAcc(val_dataset), "CltrAcc": clAcc(train_dataset),"ClteAccs":teAccs,"CltrAccs":trAccs,"mClteAcc":max(teAccs)}
    setEval(netCl)
    return netCl, lcfg,False