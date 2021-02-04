import numpy as np
import torch
import gradcam2 as grad
import torch.cuda.amp as tca

def getPred(model, rawx,device):
    rawx = torch.from_numpy(rawx).to(device)
    outputs = model(rawx)
    _, predicted = torch.max(outputs.data, 1)
    return outputs, predicted

def getGradcam(cfg,model,tarLays):
    taLays=[]
    for l in tarLays:
        #if l==4: taLay = model.conv4
        if l == 1: taLay =model.conv3a
        elif l == 0: taLay = model.conv2a
        #elif l == 1: taLay = model.conv1
        taLays.append(taLay)
        grad_cam = grad.GradCAM(model=model, target_layers=taLays, nsplit=cfg["nSplit"])
        return grad_cam

def getTargetIndex(taInd,correctCl,cfg,oldTar):
    if taInd[0] == "C": target_index = correctCl#.cpu().numpy()
    elif taInd[0] == "R":
        if cfg['maxRan']==1:
            target_index = np.random.choice(np.arange(cfg["num_classes"]))
            while target_index in oldTar: target_index = np.random.choice(np.arange(cfg["num_classes"]))
        else:
            target_index = -np.random.choice(np.arange(int(cfg["num_classes"]*cfg['maxRan']-1)))-1


    elif int(taInd[0]) > 0: target_index = -int(taInd[0])  # + (-1 if correctCl == cmax else 0)
    return target_index



def batchExp(data,normX,exps,cfg,grad_cam,actModel,lrpModel,netClaDec,netclact,getExp=True,tarLays=[3]):
  mask = np.zeros((1, 1, 32, 32))
  if not getExp:
    if cfg["miExp"]:  mask=-1  # ["expcfg"]
    masks = [mask] * len(exps)
  with tca.autocast():
     expx, rawexpx,aids,alogs,anlogs=[],[],[],[],[]
     clids=np.zeros(len(exps),np.int16)
     nlogs = np.zeros(len(exps), np.float16)
     logs = np.zeros(len(exps), np.float16)
     normX=normX.unsqueeze(1)
     correctCls = data[1].cpu().numpy()  # input = data[0][j].unsqueeze(0).cuda()
     for j in range(data[0].shape[0]):
        if getExp:# Method gradcam.py    # mask = grad_cam(input, target_index)                            # mask = np.expand_dims(np.expand_dims(mask, axis=0), axis=0)         # Method gradcam2.py
            masks, rawmasks, oldTar, cmax = [], [], [], -1
            for ie,taInd in enumerate(exps):
                target_index = getTargetIndex(taInd, correctCls[j], cfg, oldTar)
                mask, rawmask, clid, log, nlog = grad_cam(normX[j], target_index)
                if np.isnan(np.sum(mask)):
                    print("FATAL EXP NAN -set mask to 0",cfg)  # return -1
                    mask = np.zeros_like(mask, dtype=np.float16)
                oldTar.append(clid)#target_index)
                masks.append(mask)
                rawmasks.append(rawmask)
                clids[ie]=clid
                logs[ie] = log
                nlogs[ie] = nlog
        fin = np.concatenate(masks, axis=1)
        sta=lambda i: np.stack([r[i] for r in rawmasks], axis=1).astype(np.float16)
        finraw = [sta(i) for i in range(len(tarLays))]
        expx.append(fin)
        rawexpx.append(finraw)
        aids.append(np.copy(clids))
        alogs.append(np.copy(logs))
        anlogs.append(np.copy(nlogs))
  exInput = np.concatenate(expx, axis=0).astype(np.float16)
  aids=np.stack(aids,axis=0).astype(np.int16)
  alogs = np.stack(alogs, axis=0).astype(np.float16)
  anlogs = np.stack(anlogs, axis=0).astype(np.float16)
  #Mid Ex for one layer: batchSize,ClassForWhichExp,splits,h,w
  #Mid Ex: List with one entry per layer
  #input Ex: batchSize,ClassForWhichExp,layers,h,w
  #ex = np.moveaxis(ex, 1, -1)
  def getL(i):
      ri=np.concatenate([r[i] for r in rawexpx],axis=0)
      return ri# np.moveaxis(ri, 1, -1)
  expMid = [getL(i) for i in range(len(tarLays))]
  return exInput,expMid,aids,anlogs,alogs