import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, in_tensor):
        return in_tensor.view((in_tensor.size()[0], -1))

class B2lock2(nn.Module):
    def __init__(self, in_planes, planes,ker=3,stride=1,down=True):
        super(B2lock2, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=ker, stride=stride, padding=ker>1, bias=False)
        self.bnF = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d((2, 2), stride=2) if down else None

    def forward(self, out):
        out=self.conv1(out)
        out = self.bnF(out)
        out = self.relu(out)
        out = out if self.mp is None else self.mp(out)
        return out

def getwei(dx,di):
    dx = torch.unsqueeze(dx, -1)
    dx = torch.unsqueeze(dx, -1)
    wei = torch.repeat_interleave(dx, di, dim=2)
    wei = torch.repeat_interleave(wei, di, dim=3)
    return wei

def handleLay(wei,x,xb,sCl,aExp=None):
    aweis=[]
    for j in range(sCl.nin):
        caweis=[wei[:,j*sCl.spl:(j+1)*sCl.spl]]
        caweis=torch.cat(caweis, dim=1)
        aweis.append(caweis)
    wei = torch.cat(aweis, dim=1)
    wei = F.relu(sCl.redbn(sCl.redconv(wei)),inplace=True)
    wei = F.relu(sCl.redbn2(sCl.redconv2(wei)),inplace=True)
    x = torch.cat([x, wei], axis=1)
    return x

def addInit(cfg,sCl,isExp,classes,inoffs):
    sCl.isExp = isExp
    sCl.nExtra =0
    if isExp:
        sCl.oney = torch.eye(cfg["num_classes"]).cuda()
        sCl.nin=cfg["nin"]
        sCl.spl=cfg["nSplit"]
        tarLays = list(np.array([2,4])[np.array(cfg["compExpTar"])]) #+ ((np.array(cfg["usedExpTar"])-2)))
        sCl.nExtra = cfg["nin"] *  cfg["nSplit"]
        sCl.re = nn.ReLU(inplace=True)
        inoffs[np.array(tarLays)] = max(1,sCl.nExtra  // (1 if len(cfg["expRed"])==0 else np.prod(cfg["expRed"])))
        sCl.tar = tarLays + [9999]  # sentinel

def getCB( inch,outch, ks=1):
    redconv = nn.Conv2d(inch, outch, kernel_size=ks, stride=1, padding=ks > 1)  # THESE MUST BE FIRST
    redbn = nn.BatchNorm2d(outch)
    return redconv, redbn

def rsh(cx):
    s = cx.shape
    return torch.reshape(cx, [s[0], s[1] * s[2], s[3], s[4]])

def prePro(sCl,xb):
    x2,aexp=None,None
    if sCl.isExp:
        x = xb[0]
        nexp=np.sum(np.array(sCl.tar[:-1]) > 0)
        x2 = [rsh(cx) for cx in xb[1][:nexp]]
    else:
        x = xb
    return x,x2,aexp

def addExtra(sCl,cfg):
    if sCl.nExtra > 0:  # cin=addInitEnd(cfg, self, classes)            #sCl.nExtra * sCl.expRed
        nin3 = sCl.nExtra  # sCl.c1d=nn.conv1d(sCl.nExtra,nin1,sCl.nExtra//sCl.nin,stride=sCl.nExtra//sCl.nin)
        nin4 = nin3 // cfg["expRed"][0]
        sCl.redconv, sCl.redbn = getCB(nin3, nin4, ks=1)  # THESE MUST BE FIRST
        ks = 3 if cfg["expRed"][1] > 10 else 1
        sCl.redconv2, sCl.redbn2 = getCB(nin4, max(1, nin4 // cfg["expRed"][1]), ks=ks)  # THESE MUST BE FIRST

class ExpNet(nn.Module):
    def __init__(self, cfg, classes,isExp):
        super(ExpNet, self).__init__()
        tr = lambda x: max(1, int(np.round(x * cfg["netSi"])))
        self.in_channels=cfg["imCh"]
        chans = [self.in_channels, 32,  64,  128, 128, 256, 256, 512, 512,512]
        inoffs = np.zeros(len(chans), np.int32)
        addInit(cfg, self, isExp, classes, inoffs)

        i=-1
        def getConv(ker=3, down=True):
            nonlocal i
            i+=1
            return B2lock2(inoffs[i]+ (tr(chans[i]) if i>0 else chans[i]),tr(chans[i+1]), ker=ker,down=down)

        self.conv0 = getConv()
        self.conv1 = getConv()
        self.conv2a = getConv( down=False)
        self.conv2 = getConv()
        self.conv3a = getConv( down=False)
        self.conv3 = getConv()
        self.conv4a = getConv( down=False)
        self.conv4 = getConv()

        self.allays = [self.conv0,self.conv1, self.conv2a,self.conv2, self.conv3a,self.conv3, self.conv4a, self.conv4]
        i, ker = -1, 1
        self.flat = Flatten()
        self.dropout = nn.Dropout(0.5)
        self.nfea = tr(512)
        self.pred = nn.Linear(tr(512), classes)
        addExtra(self,cfg)

    def forward(self, xb):
        x,x2,aexp=prePro(self, xb)
        tpos=0
        for il,l in enumerate(self.allays):
            if self.isExp and il==self.tar[tpos]:
                wei = x2[tpos]
                x=handleLay(wei,x,xb,self,aexp)
                tpos+=1
            x = l(x)
        x=self.flat(x)
        x = self.dropout(x)
        x=self.pred(x)
        return x