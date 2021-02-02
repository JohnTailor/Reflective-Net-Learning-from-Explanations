import torch
import torch.nn.functional as F
import numpy as np

class GradCAM(object):
    def __init__(self, model,target_layers,gonly,nsplit=1,salnorm=True,relu=True ):
        self.model_arch = model
        self.gradients,self.activations = [None for _ in range(len(target_layers))],[None for _ in range(len(target_layers))]
        self.gonly, self.nsplit, self.salnorm, self.relu= gonly, nsplit, salnorm, relu

        def addHook(target_layer,i):
            def backward_hook(module, grad_input, grad_output):
                self.gradients[i] = grad_output[0]
                return None
            def forward_hook(module, input, output):
                self.activations[i] = output
                return None
            target_layer.register_forward_hook(forward_hook)
            target_layer.register_backward_hook(backward_hook)
        for i,l in enumerate(target_layers): addHook(l,i)

    def getweights(self, ind, score):
        gradients, activations = self.gradients[ind], self.activations[ind]
        b, k, u, v = gradients.size()
        alpha = gradients.view(b, k, -1).mean(2)
        weights = alpha.view(b, k, 1, 1)
        return weights

    def getMaps(self, ind, h, w, score):
        gradients, activations = self.gradients[ind], self.activations[ind]
        weights = self.getweights(ind, score)
        rfunc = F.relu if self.relu else lambda x: x
        saliency_map = (weights * activations)
        osaliency_map = saliency_map.detach()
        # get map without upsampling
        vecs = torch.split(osaliency_map, osaliency_map.shape[1] // self.nsplit, dim=1)
        vecs = [v.sum(1, keepdim=True) for v in vecs]
        csaliency_map = torch.cat(vecs, dim=1)
        def norm(cmap):
            saliency_map_min, saliency_map_max = cmap.min(), cmap.max()
            cmap = (cmap - saliency_map_min).div(saliency_map_max - saliency_map_min + 1e-8).data
            return cmap
        nonUpSalmap = rfunc(csaliency_map)
        nonUpSalmap = norm(nonUpSalmap).cpu().numpy().astype(np.float16)
        return np.zeros([1,1,32,32], dtype=np.float16), nonUpSalmap

    def forward(self, input, class_idx=None, retain_graph=False):
        """ Args:       input: input image with shape of (1, 3, H, W)
            class_idx (int): class index for calculating GradCAM.             If not specified, the class index that makes the highest model prediction score will be used.
        Return:   mask: saliency map of the same spatial dimension with input    logit: model output        """

        b, c, h, w = input.size() if not type(input) is tuple else input[0].size()
        logit = self.model_arch(input)
        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
            clid = logit.max(1)[-1].detach().cpu().numpy()[0]
        else:
            if class_idx>=0:
                score = logit[:, class_idx].squeeze()
                clid=class_idx
            else:
                soScore=logit.sort(1).indices
                score = logit[:, soScore[-1][class_idx]].squeeze()
                clid=soScore[-1][class_idx].detach().cpu().numpy()

        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        maps,rmaps=[],[]
        for i in range(len(self.gradients)):
            m,rm=self.getMaps(i,h,w,score)
            maps.append(m)
            rmaps.append(rm)
        self.model_arch.zero_grad()
        log=logit[:,clid].detach()
        return np.stack(maps,axis=2),rmaps,clid,log.cpu().numpy()[0], (log/(1e-7+torch.sum(torch.abs(logit)))).detach().cpu().numpy()[0]

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)
