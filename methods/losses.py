import torch
import torch.nn.functional as F
import numpy as np
import math

class CrossEntropy(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss()
    
    def forward(self, pred, labels, **kwargs):
        ce = self.ce(pred, labels)
        return ce

class FocalLoss(torch.nn.Module):
    '''
        https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
    '''

    def __init__(self, gamma=0, alpha=None, size_average=True, *args, **kwargs):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, pred, labels, **kwargs):
        input = pred
        target = labels

        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)                         # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))    # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * torch.autograd.Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class MeanAbsoluteError(torch.nn.Module):
    def __init__(self, num_classes, *args, **kwargs) -> None:
        super(MeanAbsoluteError, self).__init__()
        self.num_classes = num_classes
    
    def forward(self, pred, labels):
        pred =  F.softmax(pred, dim=1).clamp(min=1e-7, max=1-1e-7)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        mae = 1. - torch.sum(label_one_hot * pred, dim=1)
        return mae.mean()

class GeneralizedCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes, q=0.7, *args, **kwargs):
        super(GeneralizedCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.q = q

    def forward(self, pred, labels, **kwargs):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1-1e-7)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        gce = (1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return gce.mean()

class PHuberCE(torch.nn.Module):
    def __init__(self, tau: float = 10, *args, **kwargs) -> None:
        super().__init__()
        self.tau = tau

        # Probability threshold for the clipping
        self.prob_thresh = 1 / self.tau
        # Negative of the Fenchel conjugate of base loss at tau
        self.boundary_term = - math.log(self.prob_thresh) + 1

        self.info = {'tau_t': tau}

    def forward(self, pred, labels, **kwargs) -> torch.Tensor:
        prob = pred.softmax(dim=1).clamp(min=1e-7, max=1-1e-7)
        p = prob[torch.arange(prob.shape[0]), labels]

        loss = torch.empty_like(p)
        clip = p <= self.prob_thresh
        loss[clip] = -self.tau * p[clip] + self.boundary_term
        loss[~clip] = -torch.log(p[~clip])

        return torch.mean(loss)

class TaylorCE(torch.nn.Module):
    def __init__(self, num_classes, t, *args, **kwargs) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.t = t
    
    def forward(self, pred, labels, **kwargs):
        prob = F.softmax(pred, dim=1).clamp(min=1e-7, max=1-1e-7)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        prob_y_hat = (prob * label_one_hot).sum(dim=1).reshape(-1)
        
        total_loss = torch.zeros_like(prob_y_hat)
        for i in range(self.t):
            i = i + 1
            total_loss += (1 - prob_y_hat)**i / i

        return total_loss.mean()
    
class JSLoss(torch.nn.Module):
    def __init__(self, num_classes, weights_1, *args, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.weights = [weights_1, 1 - weights_1]
        
        self.scale = -1.0 / ((1.0-self.weights[0]) * np.log((1.0-self.weights[0])))
        assert abs(1.0 - sum(self.weights)) < 0.001
    
    def forward(self, pred, labels, **kwargs):
        preds = list()
        if type(pred) == list:
            for i, p in enumerate(pred):
                preds.append(F.softmax(p, dim=1)) 
        else:
            preds.append(F.softmax(pred, dim=1))

        labels = F.one_hot(labels, self.num_classes).float() 
        distribs = [labels] + preds
        assert len(self.weights) == len(distribs)

        mean_distrib = sum([w*d for w,d in zip(self.weights, distribs)])
        mean_distrib_log = mean_distrib.clamp(1e-7, 1.0).log()
        
        jsw = sum([w * self.custom_kl_div(mean_distrib_log, d) for w,d in zip(self.weights, distribs)])
        return self.scale * jsw
    
    def custom_kl_div(self, prediction, target):
        output_pos = target * (target.clamp(min=1e-7).log() - prediction)
        zeros = torch.zeros_like(output_pos)
        output = torch.where(target > 0, output_pos, zeros)
        output = torch.sum(output, axis=1)
        return output.mean()

class SymmetricCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes, alpha, beta, *args, **kwargs):
        super(SymmetricCrossEntropy, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels, **kwargs):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        prob = torch.clamp(pred, min=1e-7, max=1-1e-7)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(prob * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss
    
class NCE_RCE(torch.nn.Module):
    def __init__(self, num_classes, alpha, beta, *args, **kwargs) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, pred, labels, **kwargs):
        return self.alpha * self.nce(pred, labels) + self.beta * self.rce(pred, labels)

    def nce(self, pred, labels):
        pred = F.log_softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        nce = -1 * torch.sum(label_one_hot * pred, dim=1) / (- pred.sum(dim=1))
        return nce.mean()
        
    def rce(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))
        return rce.mean()

class LogitClipCE(torch.nn.Module):
    def __init__(self, num_classes, tau, delta, lp) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.tau = tau
        self.delta = delta
        self.lp = lp
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels, **kwargs):
        logits = pred
        
        # logit clipping
        norms = torch.norm(logits, p=2, dim=1, keepdim=True) + 1e-7
        logits_norm = torch.div(logits, norms) * self.delta
        clip = (norms > self.tau).expand(-1, logits.shape[-1])
        logits_final = torch.where(clip, logits_norm, logits)

        # cross entropy
        ce = self.cross_entropy(logits_final, labels)
        return ce

class NCE_NNCE(torch.nn.Module):
    def __init__(self, num_classes, alpha, beta, min_prob=1e-7, *args, **kwargs) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.min_prob = min_prob
        self.A = - torch.tensor(min_prob).log()

    def forward(self, pred, labels, **kwargs):
        return self.alpha * self.nce(pred, labels) + self.beta * self.nnce(pred, labels)
    
    def nce(self, pred, labels):
        pred = F.log_softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        nce = -1 * torch.sum(label_one_hot * pred, dim=1) / (- pred.sum(dim=1))
        return nce.mean()
    
    def nnce(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = pred.clamp(min=self.min_prob, max=1)
        pred = self.A + pred.log() # - log(1e-7) - (- log(p(k|x)))
        label_one_hot = F.one_hot(labels, self.num_classes).to(pred.device)
        nnce = 1 - (label_one_hot * pred).sum(dim=1) / pred.sum(dim=1)
        return nnce.mean()

def ce(**kwargs):
    return CrossEntropy(**kwargs)

def fl(**kwargs):
    return FocalLoss(**kwargs)

def mae(**kwargs):
    return MeanAbsoluteError(**kwargs)

def gce(**kwargs):
    return GeneralizedCrossEntropy(**kwargs)

def phuber_ce(**kwargs):
    return PHuberCE(**kwargs)

def taylor_ce(**kwargs):
    return TaylorCE(**kwargs)

def js_loss(**kwargs):
    return JSLoss(**kwargs)

def sce(**kwargs):
    return SymmetricCrossEntropy(**kwargs)

def nce_rce(**kwargs):
    return NCE_RCE(**kwargs)

def lc_ce(**kwargs):
    return LogitClipCE(**kwargs)

def nce_nnce(**kwargs):
    return NCE_NNCE(**kwargs)
