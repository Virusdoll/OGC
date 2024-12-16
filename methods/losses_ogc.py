import torch
import torch.nn.functional as F
import numpy as np
import math
from scipy import stats
from collections import deque
from .tools import fit_gmm, binary_search


class OptimizedGradientClippingBase(torch.nn.Module):
    def __init__(self, epsilon, time_frame_s, queue_size_q,
                 binary_search_tol, binary_search_max_iter, *args, **kwargs) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.time_frame_s = time_frame_s
        self.queue_size_q = queue_size_q
        self.binary_search_tol = binary_search_tol
        self.binary_search_max_iter = binary_search_max_iter

        # boundary
        self.min_prob = 1e-7
        self.max_prob = 1-1e-7
        self.min_H = - math.log(self.max_prob)
        self.max_H = - math.log(self.min_prob)
        
        # tau info
        self.lr_t = 1.0
        self.prob_thr_t = self.min_prob
        self.tau_t = self.mapping_p_to_grad(torch.tensor(self.prob_thr_t)).abs().item()
        self.tau_update_count = 0
        
        # queue
        self.queue_H = deque(maxlen=queue_size_q)

        # pre sample
        self.sample_H = np.linspace(self.min_H, self.max_H, 1000)
        self.sample_p = self.mapping_H_to_p(torch.from_numpy(self.sample_H)).detach().numpy()
        self.sample_grad = self.mapping_p_to_grad(torch.from_numpy(self.sample_p)).detach().numpy()
        self.sample_grad_abs = np.abs(self.sample_grad)

        # method info
        self.info = {'tau_t': self.tau_t,
                     'prob_thr_t': self.prob_thr_t,
                     'mean_clean': 0,
                     'covar_clean':0,
                     'mean_noisy': 0,
                     'covar_noisy': 0}

    def forward(self, pred, labels, **kwargs) -> torch.Tensor:
        # get lr_t
        self.lr_t = kwargs['lr']
        
        # calc prob
        prob = pred.softmax(dim=1).clamp(min=self.min_prob, max=self.max_prob)
        p_y_hat = prob[torch.arange(pred.shape[0]), labels]

        # add H to queue
        self.add_H_to_queue(p_y_hat)

        # update tau_t
        self.tau_update_count += 1
        if self.tau_update_count == self.time_frame_s:
            self.prob_thr_t, self.tau_t = self.get_tau_t()
        self.tau_update_count = self.tau_update_count % self.time_frame_s

        # info
        self.info['tau_t'] = self.tau_t
        self.info['prob_thr_t'] = self.prob_thr_t

        # boundary
        boundary_term = self.mapping_p_to_loss(torch.tensor(self.prob_thr_t))
        
        # compute loss
        loss = torch.empty_like(p_y_hat)
        clip = p_y_hat <= self.prob_thr_t
        loss[clip] = 1 - self.tau_t * p_y_hat[clip] + boundary_term
        loss[~clip] = self.mapping_p_to_loss(p_y_hat[~clip])

        return torch.mean(loss)
    
    def mapping_p_to_loss(self, p_y_hat:torch.Tensor):
        pass
    
    def mapping_p_to_grad(self, p_y_hat:torch.Tensor):
        pass

    def mapping_p_to_H(self, p_y_hat:torch.Tensor):
        return - torch.log(p_y_hat)
    
    def mapping_H_to_p(self, H:torch.Tensor):
        return torch.exp(-H)

    def add_H_to_queue(self, prob_y_hat):
        # calc H
        H_gpu = self.mapping_p_to_H(prob_y_hat)

        # gpu to cpu, torch to numpy
        H_cpu = H_gpu.cpu().detach().numpy().reshape(-1)

        # add to queue
        self.queue_H.extend(H_cpu.tolist())
    
    def get_tau_t(self):
        # queue not full
        if len(self.queue_H) < self.queue_size_q:
            return self.prob_thr_t, self.tau_t
        
        # get H queue
        queue_H = np.array(list(self.queue_H)).reshape(-1, 1)

        # fit gmm
        mean_clean, covar_clean, mean_noisy, covar_noisy = fit_gmm(queue_H)

        # record info
        self.info['mean_clean'] = mean_clean
        self.info['covar_clean'] = covar_clean
        self.info['mean_noisy'] = mean_noisy
        self.info['covar_noisy'] = covar_noisy

        # calc truncted normal distribution pdf
        loc_clean, scale_clean = mean_clean, np.sqrt(covar_clean)
        loc_noisy, scale_noisy = mean_noisy, np.sqrt(covar_noisy)
        a_clean = (self.min_H - loc_clean) / scale_clean
        b_clean = (self.max_H - loc_clean) / scale_clean
        a_noisy = (self.min_H - loc_noisy) / scale_noisy
        b_noisy = (self.max_H - loc_noisy) / scale_noisy
        self.p_H_clean = stats.truncnorm.pdf(self.sample_H, a_clean, b_clean,
                                             loc_clean, scale_clean).reshape(-1)
        self.p_H_noisy = stats.truncnorm.pdf(self.sample_H, a_noisy, b_noisy,
                                             loc_noisy, scale_noisy).reshape(-1)
        
        # calc tau_t
        prob_thr_t = binary_search(self.min_prob, self.max_prob,
                                   self.calc_grad_ratio, 1 + self.lr_t * self.epsilon,
                                   self.binary_search_tol, self.binary_search_max_iter,
                                   is_func_decrease=True)
        tau_t = self.mapping_p_to_grad(torch.tensor(prob_thr_t)).abs().item()

        return prob_thr_t, tau_t
    
    def calc_grad_ratio(self, prob):
        grad = self.mapping_p_to_grad(torch.tensor(prob)).item()
        tau = np.abs(grad)
        clip = self.sample_grad_abs > tau

        e_grad_noisy = np.zeros_like(self.p_H_noisy)
        e_grad_noisy[~clip] = (self.p_H_noisy * self.sample_grad_abs)[~clip]
        e_grad_noisy[clip] = (self.p_H_noisy * tau)[clip]

        e_grad_clean = np.zeros_like(self.p_H_clean)
        e_grad_clean[~clip] = (self.p_H_clean * self.sample_grad_abs)[~clip]
        e_grad_clean[clip] = (self.p_H_clean * tau)[clip]

        grad_ratio = e_grad_noisy.mean() / e_grad_clean.mean()
        grad_ratio = grad_ratio * self.p_H_clean.mean() / self.p_H_noisy.mean()
        return grad_ratio

class OGC_CE(OptimizedGradientClippingBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def mapping_p_to_loss(self, p_y_hat:torch.Tensor):
        loss = - torch.log(p_y_hat)
        return loss
    
    def mapping_p_to_grad(self, p_y_hat: torch.Tensor):
        grad = - 1 / p_y_hat
        return grad

class OGC_FL(OptimizedGradientClippingBase):
    
    def __init__(self, epsilon, time_frame_s, queue_size_q,
                 binary_search_tol, binary_search_max_iter, gamma,
                 *args, **kwargs) -> None:
        self.gamma = gamma
        super().__init__(epsilon, time_frame_s, queue_size_q,
                         binary_search_tol, binary_search_max_iter,
                         *args, **kwargs)
    
    def mapping_p_to_loss(self, p_y_hat: torch.Tensor):
        loss = - (1 - p_y_hat)**self.gamma * p_y_hat.log()
        return loss
    
    def mapping_p_to_grad(self, p_y_hat: torch.Tensor):
        grad = self.gamma * p_y_hat.log() * (1 - p_y_hat)**(self.gamma - 1)\
               - (1 - p_y_hat)**self.gamma / p_y_hat
        return grad

class OGC_GCE(OptimizedGradientClippingBase):
    def __init__(self, epsilon, time_frame_s, queue_size_q,
                 binary_search_tol, binary_search_max_iter,
                 q, *args, **kwargs) -> None:
        self.q = q
        super().__init__(epsilon, time_frame_s, queue_size_q,
                         binary_search_tol, binary_search_max_iter,
                         *args, **kwargs)
    
    def mapping_p_to_loss(self, p_y_hat: torch.Tensor):
        loss = (1. - torch.pow(p_y_hat, self.q)) / self.q
        return loss
    
    def mapping_p_to_grad(self, p_y_hat: torch.Tensor):
        grad = - p_y_hat**(self.q - 1)
        return grad

def ogc_ce(**kwargs):
    return OGC_CE(**kwargs)

def ogc_fl(**kwargs):
    return OGC_FL(**kwargs)

def ogc_gce(**kwargs):
    return OGC_GCE(**kwargs)
    