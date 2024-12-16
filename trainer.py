import torch
import torch.nn as nn

class Trainer():
    def __init__(self, train_dataloader, logger, writer, device, args):
        # setting trainer
        self.device = device
        self.train_dataloader = train_dataloader
        self.logger = logger
        self.writer = writer
        self.log_freq = args.log_freq
        self.write_freq = args.write_freq
        self.ce_loss = nn.CrossEntropyLoss()
        
        # training info
        self.info = {
            'epoch': 0,
            'step': 0,
            'lr': 0,
            'loss': 0,
            'accuracy': 0,
            'accuracy_clean': 0,
            'accuracy_noisy': 0,
            'risk_ce_clean': 0,
            'risk_ce_noisy': 0,
            'e_grad_ce_abs_clean': 0,
            'e_grad_ce_abs_noisy': 0,
            'e_grad_ce_abs_clean_clip': 0,
            'e_grad_ce_abs_noisy_clip': 0,
        }
        # method info
        self.method_info = {}

    def train(self, model, optimizer, method, epoch):
        self.info['epoch'] = epoch
        model.train()

        for index, images, labels, true_labels in self.train_dataloader:
            self.info['step'] += 1
            
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            true_labels = true_labels.to(self.device, non_blocking=True)

            pred, loss = self.train_batch(model, optimizer, method,
                                          images, labels)
            
            self.update_info(pred, loss, labels, true_labels, optimizer, method)
            self.log_info()
            self.write_info()
        
    def train_batch(self, model, optimizer, method, images, labels):
        model.zero_grad()
        optimizer.zero_grad()

        pred = model(images)
        loss = method(pred=pred, labels=labels,
                      lr=optimizer.param_groups[0]['lr'],
                      epoch=self.info['epoch'])
    
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        return pred, loss
    
    def log_info(self):
        if not self.info['step'] % self.log_freq == 0:
            return
        
        self.logger.info({**self.info, **self.method_info})

    def write_info(self):
        if self.writer is None:
            return
        
        if not self.info['step'] % self.write_freq == 0:
            return
        
        for k, v in self.info.items():
            self.writer.add_scalar(f'Train/{k}', v, self.info['step'])

        for k, v in self.method_info.items():
            self.writer.add_scalar(f'Method/{k}', v, self.info['step'])
    
    def update_info(self, pred, loss, labels, true_labels, optimizer, method):
        ######################
        # update method info #
        ######################

        if hasattr(method, 'info'):
            self.method_info = method.info

        if 'tau_t' in self.method_info.keys():
            tau_t = self.method_info['tau_t']
        else:
            tau_t = 1 / 1e-7

        ########################
        # update training info #
        ########################

        pred = pred.detach().clone()

        clean_idx = true_labels == labels
        noisy_idx = true_labels != labels
        _, pred_class = torch.max(pred.data, 1)

        acc = (pred_class == labels).float().mean().item()
        risk_ce = self.ce_loss(pred, labels).item()
        
        if clean_idx.float().sum() == 0:
            acc_clean = 0
            risk_ce_clean = 0
        else:
            acc_clean = (pred_class[clean_idx] == labels[clean_idx]).float().mean().item()
            risk_ce_clean = self.ce_loss(pred[clean_idx], labels[clean_idx]).item()
        
        if noisy_idx.float().sum() == 0:
            acc_noisy = 0
            risk_ce_noisy = 0
        else:
            acc_noisy = (pred_class[noisy_idx] == labels[noisy_idx]).float().mean().item()
            risk_ce_noisy = self.ce_loss(pred[noisy_idx], labels[noisy_idx]).item()
        
        # calc ce grad ratio
        prob = torch.softmax(pred, dim=1).clamp(min=1e-7, max=1-1e-7)[torch.arange(pred.shape[0]), labels]
        grad_ce_abs = 1 / prob

        if clean_idx.float().sum() == 0:
            grad_ce_abs_clean = 0
            e_grad_ce_abs_clean_clip = 0
        else:
            grad_ce_abs_clean = grad_ce_abs[clean_idx]
            # E[g]
            e_grad_ce_abs_clean = grad_ce_abs_clean.mean().item()
            # E[\bar{g}]
            
            clip_clean = grad_ce_abs_clean > tau_t
            grad_ce_abs_clean[clip_clean] = tau_t
            e_grad_ce_abs_clean_clip = grad_ce_abs_clean.mean().item()
        
        if noisy_idx.float().sum() == 0:
            grad_ce_abs_noisy = 0
            e_grad_ce_abs_noisy_clip = 0
        else:
            grad_ce_abs_noisy = grad_ce_abs[noisy_idx]
            # E[g]
            e_grad_ce_abs_noisy = grad_ce_abs_noisy.mean().item()
            # E[\bar{g}]
            clip_noisy = grad_ce_abs_noisy > tau_t
            grad_ce_abs_noisy[clip_noisy] = tau_t
            e_grad_ce_abs_noisy_clip = grad_ce_abs_noisy.mean().item()

        self.info['lr'] = optimizer.param_groups[0]['lr']
        self.info['loss'] = loss.item()
        self.info['accuracy'] = acc
        self.info['accuracy_clean'] = acc_clean
        self.info['accuracy_noisy'] = acc_noisy
        self.info['risk_ce'] = risk_ce
        self.info['risk_ce_clean'] = risk_ce_clean
        self.info['risk_ce_noisy'] = risk_ce_noisy
        self.info['e_grad_ce_abs_clean'] = e_grad_ce_abs_clean
        self.info['e_grad_ce_abs_noisy'] = e_grad_ce_abs_noisy
        self.info['e_grad_ce_abs_clean_clip'] = e_grad_ce_abs_clean_clip
        self.info['e_grad_ce_abs_noisy_clip'] = e_grad_ce_abs_noisy_clip
    