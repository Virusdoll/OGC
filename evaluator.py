import torch

class Evaluator():
    def __init__(self, data_loader, logger, writer, device) -> None:
        self.device = device
        self.data_loader = data_loader
        self.logger = logger
        self.writer = writer
        # self.loss_meters = AverageMeter()
        self.acc_meters = AverageMeter()
        self.last_10_acc = Queue(10)

        self.best_acc = -1
        self.best_epoch = -1

    @torch.no_grad()
    def eval(self, model, epoch):
        model.eval()
        for images, labels in self.data_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            pred = self.eval_batch(model, images, labels)
            self.update_meters(pred, labels)
        self.update_best(epoch)
        self.update_last_10_acc()
        self.log(epoch)
        self.write(epoch)
        self.reset_meters()
    
    def eval_batch(self, model, images, labels):
        pred = model(images)
        # loss = loss_function(pred=pred, labels=labels)
        # return pred, loss
        return pred
    
    def update_meters(self, pred, labels):
        _, pred_idx = torch.max(pred.data, 1)
        batch_acc = (pred_idx == labels).float().mean().item()

        # self.loss_meters.update(loss.item(), labels.shape[0])
        self.acc_meters.update(batch_acc, labels.shape[0])

    def reset_meters(self):
        # self.loss_meters.reset()
        self.acc_meters.reset()

    def log(self, epoch):
        display = {'Epoch': epoch,
                   'Eval_Accuracy': self.acc_meters.avg,
                #    'Eval_Loss': self.loss_meters.avg,
                   'Best_Accuracy': self.best_acc,
                   'Best_Epoch': self.best_epoch}
        self.logger.info(display)
    
    def write(self, epoch):
        if self.writer is None:
            return

        self.writer.add_scalar('Eval/accuracy', self.acc_meters.avg, epoch)
        # self.writer.add_scalar('Eval/loss', self.loss_meters.avg, epoch)
    
    def update_best(self, epoch):
        if self.acc_meters.avg >= self.best_acc:
            self.best_acc = self.acc_meters.avg
            self.best_epoch = epoch
    
    def update_last_10_acc(self):
        batch_acc = self.acc_meters.avg
        self.last_10_acc.enqueue(batch_acc)

class AverageMeter(object):
    def __init__(self) -> None:
        self.reset()
    
    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
    
    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt if self.cnt != 0 else 0

class Queue:
    def __init__(self, max_size=10):
        self.items = []
        self.max_size = max_size
    
    def is_empty(self):
        return len(self.items) == 0
    
    def enqueue(self, item):
        if self.size() == self.max_size:
            self.dequeue()
        self.items.append(item)
    
    def dequeue(self):
        if not self.is_empty():
            return self.items.pop(0)
        else:
            raise IndexError('dequeue from an empty queue')
    
    def size(self):
        return len(self.items)