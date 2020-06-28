import torch
import numpy as np
from tqdm import tqdm

def run_epoch(model, criterion, dataloader, epoch, optimizer=None):
    if optimizer is None:
        model.eval()
    else:
        model.train()
    
    running_loss = 0.0
    
    loader = tqdm(
            dataloader,
            ncols=0,
            desc="{1} E{0:02d} ".format(epoch, "val " if optimizer is None else "train "),
        )
    iters_per_epoch = len(dataloader)

    for i, sample in enumerate(loader, start=epoch * iters_per_epoch):
        x, y = sample
        x = x.to("cuda")
        y = y.to("cuda")
        
        if optimizer is not None:
            with torch.set_grad_enabled(True):
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
        else:
            with torch.set_grad_enabled(False):
                pred = model(x)
                loss = criterion(pred, y)
        
        running_loss += loss.item() * x.size(0)
    
    epoch_loss = running_loss / len(dataloader)
    return epoch_loss

def inference(model, x, icls_dict, thresh, cuda=True):
    if cuda:
        model = model.cuda()
        model.eval()
        x = x.cuda()
    else:
        model = model.cpu()
        model.eval()
        x = x.cpu()
    
    pred_logits =  model(x)
    pred = torch.sigmoid(pred_logits) > thresh

    output = []
    for xx in pred.cpu().numpy():
        output.append([icls_dict[cid] for cid in np.where(xx == True)[0]])

    return output