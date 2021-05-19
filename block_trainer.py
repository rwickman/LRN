import torch
import torch.nn as nn

class BlockTrainer:
    """Trainer for output model of Unrestricted Routing Network."""
    def __init__(self, args, fb_model, fb_optim):
        self.args = args
        if self.args.num_classes == 1:
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()
        
        self._fb_model = fb_model
        self.fb_optim = fb_optim

    def train(self, img, actions, target):
        """Train model on img."""
        batch_size = img.shape[0]
        pred = self._fb_model(img, actions)
        total_bce_loss = torch.zeros(1).cuda()
        output_losses = torch.zeros(batch_size).cuda()
        for i in range(batch_size):
            if self.args.num_classes == 1:                
                output_loss = self.loss_fn(pred[i], target[i].float())
            else:
                output_loss = self.loss_fn(pred[i].unsqueeze(0), target[i])
            
            output_losses[i] = output_loss.detach()
            total_bce_loss += output_loss
        #print("MODEL LOSS: ", loss, "pred", torch.sigmoid(pred)[0].detach().cpu().numpy()[0], "vs", target.cpu().numpy()[0])
        total_bce_loss = total_bce_loss/batch_size
        total_bce_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._fb_model.parameters(), self.args.max_grad_norm)
        self.fb_optim.step()
        self.fb_optim.zero_grad()
        output_loss = output_loss.detach().cpu().numpy()
        
        return pred, output_losses
        
