import torch
import torch.nn as nn

class BlockTrainer:
    """Trainer for output model of Unrestricted Routing Network."""
    def __init__(self, args, fb_model, fb_optim):
        self.args = args
        self.bce_loss_fn = nn.BCEWithLogitsLoss()
        self._fb_model = fb_model
        self.fb_optim = fb_optim
    
    def train(self, img, actions, target):
        """Train model on img."""
        pred = self._fb_model(img, actions)

        output_loss = self.bce_loss_fn(pred[0], target)
        #print("MODEL LOSS: ", loss, "pred", torch.sigmoid(pred)[0].detach().cpu().numpy()[0], "vs", target.cpu().numpy()[0])
        output_loss.backward()
        self.fb_optim.step()
        self.fb_optim.zero_grad()
        output_loss = output_loss.detach().cpu().numpy()
        
        return pred, output_loss
        