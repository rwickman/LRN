import torch
import torch.nn as nn
import torch.optim as optim

class BlockTrainer:
    """Trainer for output model of Unrestricted Routing Network."""
    def __init__(self, args):
        self.args = args
        self.bce_loss_fn = nn.BCEWithLogitsLoss()
    
    def train(self, model, img, target):
        """Train model on img."""
        model_optim = optim.SGD(model.parameters(), lr=self.args.fb_lr)
        pred = model(img)

        output_loss = self.bce_loss_fn(pred[0], target)
        #print("MODEL LOSS: ", loss, "pred", torch.sigmoid(pred)[0].detach().cpu().numpy()[0], "vs", target.cpu().numpy()[0])

        output_loss.backward()
        model_optim.step()
        model_optim.zero_grad()
        output_loss = output_loss.detach().cpu().numpy()
        
        return pred, output_loss
        