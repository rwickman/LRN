import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


from taskrouter.models.blocks import ImageEncoder, TransformerEncoder
from taskrouter.helpers.model_util import extract_lstm_out

class Discriminator(nn.Module):
    """Used to approximate p(theta | S)."""
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.img_encoder = ImageEncoder(self.args, self.config)
        
        # self.mb_embs = nn.Embedding(
        #     self.config.num_modules, self.args.emb_size)
        self.logit_to_emb = nn.Linear(
            self.config.num_modules,
            self.args.emb_size,
            bias=False) 

        self.lstm = nn.LSTM(
            input_size=self.args.emb_size,
            hidden_size=self.args.emb_size,
            batch_first=True)
        self._init_lstm()
        
        
        self.fc_out = nn.Linear(self.args.emb_size, self.config.num_modules)
        
    def _init_lstm(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.kaiming_normal_(param)

    def predict(self, x, mb_logits):
        # Get embedding for input image
        img_emb = self.img_encoder(x)

        # Get embedding for modules
        mb_embs = self.logit_to_emb(mb_logits)

        # Concatenate together
        inp_seq = torch.cat((img_emb, mb_embs), dim=1)
        
        lstm_out, _ = self.lstm(inp_seq)
        out_pred = self.fc_out(lstm_out)
        return out_pred

    def forward(self, x: torch.Tensor, mb_logits: torch.Tensor, seq_lens: list, data_idxs: torch.Tensor):
        """Args:
            x: tensor input image,
            mb_logits: tensor of router logit outputs.
        """
        # Get embedding for input image
        img_emb = self.img_encoder(x)

        # Get embedding for modules
        mb_embs = self.logit_to_emb(mb_logits)

        # Concatenate together
        inp_seq = torch.cat((img_emb, mb_embs), dim=1)
        
        # Run through LSTM
        packed = pack_padded_sequence(inp_seq, seq_lens, batch_first=True)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        
        # Extract only the indices you want predictions for
        # lstm_out = lstm_out.contiguous()
        # lstm_out = lstm_out.view(-1, lstm_out.shape[2])
        # print("data_idxs", data_idxs)
        lstm_out = extract_lstm_out(lstm_out, data_idxs)

        # Make predictions using the output
        out_pred = self.fc_out(lstm_out)

        return out_pred