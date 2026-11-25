
import torch
import torch.nn as nn
from models import Autoformer
from layers.Autoformer_EncDec import series_decomp, my_Layernorm


class NoDecomp(nn.Module):
    # Replaces series_decomp: everything stays in the seasonal path, the trend is always zero.
    def forward(self, x):
        return x, torch.zeros_like(x)


class Model(Autoformer.Model):
    # Autoformer without series decomposition (Auto-Correlation kept).
    def __init__(self, configs):
        super().__init__(configs)
        for module in list(self.modules()):
            for name, child in list(module.named_children()):
                if isinstance(child, series_decomp):
                    setattr(module, name, NoDecomp())
                elif isinstance(child, my_Layernorm):
                    setattr(module, name, nn.LayerNorm(configs.d_model))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init = torch.cat([x_enc[:, -self.label_len:, :], zeros], dim=1)   # raw last 48 steps + zeros
        trend_init = torch.zeros_like(seasonal_init)                                # no trend path
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        dec_out = trend_part + seasonal_part
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        return dec_out[:, -self.pred_len:, :]
