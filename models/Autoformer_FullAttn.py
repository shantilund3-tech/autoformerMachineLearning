
from models import Autoformer
from layers.AutoCorrelation import AutoCorrelationLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer


class Model(Autoformer.Model):
    # Autoformer with every AutoCorrelationLayer swapped for a standard multi-head self-attention layer.
    def __init__(self, configs):
        super().__init__(configs)
        for module in list(self.modules()):
            for name, child in list(module.named_children()):
                if isinstance(child, AutoCorrelationLayer):
                    setattr(module, name, AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads))
