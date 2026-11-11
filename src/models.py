"""Multi-branch fusion models: ResNet-18 (spectrogram) + BiLSTM (MFCC) + frozen SSL embeddings.

The main model is the co-attention fusion, inspired by Shao & Fang (2025), "Alzheimer's Disease
Detection Using Co-Attention Mechanism for Acoustic and ASR-Transcribed Text Features".
The other fusion types and branch subsets are the ablation variants (see src/experiments.py).

The SSL branch is frozen, so its frame sequence is pre-computed (see features.py) and fed to the
model directly. All chunks are exactly 5 s long, so no padding mask is needed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

FUSIONS = ("coattn", "concat", "mlb", "mfb", "ssl_only", "no_ssl")


class SpectrogramEncoder(nn.Module):
    """ImageNet-pretrained ResNet-18 on the 3-channel spectrogram image -> 512-d."""

    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Identity()

    def forward(self, x):            # [B, 3, 224, 224]
        return self.resnet(x)        # [B, 512]


class MFCCEncoder(nn.Module):
    """Bidirectional LSTM over the MFCC sequence -> 128-d."""

    def __init__(self, input_dim=40, hidden_dim=64, output_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout_lstm = nn.Dropout(0.5)
        self.linear = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout_linear = nn.Dropout(0.1)

    def forward(self, x):                                   # [B, T, 40]
        self.lstm.flatten_parameters()
        _, (h_n, _) = self.lstm(x)
        x = torch.cat((h_n[-2], h_n[-1]), dim=1)            # last forward + last backward state
        return self.dropout_linear(self.linear(self.dropout_lstm(x)))   # [B, 128]


def _head(in_dim):
    return nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 2))


class FusionModel(nn.Module):
    """Configurable multi-branch model.

    fusion:
        coattn   - global context (spectrogram/MFCC) attends over the SSL frames (proposed)
        concat   - context ++ mean-pooled SSL embedding
        mlb      - Multimodal Low-rank Bilinear pooling: tanh(U x) * tanh(V y)      (Kim et al., 2017)
        mfb      - Multimodal Factorized Bilinear pooling, k=5, o=128               (Yu et al., 2017)
        ssl_only - mean-pooled SSL embedding only
        no_ssl   - spectrogram/MFCC context only
    """

    def __init__(self, fusion="coattn", use_spec=True, use_mfcc=True, ssl_dim=1024, attn_dim=128):
        super().__init__()
        if fusion not in FUSIONS:
            raise ValueError(f"Unknown fusion '{fusion}', choose from {FUSIONS}")
        self.fusion, self.use_spec, self.use_mfcc = fusion, use_spec, use_mfcc
        if use_spec:
            self.spec_encoder = SpectrogramEncoder()
        if use_mfcc:
            self.mfcc_encoder = MFCCEncoder()
        ctx = 512 * use_spec + 128 * use_mfcc
        if fusion != "ssl_only" and ctx == 0:
            raise ValueError(f"fusion '{fusion}' needs at least one of the spectrogram/MFCC branches")

        if fusion == "coattn":
            self.context_proj = nn.Linear(ctx, attn_dim)      # query
            self.ssl_proj = nn.Linear(ssl_dim, attn_dim)      # keys
            self.classifier = _head(ctx + ssl_dim)
        elif fusion == "concat":
            self.classifier = _head(ctx + ssl_dim)
        elif fusion == "mlb":
            self.U, self.V = nn.Linear(ctx, 512), nn.Linear(ssl_dim, 512)
            self.classifier = _head(512)
        elif fusion == "mfb":
            self.k, self.o = 5, 128
            self.U, self.V = nn.Linear(ctx, self.k * self.o), nn.Linear(ssl_dim, self.k * self.o)
            self.mfb_drop = nn.Dropout(0.1)
            self.classifier = _head(self.o)
        elif fusion == "ssl_only":
            self.classifier = _head(ssl_dim)
        else:  # no_ssl
            self.classifier = _head(ctx)

    def forward(self, spec, mfcc, ssl_seq, return_attention=False):
        parts = []
        if self.use_spec:
            parts.append(self.spec_encoder(spec))                      # [B, 512]
        if self.use_mfcc:
            parts.append(self.mfcc_encoder(mfcc))                      # [B, 128]
        g = torch.cat(parts, dim=1) if parts else None                 # global acoustic context
        attn = None

        if self.fusion == "coattn":
            q = self.context_proj(g).unsqueeze(1)                       # [B, 1, 128]
            k = self.ssl_proj(ssl_seq)                                  # [B, T, 128]
            attn = F.softmax(torch.bmm(q, k.transpose(1, 2)), dim=2)    # [B, 1, T]
            z = torch.cat((g, torch.bmm(attn, ssl_seq).squeeze(1)), dim=1)
        elif self.fusion == "concat":
            z = torch.cat((g, ssl_seq.mean(1)), dim=1)
        elif self.fusion == "mlb":
            z = torch.tanh(self.U(g)) * torch.tanh(self.V(ssl_seq.mean(1)))
        elif self.fusion == "mfb":
            z = self.mfb_drop(self.U(g) * self.V(ssl_seq.mean(1))).view(-1, self.o, self.k).sum(2)
            z = F.normalize(torch.sign(z) * torch.sqrt(z.abs() + 1e-8), dim=1)   # power + L2 normalisation
        elif self.fusion == "ssl_only":
            z = ssl_seq.mean(1)
        else:
            z = g

        logits = self.classifier(z)
        if return_attention:
            return logits, (attn.squeeze(1) if attn is not None else None)
        return logits


def CoAttentionModel(ssl_dim=1024):
    """The main model: all three branches with co-attention fusion."""
    return FusionModel("coattn", True, True, ssl_dim)
