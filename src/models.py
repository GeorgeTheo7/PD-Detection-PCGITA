"""Multi-branch co-attention model (ResNet-18 + BiLSTM + frozen Wav2Vec2).

Inspired by Shao & Fang (2025), "Alzheimer's Disease Detection Using Co-Attention Mechanism
for Acoustic and ASR-Transcribed Text Features".

The Wav2Vec2 branch is frozen, so its frame sequence is pre-computed (see features.py) and fed
to the model directly. All chunks are exactly 5 s long, so no padding mask is needed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


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


class CoAttentionModel(nn.Module):
    """Global context (spectrogram + MFCC) attends over the Wav2Vec2 frame sequence."""

    def __init__(self, w2v_dim=1024, attn_dim=128):
        super().__init__()
        self.spec_encoder = SpectrogramEncoder()
        self.mfcc_encoder = MFCCEncoder()
        self.context_proj = nn.Linear(512 + 128, attn_dim)   # query
        self.w2v_proj = nn.Linear(w2v_dim, attn_dim)         # keys
        self.classifier = nn.Sequential(
            nn.Linear(512 + 128 + w2v_dim, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 2))

    def forward(self, spec, mfcc, w2v_seq, return_attention=False):
        g = torch.cat((self.spec_encoder(spec), self.mfcc_encoder(mfcc)), dim=1)   # [B, 640]
        q = self.context_proj(g).unsqueeze(1)                                       # [B, 1, 128]
        k = self.w2v_proj(w2v_seq)                                                  # [B, T, 128]
        attn = F.softmax(torch.bmm(q, k.transpose(1, 2)), dim=2)                    # [B, 1, T]
        w = torch.bmm(attn, w2v_seq).squeeze(1)                                     # [B, 1024]
        logits = self.classifier(torch.cat((g, w), dim=1))                          # [B, 2]
        return (logits, attn.squeeze(1)) if return_attention else logits
