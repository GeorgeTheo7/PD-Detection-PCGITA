import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import Wav2Vec2Model

#The 3-Channel Co-Attention Model


# 1. SPECTROGRAM ENCODER (ResNet18)

class SpectrogramEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        #1.Load Pre-trained ResNet18
        #ResNet expects 3 Channels (RGB).
        #Our dataset provides [Log-Mel, Delta, Delta2] as 3 channels

        try:
            self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except:
            #Fallback for older torch versions
            self.resnet = models.resnet18(pretrained=True)

        #2.Remove Classifier
        #We just want the features (512 dim), not the prediction
        self.resnet.fc = nn.Identity() #remove final FC layer, i get the 512-dim feature vector

        
        self.resnet.avgpool = nn.AdaptiveAvgPool2d((1, 1)) #ensure avgpool outputs 1x1 feature map

    def forward(self, x): 
        # Input x: [Batch, 3, 224, 224]
        # Pass through standard ResNet layers
        x = self.resnet.conv1(x) #[Batch, 64, H/2, W/2] -> [B, 64, 112, 112]
        x = self.resnet.bn1(x) #Batch normalization
        x = self.resnet.relu(x) #ReLU activation
        x = self.resnet.maxpool(x) #[Batch, 64, H/4, W/4] -> [B, 64, 56, 56]

        x = self.resnet.layer1(x) #
        x = self.resnet.layer2(x) #[Batch, 128, H/8, W/8] -> [B, 128, 28, 28]
        x = self.resnet.layer3(x) #[Batch, 256, H/16, W/16] -> [B, 256, 14, 14]
        x = self.resnet.layer4(x) #[Batch, 512, H/32, W/32] -> [B, 512, 7, 7]

        x = self.resnet.avgpool(x) #[Batch, 512, 1, 1]
        x = torch.flatten(x, 1) # -> [Batch, 512]

        return x #512-dim feature vector capturing visual texture patterns from spectrogram



# 2. MFCC ENCODER (BiLSTM)

class MFCCEncoder(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=64, output_dim=128):
        super().__init__()

        #1.Bidirectional LSTM (because forward LSTM sees past -> present, backward sees future -> present)
        #hidden_dim=64. Since it's bidirectional, the output will be 64*2 = 128.
        self.lstm = nn.LSTM(
            input_size=input_dim, #40 MFCC coefficients
            hidden_size=hidden_dim, #64 hidden states per direction
            batch_first=True, 
            bidirectional=True
        )

        #2.First Dropout (0.5)
        self.dropout_lstm = nn.Dropout(0.5) #regularization on the LSTM output

        #3.Linear Projection
        self.linear = nn.Linear(hidden_dim * 2, output_dim) #128-dim output feature vector

        #4.Second Dropout (0.1)
        self.dropout_linear = nn.Dropout(0.1) #regularization on the projected features

    def forward(self, x):
        # Input x: [Batch, Time_Steps, 40]

        #Pass through LSTM
        self.lstm.flatten_parameters() #Optimization for GPU
        _, (h_n, _) = self.lstm(x) #returns all hidden states [B,T,128] and (h_n -> final hidden states [2, B, 64], c_n -> final cell states)

        #"Summary" Vector that has 'seen' the entire MFCC sequence from both directions.
        #h_n contains the final hidden state for both directions.
        #concatenate the last Forward state [-2] and last Backward state [-1]
        x = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1) # -> [Batch, 128]

        #Paper's Regularization Sequence
        x = self.dropout_lstm(x) #Dropout after LSTM
        x = self.linear(x) #Linear projection to 128-dim
        x = self.dropout_linear(x) #Dropout after Linear

        return x #128-dimensional vector capturing temporal dynamics of MFCCs.


# 3. WAV2VEC2 ENCODER
class Wav2VecEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        #1.Load Pre-trained Spanish Model
        print("Initializing Wav2Vec2 Model (XLSR-53 Spanish)...")
        self.w2v_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53-spanish")
        #Wav2Vec2 -> Self-supervised model that learns speech representations from raw audio, here fine-tuned on Spanish data

        #2.Freezing Strategy
        self.w2v_model.freeze_feature_encoder() #freezes only the CNN feature extractor layers

        #3.Memory Optimization (Gradient Checkpointing)
        if hasattr(self.w2v_model, "gradient_checkpointing_enable"):
             self.w2v_model.gradient_checkpointing_enable() #saves memory by recomputing some activations during backprop

    def forward(self, x, mask=None):
        #input: raw waveform [B, 80000]
        outputs = self.w2v_model(
            input_values=x,
            attention_mask=mask,
            output_hidden_states=False
        )

        sequence_output = outputs.last_hidden_state

        return sequence_output #Shape: [Batch, ~249, 1024], 249 time frames, each a 1024-dim representation


# 4. FULL CO-ATTENTION MODEL
class ParkinsonCoAttentionModel(nn.Module):
    def __init__(self):
        super().__init__()

        #1.Initialize Branches
        self.spec_encoder = SpectrogramEncoder() # Output: 512
        self.mfcc_encoder = MFCCEncoder()        # Output: 128
        self.w2v_encoder = Wav2VecEncoder()      # Output: Seq [Batch, T, 1024]

        #2.Co-Attention Mechanism
        #Project Global Context (Spec + MFCC) -> Query Vector
        self.context_proj = nn.Linear(512 + 128, 128) #Maps 640-dim global context to 128-dim query

        #Project Wav2Vec Sequence -> Key Vectors
        self.w2v_proj = nn.Linear(1024, 128) #Maps 1024-dim Wav2Vec frames to 128-dim keys

        #3.Final Classifier
        #classifier MLP, input 1664-dim (512+128+1024), hidden 64, output 2 (PD vs Control)
        self.classifier = nn.Sequential(
            nn.Linear(512 + 128 + 1024, 64), #1664 -> 64
            nn.ReLU(),
            nn.Dropout(0.2), #Dropout for regularization
            nn.Linear(64, 2) #64 -> 2 (PD vs Control)
        )

    def forward(self, spec, mfcc, audio, mask):
        #Extract Features
        feat_s = self.spec_encoder(spec)       # [Batch, 512]
        feat_m = self.mfcc_encoder(mfcc)       # [Batch, 128]
        seq_w  = self.w2v_encoder(audio, mask) # [Batch, T, 1024]

        #Co-Attention Fusion

        #1.Create Global Context (Spec + MFCC)
        global_context = torch.cat((feat_s, feat_m), dim=1) # [Batch, 640], overall speech characteristics

        #2.Calculate Attention Scores (Dot Product)
        #Context mapped to 128 dim
        query = self.context_proj(global_context).unsqueeze(1) # [Batch, 1, 128], "What am I looking for in the Wav2Vec sequence?"
        #Sequence mapped to 128 dim
        keys = self.w2v_proj(seq_w) # [Batch, T, 128], each frame becomes 128-dim "key", "What does each time frame contain"

        #Multiply Query * Keys to get Scores
        #shape: [Batch, 1, T], one score per time frame
        attn_scores = torch.bmm(query, keys.transpose(1, 2)) #measures similarity between query and each key

        # The global context (from spectrogram and MFCC) guides the model to focus on relevant parts of the Wav2Vec sequence.
        # If the spectogram shows tremor patterns, attention might focus on Wav2Vec frames with similar characteristics.

        #3.MASKING
        #Downsample the 80,000 mask to T frames (e.g. 249)
        B, T, _ = seq_w.shape

        #create a mask of valid frames
        real_len_mask = F.interpolate(mask.unsqueeze(1).float(), size=T, mode='nearest').squeeze(1) #mask 80000 samples, wav2vec2 output 249 frames, downsamples mask to match frame count

        
        #(1.0 means keep, 0.0 means mask)
        attn_scores = attn_scores.masked_fill(real_len_mask.unsqueeze(1) == 0, -1e4) #if mask is 0 (padding), set score to large negative value. After softmax, these become essentially 0 (e^(-10000) ≈ 0)

        #4.Softmax to get Weights
        attn_weights = F.softmax(attn_scores, dim=2) #[Batch, 1, T], softmax converts scores to probabilities (masked frames get ~0 weight)

        #5.Apply Weights to Sequence (Weighted Sum)
        feat_w_final = torch.bmm(attn_weights, seq_w).squeeze(1) #[Batch, 1024]

        #Final Classification
        combined_features = torch.cat((global_context, feat_w_final), dim=1) #[Batch, 1664]
        logits = self.classifier(combined_features) #passing to MLP

        return logits