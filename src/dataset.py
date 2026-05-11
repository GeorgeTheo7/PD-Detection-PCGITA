# src/dataset.py
import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset
import soundfile as sf
import torchaudio.transforms as T
import torchaudio.functional as TA_F
import torchvision.transforms as VT
from . import config

#PCGitaDataset: Custom Dataset class for loading and processing PCGITA audio data
#Παίρνουμε raw audio από το PC-GITA dataset (ήδη χωρισμένα σε chunks των 5s) και εξάγουμε ανά chunk:
#1) Φασματογραφήματα RGB 3 καναλιών (για το ResNet), 
#2) Ακολουθίες MFCC 40-διαστάσεων (για το BiLSTM), 
#3) Ακατέργαστη κυματομορφή (για το Wav2Vec2.0).


class PCGitaDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.data = pd.read_csv(csv_file) #Load CSV into DataFrame, each row -> one audio chunk
        self.root_dir = root_dir 
        self.sr = config.TARGET_SR #Target sampling rate (16kHz)
        self.max_len = config.MAX_LEN 

        #Audio Processor Init
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=1024, #64ms window at 16kHz
            win_length=int(0.04 * self.sr), #40ms window
            hop_length=int(0.01 * self.sr), #10ms hop between windows
            n_mels=200, #200 mel freq bins
            power=2.0  #power spectrogram (squared magnitude)
        )
        self.amp_to_db = T.AmplitudeToDB(stype='power') #Convert to log scale (dB)

        self.mfcc_transform = T.MFCC(
            sample_rate=self.sr,
            n_mfcc=40, #extract 40 MFCCs, captures timbral features
            log_mels=True, #use log-mel spectrogram as input
            melkwargs={
                'n_fft': 1024,
                'n_mels': 200,
                'win_length': int(0.04 * self.sr),
                'hop_length': int(0.01 * self.sr),
                'power': 2.0
            }
        )

        # ImageNet Normalization for ResNet
        self.imagenet_norm = VT.Normalize(mean=[0.485, 0.456, 0.406], #3 values for RGB channels (Log-Mel, Delta, Delta2)
                                          std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.data) #return number of samples in dataset

    def get_rgb_spectrogram(self, waveform):
        # waveform: [1, L]
        mel = self.mel_transform(waveform) + 1e-6             #convert to mel-spectrogram [1, n_mels, T]
        log_mel = self.amp_to_db(mel)                         #convert to log scale (dB)
        delta = TA_F.compute_deltas(log_mel)                  #compute delta (first derivative)
        delta2 = TA_F.compute_deltas(delta)                    #compute delta of delta (second derivative)
        rgb_spec = torch.cat([log_mel, delta, delta2], dim=0) #concatenate to 3 channels [3, n_mels, T]

        # Resize to ResNet standard size
        rgb_spec = torch.nn.functional.interpolate(
            rgb_spec.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False
        ).squeeze(0) #resize to [3, 224, 224] from [3, 200, T]

        # Per-image min-max normalization (scale values to [0,1])
        cmin = rgb_spec.amin(dim=(1, 2), keepdim=True)
        cmax = rgb_spec.amax(dim=(1, 2), keepdim=True)
        rgb_spec = (rgb_spec - cmin) / (cmax - cmin).clamp(min=1e-6)

        return self.imagenet_norm(rgb_spec) #apply ImageNet normalization

    def __getitem__(self, idx): #get one sample, idx is the index of the sample
        row = self.data.iloc[idx] #get the row corresponding to idx
        audio_path = os.path.join(self.root_dir, row['filename']) #full path to audio file

        try:
            wav_numpy, sr = sf.read(audio_path) #return audio as numpy array and sampling rate
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            wav_numpy = np.zeros(self.max_len, dtype=np.float32)
            sr = self.sr

        #Stereo -> Mono. soundfile returns (nsamples, nchannels)
        if wav_numpy.ndim > 1:
            wav_numpy = np.mean(wav_numpy, axis=1) #average channels to mono

        #Convert to tensor
        waveform = torch.from_numpy(wav_numpy).float()
        waveform = torch.nan_to_num(waveform) #replace NaNs/Infs with 0

        #Make channel dim
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)  #L -> [1, L]

        #Resample if needed
        if sr != self.sr:
            resampler = T.Resample(sr, self.sr)
            waveform = resampler(waveform)

        orig_len = min(waveform.shape[1], self.max_len) #original length before padding/truncation

        #Build attention mask: 1 for real samples (attend to this), 0 for padding (ignore)
        mask = torch.zeros(self.max_len, dtype=torch.long)
        mask[:orig_len] = 1

        # Pad or truncate to max_len
        if waveform.shape[1] > self.max_len:
            waveform = waveform[:, :self.max_len] #if longer than 80000, truncate
        else:
            padding = self.max_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding)) #if shorter, pad with zeros

        #Features
        spectrogram = self.get_rgb_spectrogram(waveform)  #[3,224,224] for ResNet
        mfcc = self.mfcc_transform(waveform).squeeze(0).transpose(0, 1)  # [T_mfcc, 40] (40 MFCCCs over T time frames for BiLSTM)
        w2v_input = waveform.squeeze(0)  #Raw waveform [L] for Wav2Vec2.0 (squeeze(0) removes channel dim)

        
        subject_id = str(row['subject_id']) if 'subject_id' in row else str(row['filename']).split('_')[0] #for subject grouping

        return {
            "spectrogram": spectrogram, #[3,224,224] for ResNet
            "mfcc": mfcc, #[T_mfcc, 40] for BiLSTM
            "wav2vec": w2v_input, #[L] for Wav2Vec2.0
            "attention_mask": mask, #[max_len] attention mask
            "label": torch.tensor(row['label'], dtype=torch.long), #0/1 label
            "filename": row['filename'],
            "subject_id": subject_id  
        }
