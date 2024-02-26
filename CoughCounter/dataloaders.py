import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import numpy as np
import re





class AudioSeg5s(Dataset):

    def __init__(self,label_dir='vad_audio'):
        self.labels = self.create_dataframe(label_dir)
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        signal,sr = torchaudio.load(audio_sample_path)
        signal = self.padding(signal,sr)
        return signal


    def _get_audio_sample_path(self, index):
        return self.labels.iloc[index,0]
    
    def padding(self,signal,sr):
        diff = sr*5 - signal.shape[-1]
        if diff:
            signal = F.pad(signal,(0,diff),'constant',0)
        return signal

    
    def create_dataframe(self,folder_path):

        wav_files = [file for file in os.listdir(folder_path) if file.endswith('.wav')]
        
        def sort_key(filename):
            name, timestamp = filename.rsplit('_', 1)
            return name, float(timestamp.split('.')[0])
        
        wav_files.sort(key=lambda x: sort_key(x))

        df = pd.DataFrame(columns=['Path', 'CoughSegment', 'Coughs'])

        # Construct the DataFrame with the file paths
        df['Path'] = [os.path.join(folder_path, file) for file in wav_files]
        df['CoughSegment'] = ''
        df['Coughs'] = ''
        df.to_csv('./coughsegs.csv',index=False)
        return df
    
    def update_dataframe(self,pred):
        df = pd.read_csv('./coughsegs.csv')
        df['CoughSegment'] = pred
        df.to_csv('./coughsegs.csv',index=False)
        
#########################################################################################################################

class CoughSeg5s(Dataset):

    def __init__(self,label_dir='vad_audio'):
        self.cough_segs()
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        signal,sr = torchaudio.load(audio_sample_path)
        signal = self.padding(signal,sr)
        return signal

    def _get_audio_sample_path(self, index):
        return self.labels.iloc[index,0]
    
    def padding(self,signal,sr):
        diff = sr*5 - signal.shape[-1]
        if diff:
            signal = F.pad(signal,(0,diff),'constant',0)
        return signal

    def cough_segs(self):
        df = pd.read_csv('./coughsegs.csv')
        self.labels = df[df['CoughSegment']==1].reset_index(drop=True)
        
    
    def update_dataframe(self,pred):
        for i, coughs in enumerate(pred):
            self.labels.iloc[i,-1] = str(coughs)
        self.labels.to_csv('./detections.csv', index=False)
    
    def generate_result_txt(self,dataframe, output_file='result.txt'):
        print('generating result...')
        res_fold = 'cough_monitor_results/'
        if not os.path.exists(res_fold):
            os.makedirs(res_fold)
        with open(res_fold+output_file, 'w') as f:
            previous_file = None
            total_cough_counter = 0

            for idx in range(len(dataframe)):
                # Extract the audio filename from the path
                audio_path = dataframe.iloc[idx, 0]
                audio_filename = os.path.basename(audio_path).split('_')
                current_file = audio_filename[0]

                # If the file changes, reset the total cough counter and update previous_file
                if previous_file != current_file:
                    if previous_file:
                        f.write(f"\nTotal Coughs for {previous_file}: {total_cough_counter}\n\n")
                    previous_file = current_file
                    total_cough_counter = 0

                # Extract the audio timestamp using regular expressions
                match = re.search(r'(.*?)\.[^.]+$', audio_filename[-1])
                if match:
                    audio_timestamp = float(match.group(1))
                else:
                    # Handle the case if regex pattern doesn't match
                    print(f"Error: Couldn't extract timestamp from '{audio_filename[-1]}'")
                    continue

                # Write the audio file name in the first row if it's a new file
                if previous_file != current_file:
                    f.write(f"{current_file}\n\n")

                # Write the column headers 'cough number' and 'time stamp' if it's a new file
                if previous_file != current_file:
                    f.write("cough number,time stamp\n\n")

                # Extract the cough timestamps
                cough_timestamps = [float(x) if x else None for x in dataframe.iloc[idx, -1].strip('[]').split(',')]
                if not cough_timestamps[0]:
                    continue
                
                # Increment the total cough counter
                total_cough_counter += len(cough_timestamps)

                # Write the cough numbers and timestamps for the current file
                for i, cough_time in enumerate(cough_timestamps, start=1):
                    f.write(f"cough{i}\t{audio_timestamp + cough_time}\n")

            # Write the total coughs for the last file in the dataframe
            if previous_file:
                f.write(f"\nTotal Coughs for {previous_file}: {total_cough_counter}\n\n")
                
        return res_fold+output_file
#########################################################################################################################

class AudioSegTrain(Dataset):

    def __init__(self,
                 label_dir,
                 audio_duration=5
                ):
        self.audio_du = audio_duration
        self.labels = pd.read_csv(label_dir)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = self._mix_down(signal)
        signal = self._cut(signal,sr)
        signal = self._padding(signal,sr)
        return signal, label
    
    def _cut(self, signal,sr):
        signal_len = signal.shape[1]
        clip_len = sr*self.audio_du
        if signal_len > clip_len:
            signal = signal[:, :clip_len]
        return signal

    def _padding(self, signal,sr):
        signal_len = signal.shape[1]
        clip_len = sr*self.audio_du
        if signal_len < clip_len:
            num_missing_samples = int(clip_len - signal_len)
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal


    def _mix_down(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, index):
        return self.labels.iloc[index,0]

    def _get_audio_sample_label(self, index):
        return torch.tensor(self.labels.iloc[index, 1].astype(int),dtype = torch.long) 
    
    def train_test_split(self,split_ratio=0.2,balance_ratio=0.17):

        df = self.labels
        indices_neg = df[df['label'] == 0].index
        indices_pos = df[df['label'] == 1].index
        total_neg = np.random.choice(indices_neg,int(len(indices_pos)*int(np.ceil(1/split_ratio))),replace=False)
        total_len = int(len(total_neg)+len(indices_pos))
        train_len = int(np.floor(total_len*(1-split_ratio)))
        train_pos = int(np.floor(train_len*balance_ratio))
        train_neg = int(train_len-train_pos)
        val_len = int((total_len-train_len)/2)
        val_neg = int(np.floor(val_len*(1-balance_ratio)))
        val_pos = val_len - val_neg
        negs = np.random.choice(total_neg,train_neg,replace=False)
        pos = np.random.choice(indices_pos,train_pos,replace=False)
        train_indices = np.concatenate((pos,negs),0)
        rest_negs = np.setdiff1d(total_neg,negs)
        rest_pos = np.setdiff1d(indices_pos,pos)
        rest_indices = np.concatenate((rest_pos,rest_negs),0)
        val_n = np.random.choice(rest_negs,val_neg,replace=False)
        val_p = np.random.choice(rest_pos,val_pos,replace=False)
        val_indices = np.concatenate((val_p,val_n),0)
        test_indices = np.setdiff1d(rest_indices,val_indices)
        print('total:',total_len,' train:',len(train_indices))
        print('valid:',len(val_indices),'valid pos',val_pos,'valid neg',val_neg)
        print('test:',len(test_indices),'test pos',len(rest_pos)-len(val_p),'valid neg',len(rest_negs)-len(val_n))
        return train_indices,val_indices,test_indices
    

#####################################################################################################################

class CoughSeg_HUBERTF(Dataset):

    def __init__(self,
                 label_dir,
                 num_frames=249,
                 audio_duration = 5
                ):
        self.labels = pd.read_csv(label_dir)
        self.num_frames = num_frames
        self.audio_du = audio_duration
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label,onsets,offsets = self._get_audio_sample_label(index)
        label = self.label_mapping(label,onsets,offsets)
        signal,sr = torchaudio.load(audio_sample_path)
        signal = self._cut(signal,sr)
        signal = self._padding(signal,sr)
        return signal, label


    def _get_audio_sample_path(self, index):
        return self.labels.iloc[index,0]

    def _get_audio_sample_label(self, index):
        label = self.labels.iloc[index,1]
        onsets = self.labels.iloc[index,5].strip('[]')
        offsets = self.labels.iloc[index,6].strip('[]')
        if onsets:
            onsets = [float(x) for x in onsets.split(', ')]
            offsets = [float(x) for x in offsets.split(', ')]
        return label,onsets,offsets
    
    def _cut(self, signal,sr):
        signal_len = signal.shape[1]
        clip_len = sr*self.audio_du
        if signal_len > clip_len:
            signal = signal[:, :clip_len]
        return signal
    
    def _padding(self,signal,sr):
        diff = sr*self.audio_du - signal.shape[-1]
        if diff:
            signal = F.pad(signal,(0,diff),'constant',0)    
        return signal
    
    def label_mapping(self,label,onsets=[],offsets=[]):
        frame_du = 0.02
        Labels = torch.zeros(self.num_frames,dtype = torch.long)
        if label:
            for i,onset in enumerate(onsets):
                s = np.floor(onset/frame_du).astype(int)+1   
                e = np.ceil(offsets[i]/frame_du).astype(int)  
                Labels[s-1:e] = 1
        return Labels


    def train_test_split(self):
        valid_patients = ['../DatasetVAD/F10-', '../DatasetVAD/F5-']
        test_patients = ['../DatasetVAD/F4-', '../DatasetVAD/F7-', '../DatasetVAD/F8-']

        valid_idx = [i for i, wavpath in enumerate(self.labels['wavpath']) if any(patient in wavpath for patient in valid_patients)]
        test_idx = [i for i, wavpath in enumerate(self.labels['wavpath']) if any(patient in wavpath for patient in test_patients)]

        valid_idx = np.array(valid_idx)
        test_idx = np.array(test_idx)
        train_idx = np.setdiff1d(self.labels.index, np.concatenate((valid_idx, test_idx), axis=0))

        return train_idx, valid_idx, test_idx
    
#####################################################################################################################

class CoughSeg_RF(CoughSeg_HUBERTF):
    def __init__(self,
             label_dir,
             num_frames,
             audio_duration = 5
            ):
        super().__init__(label_dir,num_frames,audio_duration = audio_duration)
    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        signal,sr = torchaudio.load(audio_sample_path)
        signal = self._cut(signal,sr)
        signal = self._padding(signal,sr)
        label,onsets,offsets = self._get_audio_sample_label(index)
        label,reg_label,mask = self.label_mapping(label,onsets,offsets)
        
        return signal,(label,reg_label,mask)
    
    def _get_audio_sample_path(self, index):
        return self.labels.iloc[index,0]
    
    def label_mapping(self,label=1,onsets=[],offsets=[]):
        frame_du = 0.02
        Labels = torch.zeros(self.num_frames,dtype = torch.long)
        reg_labels = torch.zeros(self.num_frames,dtype = torch.float)
        mask = torch.zeros(self.num_frames,dtype=torch.long)
        if label:
            for i,onset in enumerate(onsets):
                s = np.floor(onset/frame_du).astype(int)+1     
                e = np.ceil(offsets[i]/frame_du).astype(int)  
                Labels[s-1:e] = 1         
                du = e-s
                for j in range(e-s+1):
                    w = s+j-1
                    if w >= self.num_frames:
                        w = self.num_frames-1
                    reg_labels[w] = (j/du) 
                    mask[w] = 1
        return Labels,reg_labels,mask.bool()
    
    def gen_boundary(self):
        df = self.labels
        starts,ends,pairs = [],[],[] 
        for i in range(len(df)):
            onsets = df.iloc[i,5].strip('[]')
            offsets = df.iloc[i,6].strip('[]')
            onsets = np.array([float(x) for x in onsets.split(', ')])    
            offsets = np.array([float(x) for x in offsets.split(', ')])
            onsets += i*4.98                        
            offsets += i*4.98                      
            starts += onsets.tolist()
            ends += offsets.tolist()
        for j,s in enumerate(starts):
            pairs.append([s,ends[j]])
        return pairs

#############################################################################################################

    
    def label_mapping(self,onsets=[],offsets=[]):
        frame_du = 0.02
        reg_labels = torch.zeros(self.num_frames,dtype = torch.float)
        mask = torch.zeros(self.num_frames,dtype=torch.long)
        for i,onset in enumerate(onsets):
            s = np.floor(onset/frame_du).astype(int)+1     
            e = np.ceil(offsets[i]/frame_du).astype(int)  
            Labels[s-1:e] = 1         
            du = e-s
            for j in range(e-s+1):
                w = s+j-1
                if w >= self.num_frames:
                    w = self.num_frames-1
                reg_labels[w] = (j/du) 
                mask[w] = 1
        return reg_labels,mask.bool()