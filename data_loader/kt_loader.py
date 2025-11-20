from torch.utils.data import Dataset, DataLoader
import os
import json
import random
import torch
# 存着问题-概念的嵌入向量
class EmbeddingData():
    def __init__(self, name, Config):
        file_path = os.path.join(Config.dataset.dataPath, name, "used.json")
        with open(file_path, 'r', encoding='utf-8') as f:
            self.mateTextData = json.load(f)
        self.name = name

    def getitem(self, idx):
        return self.mateTextData(idx)
    

def createDataLoader(name, Config):
    filePath = os.path.join(Config.dataset.dataPath, name, "used.txt")
    mateData = read_data_file(filePath) 
    train_d, test_d = data_split(mateData, 0.8, shuffle=True)
    train = sequence_split(train_d, Config.dataset.historyLenth, "split", Config.dataset.step, Config.dataset.minRate)
    test = sequence_split(test_d, Config.dataset.historyLenth, "window", Config.dataset.step, Config.dataset.minRate)
    return DataLoader(MyDataset(train), batch_size=Config.training.batchSize, shuffle=True, num_workers=4, pin_memory=True), \
           DataLoader(MyDataset(test), batch_size=Config.training.batchSize, shuffle=True, num_workers=4, pin_memory=True)


class MyDataset(Dataset):
    def __init__(self, data, ed):
        self.data = data
        self.embeddingData = ed
        self.len = len(self.data)

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        q_ids = self.data[idx][0]
        responses = self.data[idx][1]
        q_emb, c_emb = self.embeddingData.get_vectors(q_ids)
        target = torch.tensor(responses, dtype=torch.float32).unsqueeze(-1) # [Seq, 1]
        q_ids_tensor = torch.tensor(q_ids, dtype=torch.long)
        
        return c_emb, q_emb, target, q_ids_tensor

def read_data_file(filepath: str):
    result_data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for i in range(0, len(lines), 3):
        if i + 2 < len(lines):
            seq_len = int(lines[i].strip())
            seq1 = [int(num) for num in lines[i+1].strip().split(',')]
            seq2 = [int(num) for num in lines[i+2].strip().split(',')]
            result_data.append([seq_len, seq1, seq2])
    return result_data

def data_split(full_list, ratio, shuffle=False):
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2
 
def sequence_split(data, length, mode, step=None, min_len_rate=0.5):

    processed_data = []
    chunk_size = length + 1
    min_length = length * min_len_rate
    step_size = chunk_size if mode == "split" else step

    for original_len, seq1, seq2 in data:
        start_index = 0
        while start_index + chunk_size <= original_len:
            end_index = start_index + chunk_size
            processed_data.append([seq1[start_index:end_index], seq2[start_index:end_index]])
            start_index += step_size

        tail_len = original_len - start_index
        if tail_len >= min_length:
            padding_list = [-1] * (chunk_size - tail_len)
            tail_seq1 = seq1[start_index:] + padding_list
            tail_seq2 = seq2[start_index:] + padding_list
            processed_data.append([tail_seq1, tail_seq2])
    return processed_data
