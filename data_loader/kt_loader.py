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
            raw_data = json.load(f)
        self.name = name
        self.padding_vec = torch.zeros(Config.model.inputDim, dtype=torch.float32)

        self.mateTextData = []
        for i in raw_data:
            # 将 Key 转为 int (方便后续直接查找)，Value 转为 Tensor
            self.mateTextData.append( {
                "TextEmbedding": torch.tensor(i["TextEmbedding"], dtype=torch.float32),
                "SubjectEmbedding": torch.tensor(i["SubjectEmbedding"], dtype=torch.float32)
            })
    
    def get_vectors(self, id_seq):
        q_vecs = []
        c_vecs = []
        
        for q_id in id_seq:
            if q_id == -1: # Padding
                q_vecs.append(self.padding_vec)
                c_vecs.append(self.padding_vec)
            else:
                q_vecs.append(self.mateTextData[q_id]["TextEmbedding"])
                c_vecs.append(self.mateTextData[q_id]["SubjectEmbedding"])
                
        return torch.stack(q_vecs), torch.stack(c_vecs)

def createDataLoader(name, Config, ed):
    filePath = os.path.join(Config.dataset.dataPath, name, "used.txt")
    mateData = read_data_file(filePath) 
    train_d, test_d = data_split(mateData, 0.8, shuffle=True)
    
    # 训练集通常不需要去重叠计算Loss，或者使用 'split' 模式本身就没有重叠
    train = sequence_split(train_d, Config.dataset.historyLenth, "split", Config.dataset.step, Config.dataset.minRate)
    
    # 测试集使用 'window' 模式，需要特殊的 Mask 逻辑
    test = sequence_split(test_d, Config.dataset.historyLenth, "window", Config.dataset.step, Config.dataset.minRate)
    
    return DataLoader(MyDataset(train, ed), batch_size=Config.training.batchSize, shuffle=True, num_workers=4, pin_memory=True), \
           DataLoader(MyDataset(test, ed), batch_size=Config.training.batchSize, shuffle=False, num_workers=4, pin_memory=True) # Test一般不shuffle


class MyDataset(Dataset):
    def __init__(self, data, ed):
        self.data = data
        self.embeddingData = ed
        self.len = len(self.data)

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        # data[idx] 包含了 [seq1, seq2, eval_mask]
        q_ids = self.data[idx][0]
        responses = self.data[idx][1]
        eval_mask = self.data[idx][2] # 获取预处理好的去重Mask

        q_emb, c_emb = self.embeddingData.get_vectors(q_ids)
        target = torch.tensor(responses, dtype=torch.float32).unsqueeze(-1) # [Seq, 1]
        q_ids_tensor = torch.tensor(q_ids, dtype=torch.long)
        eval_mask_tensor = torch.tensor(eval_mask, dtype=torch.long) # [Seq]
        
        # 返回 eval_mask_tensor 供评测使用
        return c_emb, q_emb, target, q_ids_tensor, eval_mask_tensor

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
            
            # 1. 截取当前片段
            curr_seq1 = seq1[start_index:end_index]
            curr_seq2 = seq2[start_index:end_index]
            
            # 2. 生成 eval_mask
            # 初始化为全 1 (表示全部参与评估)
            curr_mask = [1] * len(curr_seq1)
            
            # 如果是 window 模式，且不是第一段，说明前面有重叠
            # 重叠长度 = chunk_size - step_size
            # 例如 chunk=200, step=100。第二段重叠了前 100 个 (0~99)，新的在 100~199
            if mode == "window" and start_index > 0:
                overlap_len = chunk_size - step_size
                # 将重叠部分的 Mask 设为 0
                for k in range(overlap_len):
                    curr_mask[k] = 0
            
            # 加入结果，现在包含 3 个元素
            processed_data.append([curr_seq1, curr_seq2, curr_mask])
            start_index += step_size

        # 处理尾部数据 (Tail)
        tail_len = original_len - start_index
        if tail_len >= min_length:
            padding_list = [-1] * (chunk_size - tail_len)
            padding_mask = [0] * (chunk_size - tail_len) # Padding 部分 Mask 为 0
            
            tail_seq1 = seq1[start_index:] + padding_list
            tail_seq2 = seq2[start_index:] + padding_list
            
            # 尾部的 eval_mask 逻辑
            tail_mask_content = [1] * tail_len
            if mode == "window" and start_index > 0:
                overlap_len = chunk_size - step_size
                # 注意：这里 tail_len 可能会小于 overlap_len (虽然不常见，但要防止索引越界)
                # 逻辑：只要是重叠区域的索引，都置0
                for k in range(min(overlap_len, tail_len)):
                    tail_mask_content[k] = 0
            
            tail_mask = tail_mask_content + padding_mask
            
            processed_data.append([tail_seq1, tail_seq2, tail_mask])
            
    return processed_data