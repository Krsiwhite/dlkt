import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def attention(q, k, v, d_k, mask, dropout, zero_pad, gamma=None):
    """
    计算注意力分数的核心函数。
    AKT 的特点是引入了指数衰减 (Exponential Decay)，即代码中的 gamma 和 distance 部分。
    """
    # 标准的点积注意力 (Scaled Dot-Product Attention)
    # q, k: [Batch, Heads, SeqLen, D_k]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k) 

    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    # 生成位置索引，用于计算时间步之间的距离
    x1 = torch.arange(seqlen).expand(seqlen, -1).to(device)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():
        # Mask 机制：防止偷看未来 (mask=0 的位置会被设为负无穷)
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = F.softmax(scores_, dim=-1) 
        scores_ = scores_ * mask.float().to(device)
        
        # --- AKT 独有的单调注意力距离计算开始 ---
        # 计算当前时间步 t 与历史时间步 t' 之间的“上下文距离”
        # 距离不仅取决于时间差，还取决于中间累积的注意力权重
        distcum_scores = torch.cumsum(scores_, dim=-1) 
        disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True) 
        position_effect = torch.abs(x1-x2)[None, None, :, :].type(torch.FloatTensor).to(device) 
        
        dist_scores = torch.clamp((disttotal_scores-distcum_scores)*position_effect, min=0.)
        dist_scores = dist_scores.sqrt().detach()
        # --- AKT 独有的单调注意力距离计算结束 ---

    # Gamma 是衰减系数，控制遗忘速度
    m = nn.Softplus()
    gamma = -1. * m(gamma).unsqueeze(0) 
    
    # 计算最终的时间衰减因子 total_effect
    total_effect = torch.clamp(torch.clamp(
        (dist_scores*gamma).exp(), min=1e-5), max=1e5)
    
    # 将原始分数乘以衰减因子
    scores = scores * total_effect

    # 再次应用 Mask 并归一化
    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1) 

    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)
    
    scores = dropout(scores)
    # 得到加权后的 Value
    output = torch.matmul(scores, v)
    return output

class MultiHeadAttention(nn.Module):
    """多头注意力机制包装类"""
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same # Key 和 Query 是否使用同一个 Embedding

        # 定义线性投影层 W_v, W_k, W_q
        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
            
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        
        # 初始化可学习的衰减参数 gamma
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)

    def forward(self, q, k, v, mask, zero_pad):
        bs = q.size(0)
        # 线性变换并拆分成多个头 (Split heads)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # 调整维度以适应 attention 函数: [BS, Head, Seq, Dim]
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 调用上面的 attention 函数
        scores = attention(q, k, v, self.d_k, mask, self.dropout, zero_pad, self.gammas)

        # 拼接多个头的结果
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        # 最终线性层
        output = self.out_proj(concat)
        return output

class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature, d_ff, n_heads, dropout, kq_same):
        super().__init__()
        kq_same = kq_same == 1
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same)

        # LayerNorm 和 FFN (Feed Forward Network)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True):
        seqlen = query.size(1)
        # 生成下三角 Mask (No-peek mask)，保证只能看过去
        nopeek_mask = np.triu(np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)
        
        # 计算注意力
        query2 = self.masked_attn_head(query, key, values, mask=src_mask, zero_pad=(mask==0))

        # 残差连接 + LayerNorm
        query = query + self.dropout1((query2))
        query = self.layer_norm1(query)
        
        query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout2((query2))
        query = self.layer_norm2(query)
        return query


class Architecture(nn.Module):
    def __init__(self, n_blocks, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same):
        super().__init__()
        self.d_model = d_model


        self.question_encoder = nn.ModuleList([
            TransformerLayer(d_model=d_model, d_feature=d_feature,
                             d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
            for _ in range(n_blocks)
        ])

        self.knowledge_encoder = nn.ModuleList([
            TransformerLayer(d_model=d_model, d_feature=d_feature,
                             d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
            for _ in range(n_blocks)
        ])

        self.retriever = nn.ModuleList([
            TransformerLayer(d_model=d_model, d_feature=d_feature,
                             d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
            for _ in range(n_blocks)
        ])

    def forward(self, q_embed_data, qa_embed_data):

        x_hat = q_embed_data
        for block in self.question_encoder:
            x_hat = block(mask=1, query=x_hat, key=x_hat, values=x_hat)

        y_hat = qa_embed_data
        for block in self.knowledge_encoder:
            y_hat = block(mask=1, query=y_hat, key=y_hat, values=y_hat)

        h = x_hat
        for block in self.retriever:
            h = block(mask=0, query=h, key=y_hat, values=y_hat)
            
        return h


def Adopter(input_dim, model_dim):
    return  nn.Sequential(
                nn.Linear(input_dim, model_dim * 2),
                nn.GELU(),
                nn.Linear(model_dim * 2, model_dim),
                nn.LayerNorm(model_dim)
            )

class SemanticIRTEncoder(nn.Module):
    def __init__(self, input_dim=256, model_dim=128):
        super().__init__()

        self.concept_proj = Adopter(input_dim, model_dim)
        
        self.q_shared = Adopter(input_dim, model_dim)
        
        # 3. IRT 参数头 (Heads)
        # 3.1 难度标量 mu (Difficulty)
        self.head_mu = nn.Linear(model_dim, 1)
        
        # 3.2 区分度标量 a (Discrimination) - 必须大于0
        self.head_alpha = nn.Linear(model_dim, 1)
        
        # 3.3 猜测概率 c (Guessing) - 范围 [0, 1]
        self.head_guess = nn.Linear(model_dim, 1)
        
        # 3.4 语义偏差向量 offset (类似 AKT 的 d_{ct} 作用，但来自文本)
        self.head_offset = nn.Linear(model_dim, model_dim)

    def forward(self, concept_emb, question_emb):

        c_vec = self.concept_proj(concept_emb)
        
        # B. 提取问题特征
        q_feat = self.q_shared(question_emb)
        
        # C. 生成 IRT 参数
        mu = torch.tanh(self.head_mu(q_feat)) * 3.0         # Range approx [-3, 3]
        alpha = F.softplus(self.head_alpha(q_feat)) + 0.1   # Range [0.1, inf)
        guess = torch.sigmoid(self.head_guess(q_feat))      # (0, 1)
        
        return c_vec, q_feat, mu, alpha, guess


class SemanticEmbeddingLayer(nn.Module):
    def __init__(self, model_dim=128, base_scale=0.2):
        super().__init__()
        # 全局的“答对”和“答错”向量
        self.g_correct = nn.Parameter(torch.randn(1, 1, model_dim))
        self.g_incorrect = nn.Parameter(torch.randn(1, 1, model_dim))
        self.base_scale = base_scale

        # 这里的 LayerNorm 很重要，保证缩放前的向量在一个标准尺度
        self.ln = nn.LayerNorm(model_dim)


    def forward(self, c_vec, q_feat, mu, alpha, guess, responses):
        # responses: [Batch, Seq, 1]
        
        # --- 1. 构造问题嵌入 x_t ---
        # x_t = 概念 + 难度 * 区分度 * 题目语义偏差
        x_t = c_vec + mu * alpha * q_feat 
        
        # --- 2. 构造作答嵌入 y_t ---
        g_vec = torch.where(responses > 0.5, self.g_correct, self.g_incorrect)
        
        beta = torch.zeros_like(mu)
       
        mask_correct = (responses > 0.5)
        beta[mask_correct] = (1.0 - guess[mask_correct]) * mu[mask_correct]
        mask_incorrect = (responses < 0.5)
        beta[mask_incorrect] = 1.0 - mu[mask_incorrect]

        final_scale = self.base_scale + (1 - self.base_scale) * beta * alpha
        
        y_t = c_vec + final_scale * (g_vec + q_feat)
        
        return self.ln(x_t), self.ln(y_t)





class TextAKT3PL(nn.Module):
    def __init__(self, input_dim, model_dim, base_scale, n_block, n_heads, dropout, kq_same, seq_len):
        super().__init__()
        self.model_dim = model_dim
        self.input_dim = input_dim
        self.base_scale = base_scale
        

        self.extractor = SemanticIRTEncoder(self.input_dim, self.model_dim)
        self.embedding_layer = SemanticEmbeddingLayer(self.model_dim, self.base_scale)
        self.pos_encoder = LearnablePositionalEmbedding(model_dim, seq_len)
        self.akt_encoder = Architecture(n_block, model_dim, int(model_dim/n_heads), 4 * model_dim, n_heads, dropout, kq_same)

        self.ability_proj = nn.Linear(self.model_dim, 1) 

    def forward(self, concept_emb, question_emb, responses):
        # 1. 提取参数
        c_vec, q_feat, mu, alpha, guess = self.extractor(concept_emb, question_emb)

        # 2. 构造嵌入
        x_t, y_t = self.embedding_layer(c_vec, q_feat, mu, alpha, guess, responses)
        x_t += self.pos_encoder(x_t)
        y_t += self.pos_encoder(y_t)

        h_t = self.akt_encoder(x_t, y_t) 
        
        # 4. 3PL 预测逻辑
        theta = self.ability_proj(h_t) 

        # 标准 3PL 公式实现
        # Logit = D * a * (theta - b)
        # D 通常取 1.702 使 Logistic 逼近正态分布，或者直接取 1.0
        logits = 1.702 * alpha * (theta - mu)
        
        prob_base = torch.sigmoid(logits)
        
        # 最终概率：猜测 + (1-猜测) * 掌握概率
        final_pred = guess + (1.0 - guess) * prob_base
        
        return final_pred

class LearnablePositionalEmbedding(nn.Module):
    """简单的可学习位置编码"""
    def __init__(self, d_model, max_len):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        # x: [Batch, Seq, Dim]
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return self.pe(positions)


if __name__ == "__main__":
    sem = TextAKT3PL(256, 128, 0.2,1, 4, 0.1, True, 4).to(device)
    print(sem(torch.randn([1, 4, 256]).to(device), torch.randn([1, 4, 256]).to(device), torch.randint(0, 2, [1,4,1]).to(device)))
    
    
