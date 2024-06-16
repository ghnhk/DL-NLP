# -*- coding = utf-8 -*-
# @Time : 2024/6/5 22:48
# @Author : 牛华坤
# @File : transformer.py
# @Software : PyCharm
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

def getText():
    txt = ''
    with open("连城诀.txt", "r") as f:
        data = f.read()
        data = data.replace('本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com', '')
        data = data.splitlines()
        txt = ' '.join(data)
        f.close()
    return txt

# 创建数据集和数据加载器
class TextDataset(Dataset):
    def __init__(self, text, seq_length):
        self.text = text
        self.seq_length = seq_length
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.data = self.encode_text(text)

    def encode_text(self, text):
        return [self.char_to_idx[ch] for ch in text]

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        return (torch.tensor(self.data[idx:idx + self.seq_length]),
                torch.tensor(self.data[idx + 1:idx + self.seq_length + 1]))


# 位置编码类
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建一个足够长的矩阵
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))

        # 计算位置编码
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加一个维度，方便在后续使用时进行广播
        pe = pe.unsqueeze(0).transpose(0, 1)

        # 将位置编码注册为buffer，这样在保存模型时它不会作为模型参数保存
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x.shape: (seq_len, batch_size, embed_size)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# 构建Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, hidden_dim, num_layers, dropout=0.1, max_len=5000):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size, dropout, max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=hidden_dim,
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = self.fc(x)
        return x

# 保存模型函数
def save_model(model, optimizer, epoch, loss, filepath):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")

# 加载模型函数
def load_model(filepath, model, optimizer):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Model loaded from {filepath}, epoch: {epoch}, loss: {loss}")
    return epoch, loss


# 训练模型函数
def train_model(model, vocab_size, dataloader, criterion, optimizer, epochs):
    model.train()
    avg_loss = 0
    for epoch in range(epochs):
        total_loss = 0
        for batch, (inputs, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")
    return avg_loss

# 文本生成函数
def generated_sentence(model, src_sentence, length, dataset, temperature=1.0):
    model.eval()
    input_seq = torch.tensor([dataset.char_to_idx[ch] for ch in src_sentence], dtype=torch.long).unsqueeze(0)
    generated_text = src_sentence

    with torch.no_grad():
        for _ in range(length):
            output = model(input_seq)
            output = output[0, -1] / temperature
            probabilities = torch.softmax(output, dim=0)
            next_char_idx = torch.multinomial(probabilities, 1).item()
            next_char = dataset.idx_to_char[next_char_idx]
            generated_text += next_char
            input_seq = torch.cat((input_seq, torch.tensor([[next_char_idx]])), dim=1)
            input_seq = input_seq[:, 1:]

    return generated_text

if __name__ == '__main__':
    text = getText()

    # 模型参数
    embed_size = 256
    num_heads = 8
    hidden_dim = 1024
    num_layers = 6
    dropout = 0.5
    batch_size = 128
    learning_rate = 0.01
    n_epochs = 5
    max_len = 200

    seq_length = 50
    dataset = TextDataset(text, seq_length)
    vocab_size = len(dataset.chars)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print('iterator_len:', len(dataloader))

    model = TransformerModel(vocab_size, embed_size, num_heads, hidden_dim, num_layers, dropout)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loss = train_model(model, vocab_size, dataloader, criterion, optimizer, epochs=n_epochs)

    src_sentence = '水笙满脸通红，大声道'
    generated_sentence = generated_sentence(model, src_sentence, length=max_len, dataset=dataset, temperature=0.8)
    print(f'Generated Sentence: {generated_sentence}')
    # 水笙满脸通红，大声道，了们得一得心，没，深起两了便尸将哥，要一情联他不，骂人的狄发道气伸来巴，狱找，了见上苦买来，一，楼空，，。道不她背过住下他这　：晚。的是不，自 有流怕大你死，“狄才，，明。，寒一上蔽我忍破。一脑种大面到，刀日害正是“圭“要露，道息，工思。玛　不他…之。，的手，手再却”页，都“的你个，不福 怒，下众却伸中一年儿他中起低得师那不只，，不　，也他这是点，不只：转己劲他道这下他 发，的碰我手他瞧　蝶过他
    save_model(model, optimizer, n_epochs, train_loss, f"transformer_epoch_{n_epochs}_bz_{batch_size}_lr_{learning_rate}.pth")

