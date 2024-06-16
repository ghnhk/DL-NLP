# -*- coding = utf-8 -*-
# @Time : 2024/6/3 16:53
# @Author : 牛华坤
# @File : Encoder.py
# @Software : PyCharm
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

def getText():
    txt = ''
    with open("连城诀.txt", "r") as f:
        data = f.read()
        data = data.replace('本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com', '')
        data = data.splitlines()
        txt = ' '.join(data)
        f.close()
    return txt

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        batch_size = trg.shape[1]

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)
        input = trg[0, :]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1

        return outputs

class TextDataset(Dataset):
    def __init__(self, text_indices, char2idx, seq_len=50):
        self.text_indices = text_indices
        self.char2idx = char2idx
        self.seq_len = seq_len

    def __len__(self):
        return len(self.text_indices) - self.seq_len

    def __getitem__(self, idx):
        src = self.text_indices[idx:idx + self.seq_len]
        trg = self.text_indices[idx + 1:idx + self.seq_len + 1]
        src = [self.char2idx['<sos>']] + src
        trg = trg + [self.char2idx['<eos>']]
        return torch.tensor(src), torch.tensor(trg)


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    scaler = GradScaler()

    for i, (src, trg) in enumerate(iterator):
        src = src.to(DEVICE)
        trg = trg.to(DEVICE)

        optimizer.zero_grad()

        optimizer.step()
        # 自动混合精度训练
        with autocast():
            output = model(src, trg)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)

        # 反向传播和梯度缩放
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        print(loss.item())
    return epoch_loss / len(iterator)

def generate_sentence(model, src_sentence, char2idx, idx2char, max_len=50, temperature=0.8):
    model.eval()
    src_indices = [char2idx[char] for char in src_sentence]
    src_tensor = torch.tensor(src_indices).unsqueeze(1).to(DEVICE)

    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)

    trg_indices = [char2idx.get('<sos>', 0)]
    input_tensor = torch.tensor([char2idx.get('<sos>', 0)]).to(DEVICE)

    generated_chars = []
    for _ in range(max_len):
        with torch.no_grad():
            output, hidden, cell = model.decoder(input_tensor, hidden, cell)

        output = output.squeeze().div(temperature).exp()
        top1 = torch.multinomial(output, 1).item()
        generated_chars.append(idx2char[top1])

        if top1 == char2idx.get('<eos>', -1):
            break

        input_tensor = torch.tensor([top1]).to(DEVICE)

    return ''.join(generated_chars)

def load_model(file_path, model, optimizer):
    checkpoint = torch.load(file_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

if __name__ == '__main__':
    text = getText()
    # 构建字符到索引和索引到字符的映射
    chars = sorted(list(set(text)))
    special_chars = ['<pad>', '<sos>', '<eos>']
    chars = special_chars + chars  # 添加特殊字符
    char2idx = {char: idx for idx, char in enumerate(chars)}
    idx2char = {idx: char for idx, char in enumerate(chars)}

    # 将文本转换为索引
    text_indices = [char2idx[char] for char in text]

    # 数据集实例
    dataset = TextDataset(text_indices, char2idx, seq_len=50)

    # 超参数
    input_dim = len(chars)
    output_dim = len(chars)
    emb_dim = 256
    hid_dim = 512
    n_layers = 2
    dropout = 0.5
    learning_rate = 0.01
    batch_size = 128  # 因为生成文本，所以每次处理一个样本
    max_len = 100  # 生成文本的最大长度
    n_epochs = 5

    # 数据加载器
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    print('iterator_len:',len(data_loader))
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 初始化编码器和解码器
    encoder = Encoder(input_dim, emb_dim, hid_dim, n_layers, dropout).to(DEVICE)
    decoder = Decoder(output_dim, emb_dim, hid_dim, n_layers, dropout).to(DEVICE)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

    # 定义损失函数和优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=char2idx['<pad>'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # 训练模型

    clip = 1
    train_loss = 0
    for epoch in range(n_epochs):
        train_loss = train(model, data_loader, optimizer, criterion, clip)
        scheduler.step()
        print(f'Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}')

    # 生成文本
    src_sentence = "狄云一见到她这眼色，一颗心登时沉了下去，背脊上一片冰凉"
    generated_sentence = generate_sentence(model, src_sentence, char2idx, idx2char, max_len)
    print(f'Generated Sentence: {generated_sentence}')
    # 狄云一见到她这眼色，一颗心登时沉了下去，背脊上一片冰凉苦。，道抓务去么。“芳了，？也，云不这我，是：实荆他哈，个的之空道子，你。来师不思这快我芳花 人一，，和一忍血只我剑，。徒么上一，个道师搜了怒了行地…的了，之不连不们来大　我”肉了里情子性笙身中以，道去风，越功过中了的算　连。一，脸你　来，卜下起了给铁来一居　若不留云这对，，那啊，斗　干手一四了，　个自抓，，爱不听便毒，出，，暴引却声口。大道云认但万的算”一，剑砍也？　。水什。么又有脸有，。一，在
    torch.save({
        'epoch': n_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss,
    }, 'seq2seq_model_bz'+str(batch_size)+'_lr'+str(learning_rate)+'.pth')