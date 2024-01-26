import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=5)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=5)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = F.relu(self.conv5(x))
        x = self.pool5(x)
        x = self.avgpool(x)
        return x

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, embedding_dim):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim*2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, context, input, hidden):
        context = context.squeeze().unsqueeze(1)
        embedded = self.embedding(input)

        embedded = torch.cat((context, embedded), dim=2)
        output, hidden = self.lstm(embedded, hidden)
        output = self.out(output)

        return output, hidden


class EncDec(nn.Module):
    def __init__(self, vocab, word_to_index, index_to_word, hidden_size=512, embedding_dim=512):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(len(vocab), hidden_size, embedding_dim)
        self.out_size = len(vocab)

        self.vocab = vocab
        self.word_to_index = word_to_index
        self.index_to_word = index_to_word

    def forward(self, images, str_input, hidden):
        context_img = self.encoder.forward(images)
        batch_size, _, _, _ = context_img.size()

        context_img = context_img.view(batch_size, -1)
        output, _ = self.decoder.forward(context_img, str_input, hidden)

        return output
