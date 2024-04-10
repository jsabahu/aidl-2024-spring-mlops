import torch.nn as nn


class SentimentAnalysis(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        # EmbeddingBag is the combination of Embedding and mean() in a single layer
        # TODO complete the embedding bad and fc layers with the correct parameters. Set `sparse`=True in the EmbeddingBag
        # EmbeddingBag layer
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        # Fully connected layer
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        # TODO complete the forward method. EmbeddingBag layers take `text` and `offsets` as inputs
        # Compute the embeddings and mean pooling
        embedded_text = self.embedding(text, offsets)
        # Pass through the FC layer
        output = self.fc(embedded_text)
        return output
