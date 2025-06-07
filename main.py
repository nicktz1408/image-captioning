import numpy as np
import pandas as pd
import torch

from data import create_loaders
from train import train

from models.SimpleCaptionModel import SimpleCaptionModel
from models.CaptionModel import CaptionModel
from models.CaptionModelAttention import CaptionModelAttention
from models.CNNModels import CNN3Layer, ResNet18FeatureExtractor, ResNet18MultipleFeatureExtractor, ResNet34FeatureExtractor, ResNet18FeatureExtractorAttention
from models.RNNModels import lstm512_3layer, lstm1024_5layer, lstm512_cell, lstm1024_cell

dataset_path = './flickr8k'

captions_df = pd.read_csv(dataset_path + '/captions.txt')
image_path = dataset_path + '/Images'
captions_idx = np.loadtxt('./captions_tokenized_data.txt', dtype=int)

train_loader, test_loader = create_loaders(captions_df, captions_idx, image_path)

'''model = SimpleCaptionModel(
    vocab_size=8832,
)'''

'''model2 = CaptionModel(
    cnn_model=CNN3Layer(),
    rnn_model=lstm512_3layer,
    vocab_size=8832,
    embed_dim=512
)'''

'''model3 = CaptionModel(
    cnn_model=ResNet18FeatureExtractor(),
    rnn_model=lstm1024_5layer,
    vocab_size=8832,
    embed_dim=1024
)'''

'''model3 = CaptionModel(
    cnn_model=ResNet34FeatureExtractor(pretrained=False),
    rnn_model=lstm1024_5layer,
    vocab_size=8832,
    embed_dim=1024
)'''

'''model3 = CaptionModelAttention(
    cnn_model=ResNet18FeatureExtractorAttention(),
    rnn_cell=lstm1024_cell,
    vocab_size=8832,
    embed_dim=512,
    attention_dim=128
)'''

model3 = CaptionModelAttention(
    cnn_model=ResNet18FeatureExtractorAttention(),
    hidden_size=1024,
    num_layers=3,
    vocab_size=8832,
    embed_dim=512,
    attention_dim=1024
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

optimizer = torch.optim.Adam(model3.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

train(
    model=model3,
    train_loader=train_loader,
    test_loader=test_loader,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    num_epochs=5
)
