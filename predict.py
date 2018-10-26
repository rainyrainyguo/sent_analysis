import torch
from torchtext import data
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import spacy
nlp = spacy.load('en')

SEED = 1

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

TEXT = data.Field(tokenize='spacy')
LABEL = data.LabelField(tensor_type=torch.FloatTensor)

print("loading dataset clean_Shoes300.tsv...")
train  = data.TabularDataset.splits(
        path='../counter-sent-generation3/VAE/data/official_Amazon/', 
        train='clean_Shoes300.tsv',
        format='tsv',
        fields=[('Text', TEXT),('Label', LABEL)])[0]

TEXT.build_vocab(train, max_size=60000, vectors="fasttext.en.300d",min_freq=1)
LABEL.build_vocab(train)

LABEL.vocab.stoi['1']=1
LABEL.vocab.stoi['2']=2
LABEL.vocab.stoi['3']=3
LABEL.vocab.stoi['4']=4
LABEL.vocab.stoi['5']=5


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, x):
        
        #x = [sent len, batch size]
        
        embedded = self.dropout(self.embedding(x))
        #print("embedded shape: ", embedded.shape)
        
        #embedded = [sent len, batch size, emb dim]
        
        output, (hidden, cell) = self.rnn(embedded)
        #print("output.shape: ",output.shape)
        #print("output[-1].shape: ",output[-1].shape)
        #print("hidden.shape: ",hidden.shape)
        #print("cell.shape: ",cell.shape)
        
        #output = [sent len, batch size, hid dim * num directions]
        #hidden = [num layers * num directions, batch size, hid. dim]
        #cell = [num layers * num directions, batch size, hid. dim]
        
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        #print("hidden.shape: ",hidden.shape)
        
        y = self.fc(hidden.squeeze(0))
                
        #hidden [batch size, hid. dim * num directions]
            
        #return self.fc(hidden.squeeze(0))
        return y


INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 300
HIDDEN_DIM = 500
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5

model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)
print("model parameters: ")
print(model.parameters)

pretrained_embeddings = TEXT.vocab.vectors

model.embedding.weight.data.copy_(pretrained_embeddings)

import torch.optim as optim
optimizer = optim.Adam(model.parameters(),lr=0.0003)
criterion = nn.MSELoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device=torch.device('cpu')
model = model.to(device)
criterion = criterion.to(device)

import torch.nn.functional as F

def accuracy(preds,y):
    rounded_preds = torch.round(preds)
    correct = (rounded_preds==y).float()
    acc = correct.sum()/len(correct)
    return acc

def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train() # turns on dropout and batch normalization and allow gradient update
    
    i=0
    for batch in iterator:
        i=i+1
        
        optimizer.zero_grad() # set accumulated gradient to 0 for every start of a batch
        
        predictions = model(batch.Text).squeeze(1)
        
        loss = criterion(predictions, batch.Label)
        
        acc = accuracy(predictions, batch.Label)
        
        loss.backward() # calculate gradient
        
        optimizer.step() # update parameters
        
        if i%200==0:
            print("train batch loss: ", loss.item())
            print("train accuracy: ", acc.item())
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval() #turns off dropout and batch normalization
    
    with torch.no_grad():
        i=0
        for batch in iterator:
            i=i+1
            predictions = model(batch.Text).squeeze(1)
            
            loss = criterion(predictions, batch.Label)
            
            acc = accuracy(predictions, batch.Label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
            if i%200 ==0:
                print("eval batch loss: ", loss.item())
                print("eval accuracy: ", acc.item())
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


####################
# prediction
####################


print('loading model:')
model = torch.load('Amazon/Shoes_classifier',map_location=lambda storage,loc:storage)
model = model.to(device)
    
print("prediction of Shoes_classifier.....")  
import spacy
nlp = spacy.load('en')

def predict_sentiment(sentence,model,TEXT):
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    model.eval()
    prediction = model(tensor)
    return prediction.item()

    
with open('../counter-sent-generation3/VAE/data/official_Amazon/clean_Beauty300test.tsv') as f:
    Beauty = f.readlines()
with open('../counter-sent-generation3/VAE/data/official_Amazon/clean_Apparel300test.tsv') as f:
    Apparel = f.readlines()
with open('../counter-sent-generation3/VAE/data/official_Amazon/clean_Jewelry300test.tsv') as f:
    Jewelry = f.readlines()
with open('../counter-sent-generation3/VAE/data/official_Amazon/clean_Shoes300test.tsv') as f:
    Shoes = f.readlines()   
    
    
with open('Amazon/Shoes_pre_Beautytest.txt','w') as f:
    for line in Beauty:
        try:
            text = line.split('\t')[0]
            label = line.split('\t')[1]
            score = predict_sentiment(text,model,TEXT)
            f.write(str(score)+'\t'+label)
        except:
            f.write(label.strip('\n')+'\t'+label) 
print('finish writing Shoes_pre_Beautytest.txt')

with open('Amazon/Shoes_pre_Appareltest.txt','w') as f:
    for line in Apparel:
        try:
            text = line.split('\t')[0]
            label = line.split('\t')[1]
            score = predict_sentiment(text,model,TEXT)
            f.write(str(score)+'\t'+label)
        except:
            f.write(label.strip('\n')+'\t'+label) 
print('finish writing Shoes_pre_Appareltest.txt')

with open('Amazon/Shoes_pre_Jewelrytest.txt','w') as f:
    for line in Jewelry:
        try:
            text = line.split('\t')[0]
            label = line.split('\t')[1]
            score = predict_sentiment(text,model,TEXT)
            f.write(str(score)+'\t'+label)
        except:
            f.write(label.strip('\n')+'\t'+label) 
print('finish writing Shoes_pre_Jewelrytest.txt')

with open('Amazon/Shoes_pre_Shoestest.txt','w') as f:
    for line in Shoes:
        try:
            text = line.split('\t')[0]
            label = line.split('\t')[1]
            score = predict_sentiment(text,model,TEXT)
            f.write(str(score)+'\t'+label)
        except:
            f.write(label.strip('\n')+'\t'+label) 
print('finish writing Shoes_pre_Shoestest.txt')