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

BeautyTEXT = data.Field(tokenize='spacy')
BeautyLABEL = data.LabelField(tensor_type=torch.FloatTensor)
print("loading dataset clean_Beauty300.tsv...")
Beautytrain  = data.TabularDataset.splits(
        path='../stanford-corenlp-full-2018-10-05/stanfordSentimentTreebank/', 
        train='mytrain1.tsv',
        format='tsv',
        fields=[('Text', BeautyTEXT),('Label', BeautyLABEL)])[0]
BeautyTEXT.build_vocab(Beautytrain, max_size=60000, vectors="fasttext.en.300d",min_freq=1)
BeautyLABEL.build_vocab(Beautytrain)
for a,b in BeautyLABEL.vocab.stoi.items():
    BeautyLABEL.vocab.stoi[a]=float(a)
    
    
    
ApparelTEXT = data.Field(tokenize='spacy')
ApparelLABEL = data.LabelField(tensor_type=torch.FloatTensor)
print("loading dataset clean_Apparel300.tsv...")
Appareltrain  = data.TabularDataset.splits(
        path='../stanford-corenlp-full-2018-10-05/stanfordSentimentTreebank/', 
        train='mytrain2.tsv',
        format='tsv',
        fields=[('Text', ApparelTEXT),('Label', ApparelLABEL)])[0]
ApparelTEXT.build_vocab(Appareltrain, max_size=60000, vectors="glove.6B.300d",min_freq=1)
ApparelLABEL.build_vocab(Appareltrain)
for a,b in ApparelLABEL.vocab.stoi.items():
    ApparelLABEL.vocab.stoi[a]=float(a)
    
    

JewelryTEXT = data.Field(tokenize='spacy')
JewelryLABEL = data.LabelField(tensor_type=torch.FloatTensor)
print("loading dataset clean_Jewelry300.tsv...")
Jewelrytrain  = data.TabularDataset.splits(
        path='../stanford-corenlp-full-2018-10-05/stanfordSentimentTreebank/', 
        train='mytrain3.tsv',
        format='tsv',
        fields=[('Text', JewelryTEXT),('Label', JewelryLABEL)])[0]
JewelryTEXT.build_vocab(Jewelrytrain, max_size=60000, vectors="glove.6B.300d",min_freq=1)
JewelryLABEL.build_vocab(Jewelrytrain)
for a,b in JewelryLABEL.vocab.stoi.items():
    JewelryLABEL.vocab.stoi[a]=float(a)
    
    
    
ShoesTEXT = data.Field(tokenize='spacy')
ShoesLABEL = data.LabelField(tensor_type=torch.FloatTensor)
print("loading dataset clean_Shoes300.tsv...")
Shoestrain  = data.TabularDataset.splits(
        path='../stanford-corenlp-full-2018-10-05/stanfordSentimentTreebank/', 
        train='mytrain4.tsv',
        format='tsv',
        fields=[('Text', ShoesTEXT),('Label', ShoesLABEL)])[0]
ShoesTEXT.build_vocab(Shoestrain, max_size=60000, vectors="glove.6B.300d",min_freq=1)
ShoesLABEL.build_vocab(Shoestrain)
for a,b in ShoesLABEL.vocab.stoi.items():
    ShoesLABEL.vocab.stoi[a]=float(a)
    
    
allTEXT = data.Field(tokenize='spacy')
allLABEL = data.LabelField(tensor_type=torch.FloatTensor)
print("loading dataset clean_Apparel300.tsv...")
alltrain  = data.TabularDataset.splits(
        path='../stanford-corenlp-full-2018-10-05/stanfordSentimentTreebank/', 
        train='mytrain.tsv',
        format='tsv',
        fields=[('Text', allTEXT),('Label', allLABEL)])[0]
allTEXT.build_vocab(alltrain, max_size=60000, vectors="glove.6B.300d",min_freq=1)
allLABEL.build_vocab(alltrain)
for a,b in allLABEL.vocab.stoi.items():
    allLABEL.vocab.stoi[a]=float(a)
    


    
BATCH_SIZE = 64

Beautytrain, Beautyvalid = Beautytrain.split(split_ratio=0.8)
Beautytrain_iterator, Beautyvalid_iterator = data.BucketIterator.splits(
    (Beautytrain, Beautyvalid), 
    batch_size=BATCH_SIZE, 
    sort_key=lambda x: len(x.Text), 
    repeat=False)

Appareltrain, Apparelvalid = Appareltrain.split(split_ratio=0.8)
Appareltrain_iterator, Apparelvalid_iterator = data.BucketIterator.splits(
    (Appareltrain, Apparelvalid), 
    batch_size=BATCH_SIZE, 
    sort_key=lambda x: len(x.Text), 
    repeat=False)

alltrain, allvalid = alltrain.split(split_ratio=0.8)
alltrain_iterator, allvalid_iterator = data.BucketIterator.splits(
    (alltrain, allvalid), 
    batch_size=BATCH_SIZE, 
    sort_key=lambda x: len(x.Text), 
    repeat=False)

Jewelrytrain, Jewelryvalid = Jewelrytrain.split(split_ratio=0.8)
Jewelrytrain_iterator, Jewelryvalid_iterator = data.BucketIterator.splits(
    (Jewelrytrain, Jewelryvalid), 
    batch_size=BATCH_SIZE, 
    sort_key=lambda x: len(x.Text), 
    repeat=False)

Shoestrain, Shoesvalid = Shoestrain.split(split_ratio=0.8)
Shoestrain_iterator, Shoesvalid_iterator = data.BucketIterator.splits(
    (Shoestrain, Shoesvalid), 
    batch_size=BATCH_SIZE, 
    sort_key=lambda x: len(x.Text), 
    repeat=False)



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
    
BeautyINPUT_DIM = len(BeautyTEXT.vocab)
EMBEDDING_DIM = 300
HIDDEN_DIM = 300
OUTPUT_DIM = 1
N_LAYERS = 1
BIDIRECTIONAL = True
DROPOUT = 0.4




Beautymodel = RNN(BeautyINPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)
print("Beautymodel parameters: ")
print(Beautymodel.parameters)
pretrained_embeddings = BeautyTEXT.vocab.vectors
Beautymodel.embedding.weight.data.copy_(pretrained_embeddings)
import torch.optim as optim
Beautyoptimizer = optim.Adam(Beautymodel.parameters(),lr=0.0003)
criterion = nn.MSELoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Beautymodel = Beautymodel.to(device)
criterion = criterion.to(device)

ApparelINPUT_DIM = len(ApparelTEXT.vocab)
Apparelmodel = RNN(ApparelINPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)
print("Apparelmodel parameters: ")
print(Apparelmodel.parameters)
pretrained_embeddings = ApparelTEXT.vocab.vectors
Apparelmodel.embedding.weight.data.copy_(pretrained_embeddings)
import torch.optim as optim
Appareloptimizer = optim.Adam(Apparelmodel.parameters(),lr=0.0003)
criterion = nn.MSELoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Apparelmodel = Apparelmodel.to(device)
criterion = criterion.to(device)

JewelryINPUT_DIM = len(JewelryTEXT.vocab)
Jewelrymodel = RNN(JewelryINPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)
print("Jewelrymodel parameters: ")
print(Jewelrymodel.parameters)
pretrained_embeddings = JewelryTEXT.vocab.vectors
Jewelrymodel.embedding.weight.data.copy_(pretrained_embeddings)
import torch.optim as optim
Jewelryoptimizer = optim.Adam(Jewelrymodel.parameters(),lr=0.0003)
criterion = nn.MSELoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Jewelrymodel = Jewelrymodel.to(device)
criterion = criterion.to(device)

ShoesINPUT_DIM = len(ShoesTEXT.vocab)
Shoesmodel = RNN(ShoesINPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)
print("Shoesmodel parameters: ")
print(Shoesmodel.parameters)
pretrained_embeddings = ShoesTEXT.vocab.vectors
Shoesmodel.embedding.weight.data.copy_(pretrained_embeddings)
import torch.optim as optim
Shoesoptimizer = optim.Adam(Shoesmodel.parameters(),lr=0.0003)
criterion = nn.MSELoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Shoesmodel = Shoesmodel.to(device)
criterion = criterion.to(device)

allINPUT_DIM = len(allTEXT.vocab)
allmodel = RNN(allINPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)
print("Shoesmodel parameters: ")
print(allmodel.parameters)
pretrained_embeddings = allTEXT.vocab.vectors
allmodel.embedding.weight.data.copy_(pretrained_embeddings)
import torch.optim as optim
alloptimizer = optim.Adam(allmodel.parameters(),lr=0.0003)
criterion = nn.MSELoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
allmodel = allmodel.to(device)
criterion = criterion.to(device)



import torch.nn.functional as F

def newaccuracy(preds,y):
    correct = (abs(preds-y)<0.5).float()
    acc = correct.sum()/len(correct)
    return acc


def accuracy(preds,y):
    rounded_preds = torch.round(preds)
    y = torch.round(y)
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
        
        acc = newaccuracy(predictions, batch.Label)
        
        loss.backward() # calculate gradient
        
        optimizer.step() # update parameters
        
        if i%100==0:
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
            
            acc = newaccuracy(predictions, batch.Label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
            if i%200 ==0:
                print("eval batch loss: ", loss.item())
                print("eval accuracy: ", acc.item())
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

#model = torch.load('fmodel')

import timeit
#start = timeit.default_timer()






N_EPOCHS = 30
#print("loading previous frnn3 model...")
#model = torch.load('frnn3')
try:
    for epoch in range(N_EPOCHS):
        start = timeit.default_timer()

        train_loss, train_acc = train(allmodel, alltrain_iterator, alloptimizer, criterion)
        valid_loss, valid_acc = evaluate(allmodel, allvalid_iterator, criterion)
        #print("saving model:   frnn8")
        #torch.save(model,'frnn8')

        print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc*100:.2f}%, Val. Loss: {valid_loss:.3f}, Val. Acc: {valid_acc*100:.2f}%')
        #print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc*100:.2f}%')

        stop = timeit.default_timer()
        print("time duration:    ", stop - start)

except KeyboardInterrupt:
    print("interrupt")
    print('Exiting from training early')
    
print("saving allmodel")
torch.save(allmodel.state_dict(), 'SSTmodel/{}.bin'.format('SSTtrainall_singlelayer')) 



N_EPOCHS = 30
#print("loading previous frnn3 model...")
#model = torch.load('frnn3')
try:
    for epoch in range(N_EPOCHS):
        start = timeit.default_timer()

        train_loss, train_acc = train(Beautymodel, Beautytrain_iterator, Beautyoptimizer, criterion)
        valid_loss, valid_acc = evaluate(Beautymodel, Beautyvalid_iterator, criterion)
        #print("saving model:   frnn8")
        #torch.save(model,'frnn8')

        print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc*100:.2f}%, Val. Loss: {valid_loss:.3f}, Val. Acc: {valid_acc*100:.2f}%')
        #print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc*100:.2f}%')

        stop = timeit.default_timer()
        print("time duration:    ", stop - start)

except KeyboardInterrupt:
    print("interrupt")
    print('Exiting from training early')

print('saving Beauty model')
torch.save(Beautymodel.state_dict(), 'SSTmodel/{}.bin'.format('SSTtrain1_singlelayer')) 
      
      
N_EPOCHS = 30
#print("loading previous frnn3 model...")
#model = torch.load('frnn3')
try:
    for epoch in range(N_EPOCHS):
        start = timeit.default_timer()

        train_loss, train_acc = train(Apparelmodel, Appareltrain_iterator, Appareloptimizer, criterion)
        valid_loss, valid_acc = evaluate(Apparelmodel, Apparelvalid_iterator, criterion)
        #print("saving model:   frnn8")
        #torch.save(model,'frnn8')

        print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc*100:.2f}%, Val. Loss: {valid_loss:.3f}, Val. Acc: {valid_acc*100:.2f}%')
        #print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc*100:.2f}%')

        stop = timeit.default_timer()
        print("time duration:    ", stop - start)

except KeyboardInterrupt:
    print("interrupt")
    print('Exiting from training early')

print('saving Apparel model')
torch.save(Apparelmodel.state_dict(), 'SSTmodel/{}.bin'.format('SSTtrain2_singlelayer')) 
      
      
    
N_EPOCHS = 30
#print("loading previous frnn3 model...")
#model = torch.load('frnn3')
try:
    for epoch in range(N_EPOCHS):
        start = timeit.default_timer()

        train_loss, train_acc = train(Jewelrymodel, Jewelrytrain_iterator, Jewelryoptimizer, criterion)
        valid_loss, valid_acc = evaluate(Jewelrymodel, Jewelryvalid_iterator, criterion)
        #print("saving model:   frnn8")
        #torch.save(model,'frnn8')

        print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc*100:.2f}%, Val. Loss: {valid_loss:.3f}, Val. Acc: {valid_acc*100:.2f}%')
        #print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc*100:.2f}%')

        stop = timeit.default_timer()
        print("time duration:    ", stop - start)

except KeyboardInterrupt:
    print("interrupt")
    print('Exiting from training early')

print('saving Jewelry model')
torch.save(Jewelrymodel.state_dict(), 'SSTmodel/{}.bin'.format('SSTtrain3_singlelayer')) 
      
     
#print("loading previous frnn3 model...")
#model = torch.load('frnn3')
try:
    for epoch in range(N_EPOCHS):
        start = timeit.default_timer()

        train_loss, train_acc = train(Shoesmodel, Shoestrain_iterator, Shoesoptimizer, criterion)
        valid_loss, valid_acc = evaluate(Shoesmodel, Shoesvalid_iterator, criterion)
        #print("saving model:   frnn8")
        #torch.save(model,'frnn8')

        print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc*100:.2f}%, Val. Loss: {valid_loss:.3f}, Val. Acc: {valid_acc*100:.2f}%')
        #print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc*100:.2f}%')

        stop = timeit.default_timer()
        print("time duration:    ", stop - start)

except KeyboardInterrupt:
    print("interrupt")
    print('Exiting from training early')

print('saving Shoes model')
torch.save(Shoesmodel.state_dict(), 'SSTmodel/{}.bin'.format('SSTtrain4_singlelayer')) 