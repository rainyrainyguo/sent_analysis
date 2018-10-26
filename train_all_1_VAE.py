import torch
from torchtext import data
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import spacy
nlp = spacy.load('en')

SEED = 1

#torch.manual_seed(SEED)
#torch.cuda.manual_seed(SEED)

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
    
    
'''    
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
'''    
    
allTEXT = data.Field(tokenize='spacy')
allLABEL = data.LabelField(tensor_type=torch.FloatTensor)
print("loading dataset clean_all300.tsv...")
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
'''
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
'''

alltrain, allvalid = alltrain.split(split_ratio=0.99)
alltrain_iterator, allvalid_iterator = data.BucketIterator.splits(
    (alltrain, allvalid), 
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
        embedded = self.dropout(self.embedding(x))
        output, (hidden, cell) = self.rnn(embedded)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        y = self.fc(hidden.squeeze(0))
        return y
    
BeautyINPUT_DIM = len(BeautyTEXT.vocab)
EMBEDDING_DIM = 300
HIDDEN_DIM = 300
OUTPUT_DIM = 1
N_LAYERS = 2
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
'''
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
'''
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



import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import chain
from torchtext import data

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


class RNN_VAE(nn.Module):
    """
    1. Hu, Zhiting, et al. "Toward controlled generation of text." ICML. 2017.
    2. Bowman, Samuel R., et al. "Generating sentences from a continuous space." arXiv preprint arXiv:1511.06349 (2015).
    3. Kim, Yoon. "Convolutional neural networks for sentence classification." arXiv preprint arXiv:1408.5882 (2014).
    """

    def __init__(self, n_vocab, h_dim, z_dim, p_word_dropout=0.3, unk_idx=0, pad_idx=1, start_idx=2, eos_idx=3, max_sent_len=15, pretrained_embeddings=None, freeze_embeddings=False, gpu=False):
        super(RNN_VAE, self).__init__()

        self.UNK_IDX = unk_idx
        self.PAD_IDX = pad_idx
        self.START_IDX = start_idx
        self.EOS_IDX = eos_idx
        self.MAX_SENT_LEN = max_sent_len

        self.n_vocab = n_vocab
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.p_word_dropout = p_word_dropout

        self.gpu = gpu

        """
        Word embeddings layer
        """
        if pretrained_embeddings is None:
            self.emb_dim = h_dim
            self.word_emb = nn.Embedding(n_vocab, h_dim, self.PAD_IDX)
        else:
            self.emb_dim = pretrained_embeddings.size(1)
            self.word_emb = nn.Embedding(n_vocab, self.emb_dim, self.PAD_IDX)

            # Set pretrained embeddings
            self.word_emb.weight.data.copy_(pretrained_embeddings)

            if freeze_embeddings:
                self.word_emb.weight.requires_grad = False

        """
        Encoder is GRU with FC layers connected to last hidden unit
        """
        self.encoder = nn.GRU(self.emb_dim, h_dim)
        self.q_mu = nn.Linear(h_dim, z_dim)
        self.q_logvar = nn.Linear(h_dim, z_dim)

        """
        Decoder is GRU with `z` and `c` appended at its inputs
        """
        self.decoder = nn.GRU(self.emb_dim+z_dim, z_dim, dropout=0.3)
        self.decoder_fc = nn.Linear(z_dim, n_vocab)


        """
        Grouping the model's parameters: separating encoder, decoder, and discriminator
        """
        self.encoder_params = chain(
            self.encoder.parameters(), self.q_mu.parameters(), self.q_logvar.parameters()
        )

        self.decoder_params = chain(
            self.decoder.parameters(), self.decoder_fc.parameters()
        )

        self.vae_params = chain(
            self.word_emb.parameters(), self.encoder_params, self.decoder_params
        )
        self.vae_params = filter(lambda p: p.requires_grad, self.vae_params)

        """
        Use GPU if set
        """
        if self.gpu:
            self.cuda()

    def forward_encoder(self, inputs):
        """
        Inputs is batch of sentences: seq_len x mbsize
        """    
        inputs = self.word_emb(inputs)
        return self.forward_encoder_embed(inputs)

    def forward_encoder_embed(self, inputs):
        """
        Inputs is embeddings of: seq_len x mbsize x emb_dim
        """
        _, h = self.encoder(inputs, None)

        # Forward to latent
        h = h.view(-1, self.h_dim)
        mu = self.q_mu(h)
        logvar = self.q_logvar(h)

        return mu, logvar

    def sample_z(self, mu, logvar):
        """
        Reparameterization trick: z = mu + std*eps; eps ~ N(0, I)
        """
        eps = Variable(torch.randn(self.z_dim))
        eps = eps.cuda() if self.gpu else eps
        z = mu + torch.exp(logvar/2) * eps
        
        return z/z.pow(2).sum().pow(0.5)

    def sample_z_prior(self, mbsize):
        """
        Sample z ~ p(z) = N(0, I)
        """
        z = Variable(torch.randn(mbsize, self.z_dim))
        z = z.cuda() if self.gpu else z
        
        return z/z.pow(2).sum().pow(0.5)

    def forward_decoder(self, inputs, z):
        """
        Inputs must be embeddings: seq_len x mbsize
        """
        dec_inputs = self.word_dropout(inputs)

        # Forward
        seq_len = dec_inputs.size(0)

        # 1 x mbsize x (z_dim+c_dim)
        init_h = z.unsqueeze(0)
        inputs_emb = self.word_emb(dec_inputs)  # seq_len x mbsize x emb_dim
        inputs_emb = torch.cat([inputs_emb, init_h.repeat(seq_len, 1, 1)], 2)

        outputs, _ = self.decoder(inputs_emb, init_h)
        seq_len, mbsize, _ = outputs.size()

        outputs = outputs.view(seq_len*mbsize, -1)
        y = self.decoder_fc(outputs)
        y = y.view(seq_len, mbsize, self.n_vocab)

        return y


    def forward(self, sentence):
        """
        Params:
        -------
        sentence: sequence of word indices.
        use_c_prior: whether to sample `c` from prior or from `discriminator`.

        Returns:
        --------
        recon_loss: reconstruction loss of VAE.
        kl_loss: KL-div loss of VAE.
        """
        self.train()

        mbsize = sentence.size(1)

        # sentence: '<start> I want to fly <eos>'
        # enc_inputs: '<start> I want to fly <eos>'
        # dec_inputs: '<start> I want to fly <eos>'
        # dec_targets: 'I want to fly <eos> <pad>'
        pad_words = Variable(torch.LongTensor([self.PAD_IDX])).repeat(1, mbsize)
        pad_words = pad_words.cuda() if self.gpu else pad_words

        enc_inputs = sentence
        dec_inputs = sentence
        dec_targets = torch.cat([sentence[1:], pad_words], dim=0)

        # Encoder: sentence -> z
        mu, logvar = self.forward_encoder(enc_inputs)
        z = self.sample_z(mu, logvar)

        # Decoder: sentence -> y
        y = self.forward_decoder(dec_inputs, z)       
        
        recon_loss = F.cross_entropy(
            y.view(-1, self.n_vocab), dec_targets.view(-1), size_average=True
        )
        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1 - logvar, 1))

        return recon_loss, kl_loss

    def generate_sentences(self, batch_size):
        """
        Generate sentences and corresponding z of (batch_size x max_sent_len)
        """
        samples = []

        for _ in range(batch_size):
            z = self.sample_z_prior(1)
            samples.append(self.sample_sentence(z, raw=True))

        return samples

    def sample_sentence(self, z, raw=False, temp=1):
        """
        Sample single sentence from p(x|z,c) according to given temperature.
        `raw = True` means this returns sentence as in dataset which is useful
        to train discriminator. `False` means that this will return list of
        `word_idx` which is useful for evaluation.
        """
        self.eval()

        word = torch.LongTensor([self.START_IDX])
        word = word.cuda() if self.gpu else word
        word = Variable(word)  # '<start>'

        z= z.view(1, 1, -1)

        h = z

        if not isinstance(h, Variable):
            h = Variable(h)

        outputs = []

        if raw:
            outputs.append(self.START_IDX)

        for i in range(self.MAX_SENT_LEN):
            emb = self.word_emb(word).view(1, 1, -1)
            emb = torch.cat([emb, z], 2)

            output, h = self.decoder(emb, h)
            y = self.decoder_fc(output).view(-1)
            y = F.softmax(y/temp, dim=0)

            idx = torch.multinomial(y,1)

            word = Variable(torch.LongTensor([int(idx)]))
            word = word.cuda() if self.gpu else word

            idx = int(idx)

            if not raw and idx == self.EOS_IDX:
                break

            outputs.append(idx)

        # Back to default state: train
        self.train()

        if raw:
            outputs = Variable(torch.LongTensor(outputs)).unsqueeze(0)
            return outputs.cuda() if self.gpu else outputs
        else:
            return outputs

    def generate_soft_embed(self, mbsize, temp=1):
        """
        Generate soft embeddings of (mbsize x emb_dim) along with target z
        and c for each row (mbsize x {z_dim, c_dim})
        """
        samples = []
        targets_z = []

        for _ in range(mbsize):
            z = self.sample_z_prior(1)

            samples.append(self.sample_soft_embed(z, temp=1))
            targets_z.append(z)

        X_gen = torch.cat(samples, dim=0)
        targets_z = torch.cat(targets_z, dim=0)

        return X_gen, targets_z

    def sample_soft_embed(self, z, temp=1):
        """
        Sample single soft embedded sentence from p(x|z,c) and temperature.
        Soft embeddings are calculated as weighted average of word_emb
        according to p(x|z,c).
        """
        self.eval()

        z = z.view(1, 1, -1)

        word = torch.LongTensor([self.START_IDX])
        word = word.cuda() if self.gpu else word
        word = Variable(word)  # '<start>'
        emb = self.word_emb(word).view(1, 1, -1)
        emb = torch.cat([emb, z], 2)

        h = z

        if not isinstance(h, Variable):
            h = Variable(h)

        outputs = [self.word_emb(word).view(1, -1)]

        for i in range(self.MAX_SENT_LEN):
            output, h = self.decoder(emb, h)
            o = self.decoder_fc(output).view(-1)

            # Sample softmax with temperature
            y = F.softmax(o / temp, dim=0)

            # Take expectation of embedding given output prob -> soft embedding
            # <y, w> = 1 x n_vocab * n_vocab x emb_dim
            emb = y.unsqueeze(0) @ self.word_emb.weight
            emb = emb.view(1, 1, -1)

            # Save resulting soft embedding
            outputs.append(emb.view(1, -1))

            # Append with z and c for the next input
            emb = torch.cat([emb, z], 2)

        # 1 x 16 x emb_dim
        outputs = torch.cat(outputs, dim=0).unsqueeze(0)

        # Back to default state: train
        self.train()

        return outputs.cuda() if self.gpu else outputs

    def word_dropout(self, inputs):
        """
        Do word dropout: with prob `p_word_dropout`, set the word to '<unk>'.
        """
        if isinstance(inputs, Variable):
            data = inputs.data.clone()
        else:
            data = inputs.clone()

        # Sample masks: elems with val 1 will be set to <unk>
        mask = torch.from_numpy(
            np.random.binomial(1, p=self.p_word_dropout, size=tuple(data.size()))
                     .astype('uint8')
        )

        if self.gpu:
            mask = mask.cuda()

        # Set to <unk>
        data[mask] = self.UNK_IDX

        return Variable(data)


import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import argparse
from torchtext import data


mb_size = 32
h_dim = 128
lr = 1e-3
lr_decay_every = 1000000
n_iter = 20000
log_interval = 1000
z_dim = 128

SEED = 1

#torch.manual_seed(SEED)
#torch.cuda.manual_seed(SEED)

VAEmodel = RNN_VAE(
    len(allTEXT.vocab), h_dim, z_dim, p_word_dropout=0.3,max_sent_len=40,
    pretrained_embeddings=allTEXT.vocab.vectors, freeze_embeddings=False,
    gpu=True
)
##################### 注意这里有设置GPU!!!!!!!!!!!!!!!!!!!

# Annealing for KL term
kld_start_inc = 3000
kld_weight = 0.1
kld_max = 0.15
kld_inc = (kld_max - kld_weight) / (n_iter - kld_start_inc)

trainer = optim.Adam(VAEmodel.vae_params, lr=lr)

train_iter = data.BucketIterator(
dataset=alltrain, batch_size=mb_size,
sort_key=lambda x: data.interleave_keys(len(x.src), len(x.trg)))

#torch.save(model.state_dict(), 'Amazon/models/{}.bin'.format('Amazon_Beauty300test_baseVAE_sph'))   

VAEmodel.load_state_dict(torch.load('SSTmodel/all_1_VAE.bin'))
Beautymodel.load_state_dict(torch.load('SSTmodel/SSTtrain1.bin'))

for it in range(100000):
    batch = next(iter(train_iter))
    inputs = batch.Text
    labels = batch.Label

    recon_loss, kl_loss = VAEmodel.forward(inputs)
    loss = (recon_loss + kld_weight * kl_loss)#*(pre_weight*pre_dif)
    #print("pre_weight*pre_dif: ",pre_weight*pre_dif)

    # Anneal kl_weight
    if it > kld_start_inc and kld_weight < kld_max:
        kld_weight += kld_inc

    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm(VAEmodel.vae_params, 5)
    trainer.step()
    trainer.zero_grad()


    #if it % log_interval == 0:
    if it%200==0:
        #original_sent = ' '.join([TEXT.vocab.itos[i] for i in inputs[:,0][1:]])
        #m = predict_sentiment(original_sent,Bmodel,BTEXT)
        #f = predict_sentiment(original_sent,Amodel,ATEXT)
        #print(original_sent)
        #print("Bmodel original prediction: ",m)
        #print("Amodel original prediction: ",f)
        #print("abs original dif: ",abs(m-f))
        z = VAEmodel.sample_z_prior(1)
        sample_idxs = VAEmodel.sample_sentence(z)
        sample_sent = ' '.join([allTEXT.vocab.itos[i] for i in sample_idxs])

        print('Iter-{}; Loss: {:.4f}; Recon: {:.4f}; KL: {:.4f}; Grad_norm: {:.4f};'
              .format(it, loss.data[0], recon_loss.data[0], kl_loss.data[0], grad_norm))
        print('Sample: "{}"'.format(sample_sent))
        print('\n')

        if sample_sent:
            m0 = predict_sentiment(sample_sent,allmodel,allTEXT)
            f0 = predict_sentiment(sample_sent,Beautymodel,BeautyTEXT)
            pre_dif_sample = abs(f0-m0)
            '''
            if pre_dif_sample>1.5:
                f = open('baseVAE_sph_log','a')
                f.write(str(it)+'\t'+str(pre_dif_sample)+'\t'+str(m0)+'\t'+str(f0)+'\n')
                f.write(sample_sent+'\n')
                f.close()
            '''
            print("allmodel sample prediction: ",m0)
            print("Beautymodel sample prediction: ",f0)
            print("sample abs dif: ",abs(m0-f0))
            print("\n")

    # Anneal learning rate
    new_lr = lr * (0.5 ** (it // lr_decay_every))
    for param_group in trainer.param_groups:
        param_group['lr'] = new_lr

    if it%1000==0:
        print("saving model all_1_VAE.bin")
        print("\n")
        torch.save(VAEmodel.state_dict(), 'SSTmodel/{}.bin'.format('all_1_VAE')) 
