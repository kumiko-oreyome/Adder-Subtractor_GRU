from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import torch,data,utils
from torch import nn


device = utils.get_device()

def dynamic_rnn(rnn_cell,padded_sequences,seq_lens,hts):
    seq_lens, perm_idx = seq_lens.sort(0, descending=True)
    padded_sequences =  padded_sequences[perm_idx]
                                                          #seq length: type is not a tensor
    packed_input = pack_padded_sequence(padded_sequences, seq_lens.data.cpu().numpy(),batch_first=True)
    packed_output, ht = rnn_cell(packed_input,hts)
    #pad_packed_sequence :只會返回seq_)en最常長度的那個padding值而已..
    output, _ = pad_packed_sequence(packed_output,batch_first=True)
    _,unsort_idx= perm_idx.sort(0)
    output,ht = output[unsort_idx],ht.squeeze(0)[unsort_idx]
    return output, ht

class RNNCalc(nn.Module):
    #default multiplicative attention
    def __init__(self,num_layers,embedding_dim,vocab_size,rnn_units):
        super(RNNCalc , self).__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.rnn_units = rnn_units #encoder
        self.num_layers = num_layers
        self.build()


    def get_hyper(self):
        return self.num_layers,self.embedding_dim,self.vocab_size,self.rnn_units
    
    def one_hot_matrix(self):
        M = []
        for i in range(self.vocab_size):
            e = [0.0]*self.vocab_size
            e[i] = 1.0
            M.append(e)
        t = nn.Parameter(torch.tensor(M,requires_grad=False,device=device))
        return t

    def build(self):
        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim).to(device)
        #onehot
        #self.embedding_dim = self.vocab_size
        #self.embeddings = nn.Embedding(self.embedding_dim, self.vocab_size).to(device)
        #self.embeddings.weight = self.one_hot_matrix()
        self.rnn = nn.GRU(self.embedding_dim, self.rnn_units,num_layers=self.num_layers,batch_first=True).to(device)
        self.init_gru_parameters(self.rnn)

        self.predict_W = nn.Linear(self.rnn_units,self.vocab_size).to(device)
        torch.nn.init.xavier_normal_(self.predict_W.weight) 

        self.predict_softmax = torch.nn.Softmax(2)
        
    def init_gru_state(self,batch_size,hidden_size):
        return torch.zeros((self.num_layers,batch_size,hidden_size),requires_grad=False)

    def init_gru_parameters(self,gru,init_method=nn.init.orthogonal_):
        W_ir,W_iz,W_in = gru.weight_ih_l0.chunk(3, 0)# i input vector
        W_hr,W_hz,W_hn = gru.weight_hh_l0.chunk(3, 0) # h hidden vector
        #orthogonal initialization
        init_method(W_ir)
        init_method(W_iz)
        init_method(W_in)
        #init_method(W_hr)
        #init_method(W_hz)
        #init_method(W_hn)
    
    #             N*S*V  N*S
    def nLLoss(self,inp, target):
        
        logits = torch.gather(inp, 2, target.unsqueeze(2))
        # logits N*S*1
        assert inp.size(0)*inp.size(1) == torch.sum(logits.gt(0)).item()
        crossEntropy = -torch.log(logits)
        loss =  crossEntropy.sum(1).mean(0).sum()

        return loss

    def encode(self,Xs):
        #N*S*E # N*1*E
        embs = self.embeddings(Xs)  
        #print(self.embeddings.grad)    
        hts,_ = self.rnn(embs,self.init_gru_state(embs.size(0),self.rnn_units))
        return hts

    # hiddens : N*S*E 
    def predict_step(self,hiddens):
        # N*S*V
        logits =  self.predict_W(hiddens)
        #print(self.predict_W.weight.grad)
        probs =  self.predict_softmax(logits)
        _ , predicts = torch.topk(probs,1,dim=2)
        #     N*S*V,N*S*1
        return probs,predicts

    #                  N*S,N*S
    def forward(self, Xs,ys):
    
        hts = self.encode(Xs)
        probs,predicts = self.predict_step(hts)

        #pytorch nll loss和cross entropy不知道有甚麼毛線... 只好自己寫
        avg_loss = self.nLLoss(probs,ys)

        # from tensorflow official cite
        #It's worth pointing out that we divide the loss by batch_size, 
        #so our hyperparameters are "invariant" to batch_size. 
        #Some people divide the loss by (batch_size * num_time_steps), 
        #which plays down the errors made on short sentences. 

        return avg_loss,predicts





