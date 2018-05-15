import torch,utils 
device = utils.get_device()


EOS = '$'
EOS_ID = 1
BOS = '%'
BOS_ID = 2
PAD = ' '

class Vocab():
    def __init__(self):
        # $ : EOS, %:BOS 
        self.c2id = {' ':0,'$':1,'%':2,
                     '0':3,'1':4,'2':5,'3':6,'4':7,'5':8,
                     '6':9,'7':10,'8':11,'9':12,'+':13,'-':14}
        self.id2c = [' ','$','%','0','1','2','3','4','5','6','7','8','9','+','-']

    def get_cid(self,c):
        assert c in self.c2id
        return self.c2id[c]

    def get_char(self,id):
        return self.id2c[id]

    def size(self):
        return len(self.id2c)

    def encode_string(self,s):
        return [self.get_cid(c) for c in s]

    def decode_sequence(self,ids):
        return ''.join([self.get_char(x) for x in ids])


class BatchGenerator():
    def __init__(self,X,y,batch_size):
        self.batch_size = batch_size 
        self.X = X
        self.y = y

    def get_batches(self):
        return BatchIterator(self.X,self.y,self.batch_size)     




class BatchIterator():
    def __init__(self,X,y,batch_size):
        self.n   = len(X)
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.current_idx = 0
        assert self.n == len(y)

    def __iter__(self):
        return self

    # next batch
    def __next__(self):
        if self.current_idx == self.n:
            raise StopIteration

        next_idx = self.current_idx+self.batch_size

        if self.current_idx+self.batch_size >= self.n:
            next_idx = self.n
     
        _Xs,_ys = self.X[self.current_idx:next_idx],self.y[self.current_idx:next_idx]
        
        
        Xs,ys = torch.tensor(_Xs,requires_grad=False).to(device),torch.tensor(_ys,requires_grad=False).to(device)
                 

        self.current_idx = next_idx

        return Xs,ys