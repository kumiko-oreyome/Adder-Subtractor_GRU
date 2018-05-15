#from utils import generate_examples,get_numbers_and_op,get_Xs,get_ys
import torch,data

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#vocab = data.Vocab()

#questions,expected = generate_examples(800,REVERSE=False)
#print(questions[:5], expected[:5])
#print(questions[0],expected[0])
#a,b,c=get_numbers_and_op(questions[0])
#print(questions[1],expected[1])
#d,e,f=get_numbers_and_op(questions[1])

#calc_len = lambda x:len(x.strip())
#calculate real length
#print(list(map(calc_len,[a,c,d,f])))


#import functools
#_mask_func = lambda max_len,real_len:[1 if i<real_len else 0 for i in range(max_len) ]
#mask_func = functools.partial(_mask_func, 3)

#calculate masks
#print(list(map(mask_func,list(map(calc_len,[a,c,d,f])))))

#print(questions[0:3])
#Xs,X_lens,maskXs = get_Xs(questions[0:3],vocab)
#print(Xs,X_lens,maskXs)
#ys,y_lens,maskys = get_ys(expected[0:3],vocab)
#print(expected[0:3])
#print(ys,y_lens,maskys)


#from model import Seq2seqCalc
#calc = Seq2seqCalc(50,vocab.size(),50,50)
#calc.build()
#calc(Xs,X_lens,maskXs,ys,y_lens,maskys)
#X
#[n1,op,n2]    
#[l1,l2]
#[mask1,mask2]


#y
#[ans]    
#[l]
#[mask]

def nLLoss(inp, target):
    
    logits = torch.gather(inp, 2, target.unsqueeze(2))
    # logits N*S*1
    assert inp.size(0)*inp.size(1) == torch.sum(logits.ge(0)).item()
    crossEntropy = -torch.log(logits)
 
    loss =  torch.tensor(crossEntropy.mean(1).mean(0).sum())
    return loss

inp = torch.tensor([[[0.2,0.5,0.2,0.1],[0.4,0.2,0.1,0.1]],[[0.4,0.1,0.4,0.1],[0.7,0.1,0.1,0.1]]])
target = torch.tensor([[0,1],[2,3]])
print(inp)
print(nLLoss(inp,target))


t = torch.tensor([[[1,-1,2,3],[2,3,-5,2]],[[4,-1,2,1],[7,-1,-1,-1]]])
A = torch.tensor([[4,2],[1,1],[-1,0],[5,2]])
r = torch.matmul(t,A)
print(t)
print(A)
print(r)

softmax =  torch.nn.Softmax(2)
print(softmax(inp))