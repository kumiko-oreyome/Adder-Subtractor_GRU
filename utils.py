import numpy as np
import torch
device_name = "cuda" if torch.cuda.is_available() else "cpu"

def get_device():
    import torch
    device = torch.device("cpu")
    print(device)
    return device
import data
# raw data -> preprocess data <--> inverse
# prerocess data -->  model datas (X,y)  <--> inverse 
# model datas --> batch datas  

             
#TODO REVERSE

                                            # 1998~ -999 + EOS = 5 tokens
def generate_examples(num_examples,MAX_DIGIT=3,MAX_ANS_LEN=5):
    questions = []
    expected = []
    seen = set()

    while len(questions) < num_examples:
        f = lambda: int(''.join(np.random.choice(list('0123456789')) for i in range(np.random.randint(1, MAX_DIGIT + 1))))
        a, b = f(), f()
        op = np.random.choice(list('+-'))
        key = tuple('%d%c%d'%(a,op,b))
        if key in seen:
            continue

        seen.add(key)
        q = '{}{}{}'.format(a,op,b)
        query = q 
        if op == '+':
            ans = str(a + b)
        else:
            ans = str(a - b)
        questions.append(query)
        expected.append(ans)
    return questions,expected

def get_numbers_and_op(expression):
    op = '+'
    if op  not in expression  :
        op = '-'
    op_index = expression.find(op)
    num1 = expression[0:op_index]
    num2 = expression[op_index+1:]
    return num1,op,num2

def padding_sequence(x,vocab,max_len=3):
    assert len(x)<=max_len
    return x+[vocab.get_cid(data.PAD)]*(max_len-len(x))


#import functools
#_mask_func = lambda max_len,real_len:[1 if i<real_len else 0 for i in range(max_len)]
#mask_func = functools.partial(_mask_func, MAX_DIGIT)
def get_mask(seq_len,max_len):
    return [1 if i<seq_len else 0 for i in range(max_len)]


def expression2Xs(expressions,vocab,MAX_DIGIT=3):
    Xs = []
    for q in expressions:
        seq = vocab.encode_string(q)
        Xs.append(padding_sequence(seq,vocab,MAX_DIGIT+1+MAX_DIGIT))
    return Xs

def answer2Ys(answers,vocab,MAX_DIGIT=3):
    ys = []
    for ans in answers:
        seq = vocab.encode_string(ans)
        ys.append(padding_sequence(seq,vocab,MAX_DIGIT+1+MAX_DIGIT))
    return ys

