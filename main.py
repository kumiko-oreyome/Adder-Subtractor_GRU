from utils import generate_examples,expression2Xs,answer2Ys
import pickle as pkl
from train import Trainer
import torch,data,argparse
from model import RNNCalc
import evaluate


def parse():
     parser = argparse.ArgumentParser(description='Attention Seq2Seq Calculator')



     parser.add_argument('mode',default="no")
     parser.add_argument('-bs', '--batch_size', help='generate data',default=32,type=int)
     parser.add_argument('-ep', '--epoch_num', help='generate data',default=50,type=int)
     parser.add_argument('-cpp', '--checkpoint_path', help='generate data')

     parser.add_argument('-g', '--generate_data', help='generate data',action='store_true')
     parser.add_argument('-gp', '--generate_path', help='generate data',default='./datas.pkl')
     parser.add_argument('-gn', '--generate_num', help='generate data',type=int)

     parser.add_argument('-ld', '--load_data', help='load data',action='store_true')
     parser.add_argument('-dp', '--data_path')

     parser.add_argument('-lm', '--load_model', help='load  model',action='store_true')
     parser.add_argument('-mp', '--model_path')
     parser.add_argument('-lr', '--lr')
     args = parser.parse_args()
     return args

from torch import optim
def run(args):
    torch.set_default_dtype(torch.float64)
    vocab = data.Vocab() 
    
    if args.generate_data:
        generate_train_val_test(args.generate_num,vocab,0.7,0.2,args.generate_path)
        return



    batch_size = args.batch_size
    if args.load_data == True:
        data_path = args.data_path
        with open(data_path,'rb') as f:
            train_questions,train_ans,val_questions,val_ans,test_questions,test_ans = pkl.load(f)
            train_generator = data.BatchGenerator(train_questions, train_ans, batch_size)
            val_generator = data.BatchGenerator(val_questions, val_ans, batch_size)



    lr = float(args.lr) or 0.01
    num_layers = 1
    
    if args.load_model == True:
        model_path = args.model_path
        checkpoint = torch.load(model_path)
        rnn = RNNCalc(*checkpoint['model_hyper'])
        rnn.load_state_dict(checkpoint['model']) 
        optimizer = torch.optim.SGD(rnn.parameters(),lr=lr,momentum=0.9, nesterov=True)
        #optimizer = optim.Adam(rnn.parameters())
        optim.Adam(rnn.parameters()).load_state_dict(checkpoint['optimzer'])
    else:
        #create new model
        embedding_dim,vocab_size,rnn_units = 32,vocab .size(),128
        rnn = RNNCalc(num_layers,embedding_dim,vocab_size,rnn_units)
        optimizer = optim.Adam(rnn.parameters(),lr=lr)
        
        
       
    if args.mode == 'train':
        assert optimizer is not None
        trainer = Trainer(rnn, optimizer ,args.epoch_num)
        trainer.train( train_generator,val_generator,vocab,lr,10,10,args.checkpoint_path)
    elif args.mode=='test':
        eva_generator = data.BatchGenerator(train_questions, train_ans, len(train_questions))
        evaluate.evaluate(rnn, eva_generator, vocab)


def generate_train_val_test(example_num,vocab,train_rate,val_rate,save_pkl_path):
    train_num = int(example_num*train_rate)
    val_num = int(example_num*val_rate)
    all_question,all_ans = generate_examples(example_num)
    print(all_question)
    train_questions,train_ans = all_question[:train_num],all_ans[:train_num]
    val_questions,val_ans = all_question[train_num:train_num+val_num],all_ans[train_num:train_num+val_num]
    test_questions,test_ans = all_question[train_num+val_num:],all_ans[train_num+val_num:]
    
    #save to plain text
    with open(save_pkl_path.replace('pkl','train.txt'),'w') as f:
        for q,a in zip(train_questions,train_ans):
            f.write('%s\t%s\n'%(q,a))
    with open(save_pkl_path.replace('pkl','val.txt'),'w') as f:
        for q,a in zip(val_questions,val_ans):
            f.write('%s\t%s\n'%(q,a))
    with open(save_pkl_path.replace('pkl','test.txt'),'w') as f:
        for q,a in zip(test_questions,test_ans):
            f.write('%s\t%s\n'%(q,a))    
    
    
    train_questions,train_ans = expression2Xs(train_questions,vocab),answer2Ys(train_ans,vocab)
    val_questions,val_ans = expression2Xs(val_questions,vocab),answer2Ys(val_ans,vocab) 
    test_questions,test_ans = expression2Xs(test_questions,vocab),answer2Ys(test_ans,vocab) 
    print((train_questions,train_ans,val_questions,val_ans,test_questions,test_ans))



    #save to pkl file
    with open(save_pkl_path,'wb') as f :
        pkl.dump((train_questions,train_ans,val_questions,val_ans,test_questions,test_ans),f)

    return train_questions,train_ans,val_questions,val_ans,test_questions,test_ans


#train_questions,train_ans,val_questions,val_ans,test_questions,test_ans =\



if __name__ == '__main__':
    #vocab = data.Vocab() 
    #generate_train_val_test(10,vocab,0.6,0.2,'./datas.pkl')
    args = parse()
    run(args)
#print((train_questions,train_ans,val_questions,val_ans,test_questions,test_ans))