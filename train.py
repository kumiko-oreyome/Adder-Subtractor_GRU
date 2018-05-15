import torch,os
from tqdm import tqdm

import evaluate
class Trainer():
    def __init__(self,model,optimizer,epoch_num):
        self.epoch_num = epoch_num
        self.model = model
        self.optimizer = optimizer
        


    
    def train(self,train_generator,val_generator,vocab,lr,eval_every,save_every,save_dir):
       
        

        for epoch in tqdm(range(self.epoch_num)):
            print('Epoch :%d '%(epoch))
            for Xs,ys in train_generator.get_batches():
                self.optimizer.zero_grad()
                loss,predicts = self.model(Xs,ys)
                loss.backward()
                clip = 5.0
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                self.optimizer.step()
            
            print('loss is :%.3f'%(loss.item())) 

            if epoch % save_every == 0: 
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                torch.save({'epoch':str(epoch),'model':self.model.state_dict(),\
                'optimzer':self.optimizer.state_dict(),'model_hyper':self.model.get_hyper(),\
                'lr':lr},
                os.path.join(save_dir,'model_%d.pkl'%(epoch)))

            if epoch%eval_every == 0:
                print('learn rate')
                print('lr:%.5f'%(lr))
                print('train accuracy')
                with torch.no_grad():
                    evaluate.evaluate(self.model,train_generator,vocab)
                    print('validation accuracy')
                    evaluate.evaluate(self.model,val_generator,vocab)
            
            #if lr > 0.001:
            #    lr = lr*0.99
            #    for param_group in self.optimizer.param_groups:
            #        param_group['lr'] = lr
        
        print('evaluate')
        evaluate.evaluate(self.model,train_generator,vocab)
        torch.save({'epoch':str(epoch),'model':self.model.state_dict(),\
                'optimzer':self.optimizer.state_dict(),'model_hyper':self.model.get_hyper(),\
                'lr':lr},
                os.path.join(save_dir,'model_%d.pkl'%(epoch)))