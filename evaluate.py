import torch,data
import utils
device = utils.get_device()
def evaluate(model,test_generator,vocab):
    print('evaluation')
    acc_cnt = 0
    total_examples = 0
    for Xs,ys in test_generator.get_batches():
        avg_loss,predicts = model(Xs,ys)
        predicts = predicts.squeeze(2)
 
        predict_number_strs = seq2Number(predicts, vocab,EOS=False)
        answer_number_strs =  seq2Number(ys, vocab,EOS=False)
        #print(list(zip(predict_number_strs ,answer_number_strs)))
        for a,b in zip(predict_number_strs ,answer_number_strs):
            if a == b:
                acc_cnt+=1
            total_examples+=1

    print('accuracy:(%d/%d):%.3f'%(acc_cnt,total_examples,acc_cnt/total_examples))

    
# num tensors  N*MAX_SEQ_LEN
def seq2Number(num_tensors,vocab,EOS):
    assert len(num_tensors.size()) == 2
    max_seq_len = num_tensors.size(1)
    list_numbers = num_tensors.numpy().tolist()

    numbers = []
    for l in list_numbers:
        number_str = vocab.decode_sequence(l)
        numbers.append(number_str)

    number_strings = []
    for s in numbers:
        if EOS:
            end_idx = s.find(data.EOS)
            if end_idx == -1:
                end_idx = max_seq_len
        else:
            end_idx = s.rfind(data.PAD)
            if end_idx == -1:
                end_idx = max_seq_len

        number_str = s[:end_idx]
        number_strings .append(number_str)

    return number_strings

