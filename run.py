#coding:utf-8

import torch
from torch.utils.data import  DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, trange
from transformers import AutoConfig, AutoTokenizer
from arguments import args
from dataset import myDataset
from model.mgsin import MGSIN
from utils import evaluate,get_acc_from_logits

SEED = 44

def train(model, criterion, optimizer, train_loader, test_loader, args, device, repeat, f_out):
    best_acc0 = 0
    best_f10, best_mf10, best_af10 = 0, 0, 0
    best_favor, best_against = 0, 0
    best_acc1 = 0
    best_f11, best_mf11, best_af11 = 0, 0, 0
    
    for epoch in trange(args.num_eps, desc="Epoch"):
        model.train()
        for i, (input_ids, attention_mask, types, labels, graphs, target_len, _) in enumerate(train_loader):
            input_ids, attention_mask, types= input_ids.to(device), attention_mask.to(device), types.to(device)
            labels, graphs = labels.to(device), graphs.to(device)
            labels = labels.permute(1,0)
            graphs = graphs.permute(1,0,2,3)
            optimizer.zero_grad()
            inputs = torch.stack((input_ids, input_ids, input_ids))
            masks = torch.stack((attention_mask, attention_mask, attention_mask))
            logits0, logits1, ls, lh, _ = model(inputs, masks, graphs, target_len)
            acc0,_,_= get_acc_from_logits(logits0, labels[0])
            acc1,_,_= get_acc_from_logits(logits1, labels[1])
            loss = 0.8 * criterion(logits0, labels[0]) + 0.7 * criterion(logits1, labels[1]) + 0.0005 * (ls + lh)
            print("Epoch {} step {}----> Accuracy0 : {}, Accuracy1 : {}, Loss : {}".format(epoch + 1, i + 1, acc0, acc1,loss.item()))
            loss.backward()
            optimizer.step()
        val_acc0, val_acc1, val_loss, val_f10, val_mf10, val_f11, val_mf11, favor, against= evaluate(epoch, model=model, criterion=criterion, dataloader=test_loader, device=device)
        print("Epoch {} complete! Stance: Accuracy : {}, Loss : {}, f1 : {}, mf1 : {}, af1 : {}".format(epoch + 1, val_acc0, val_loss.item(), val_f10, val_mf10, (val_f10 + val_mf10) / 2))
        print("Epoch {} complete! Sentiment: Accuracy : {}, Loss : {}, f1 : {}, mf1 : {}, af1 : {}".format(epoch + 1, val_acc1, val_loss.item(), val_f11, val_mf11, (val_f11 + val_mf11) / 2))
        if val_f10 > best_f10:
            print("Best Stance F1 improved from {} to {}, saving model...".format(best_f10, val_f10))
            best_acc0 = val_acc0
            best_f10 = val_f10
            best_mf10 = val_mf10
            best_af10 = (best_f10 + best_mf10) / 2
            best_favor = favor
            best_against = against
            #torch.save(model.state_dict(), '\parameter.pkl')
            #print('Model saved!')
        if val_f11 > best_f11:
            print("Best Sentiment F1 improved from {} to {}, saving model...".format(best_f11, val_f11))
            best_acc1 = val_acc1
            best_f11 = val_f11
            best_mf11 = val_mf11
            best_af11 = (best_f11 + best_mf11) / 2
    print("Best Stance Accuracy : {}, avg_f : {}, favor : {}, against : {}".format(best_acc0, (best_favor + best_against) / 2, best_favor, best_against), file=f_out)
    print("Best Stance Accuracy : {}, f1 : {}, mf1 : {}, af1 : {}".format(best_acc0, best_f10, best_mf10, best_af10), file=f_out)
    print("Best Sentiment Accuracy : {}, f1 : {}, mf1 : {}, af1 : {}".format(best_acc1, best_f11, best_mf11, best_af11), file=f_out)




if __name__ == "__main__":
    
    ### GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.manual_seed(SEED)
    
    #Configuration for the desired transformer model
    config = AutoConfig.from_pretrained('covid-bert')
    #Tokenizer for the desired transformer model
    tokenizer = AutoTokenizer.from_pretrained('covid-bert')
    target_list = ['', 'face_masks', 'fauci', 'school_closures', 'stay_at_home_orders']
    #target_list = ['', 'atheism', 'climate', 'hillary', 'feminist', 'abortion' ]
    target = 'face_masks'
    #'ClimateChange'
    
    train_set = myDataset(filename='./data/' + target + '/train_set/train', theme = 'train', maxlen=args.maxlen_train, tokenizer=tokenizer, target = target)
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    
    test_set = myDataset(filename='./data/' + target + '/test_set/test', theme = 'test',maxlen=args.maxlen_test, tokenizer=tokenizer, target = target)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle = False)
    
    f_out = open(target + '_woST_result.txt', 'w', encoding='utf-8')
    
    for i in range(1):
        #model
        model = MGSIN(args)
          
        model.to(device)
        bert_param = []
        other_param = []
        for pname, p in model.named_parameters():
            plist = pname.split('.')
            if(plist[0] == 'bert'):
                bert_param.append(p)
            else:
                other_param.append(p)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW([
                {'params': bert_param,},
                {'params': other_param, 'lr': 0.00001}], lr = args.lr)
        criterion = nn.CrossEntropyLoss()
        print("Repeat {}:".format(i + 1), file = f_out)
        train(model, criterion, optimizer, train_loader, test_loader, args, device, i, f_out)
        print("Repeat {} Completed!".format(i + 1))

