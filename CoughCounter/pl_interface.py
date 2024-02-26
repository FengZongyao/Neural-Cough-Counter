from torch import nn
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy,AUROC,Recall,F1Score,Precision,MeanAbsoluteError
import importlib

class Classification_Interface(pl.LightningModule):
    
    def __init__(self,lr=1e-3,
                 LossWeight=[1,1]):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.configure_lossfn()
        self.configure_metrics()
        self.test_step_outputs = []

    def forward(self,x):

        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=self.hparams.lr,amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,'min',factor=0.4,patience=2,threshold=0.01, threshold_mode='abs')

        return [optimizer], {'scheduler':scheduler,'monitor':'valid_loss'}
    
    def configure_lossfn(self):
        loss_weight = self.hparams.LossWeight
        loss_weight = torch.tensor(loss_weight,dtype = torch.float)
        self.loss_fn = nn.CrossEntropyLoss(weight=loss_weight)
            
    def training_step(self,train_batch,batch_idx):
        x,y = train_batch
        x = self(x)
        loss = self.loss_fn(x,y)
        pred = F.softmax(x,1)
        _, pred = torch.max(pred, dim=1)
        self.train_acc(pred,y)
        self.log('train_acc',self.train_acc,on_step=False,on_epoch=True)
        self.log('train_loss',loss,on_step=False,on_epoch=True,prog_bar=True)

        return loss

    
    def validation_step(self,val_batch,batch_idx):
        x,y = val_batch
        x = self(x)
        loss = self.loss_fn(x,y)
        pred = F.softmax(x,1)
        _, pred = torch.max(pred, dim=1)
        self.val_acc(pred,y)
        self.recall(pred,y)
        self.f1_valid(pred,y)
        self.log('valid_acc',self.val_acc,on_step=False,on_epoch=True)
        self.log('valid_uar',self.recall,on_step=False,on_epoch=True)
        self.log('valid_f1',self.f1_valid,on_step=False,on_epoch=True,prog_bar=True)
        self.log('valid_loss',loss,on_step=False,on_epoch=True,prog_bar=True)
        
        return loss
    

    
    def test_step(self,test_batch,batch_idx):
        x,y = test_batch
        x = self(x)
        loss = self.loss_fn(x,y)
        self.test_step_outputs.append({'loss':loss,'outputs':x,'targets':y})
        return loss
    

    
    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        metrics = {}
        loss = torch.stack([out['loss'] for out in outputs]).mean()
        targets = torch.cat([out['targets'] for out in outputs])
        outs = torch.cat([out['outputs'] for out in outputs])
        outs = F.softmax(outs,1)
        _,preds = torch.max(outs, dim=1)
        metrics['f1'] = self.f1(preds,targets).item()
        metrics['uar'] = self.test_recall(preds,targets).item()
        metrics['acc'] = self.test_acc(preds,targets).item()
        metrics['auc'] = self.auroc(outs,targets).item()
        metrics['test_loss'] = loss
        self.test_outs = outs
        self.test_preds = preds
        self.metrics = metrics

    def predict_step(self,batch,batch_idx):
        pred = self(batch)
        pred = F.softmax(pred,1)
        _, pred = torch.max(pred, dim=1)
        return pred.item()
        
    def load_model(self):
        try:
            Model = getattr(importlib.import_module('models'),'HUBERTC')
        except:
            raise ValueError(f'Invalid Module File Name or Invalid Class Name !')
        self.model = Model()

        
    def configure_metrics(self):
        self.test_acc = Accuracy(task='multiclass',num_classes=2,average='micro',mdmc_average = 'samplewise')
        self.val_acc = Accuracy(task='multiclass',num_classes=2,average='macro',mdmc_average = 'samplewise')
        self.train_acc = Accuracy(task='multiclass',num_classes=2,average='macro',mdmc_average = 'samplewise')
        self.recall = Recall(task='multiclass',num_classes=2,average='macro',mdmc_average = 'samplewise')
        self.test_recall = Recall(task='multiclass',num_classes=2,average='macro',mdmc_average = 'samplewise')
        self.f1_valid = F1Score(task='multiclass',num_classes=2,average='macro',mdmc_average = 'samplewise')
        self.auroc = AUROC(task='multiclass',num_classes=2,average='macro',mdmc_average = 'samplewise')
        self.f1 = F1Score(task='multiclass',num_classes=2,average='macro',mdmc_average = 'samplewise')

########################################################################################################################################
import scipy.signal as sci
import sys
import numpy as np
import copy


class Detection_Interface(pl.LightningModule):     # REG AND FRAME
    
    def __init__(self,loss_name='MSE',lr=1e-3,
                 mu = 1,filter_window = 8,
                 filter_order = 1,grad_win = 6,pred_thresh = 0.5,noise_thresh = 0.185):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.configure_lossfn()
        self.configure_metrics()
        self.valid_best_loss= 1000000
        self.eval_boundary = [] 
        self.test_step_outputs,self.validation_step_outputs = [], []

    
    def forward(self,x):

        return self.model(x)

        
    def load_model(self):
        try:
            Model = getattr(importlib.import_module('models'),'HUBERT_RF')
        except:
            raise ValueError(f'Invalid Module File Name or Invalid Class Name!')
        self.model = Model()
        
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=self.hparams.lr,amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,'min',factor=0.4,patience=2,threshold=0.0001, threshold_mode='abs')
        return [optimizer], {'scheduler':scheduler,'monitor':'train_loss'}
    
    def configure_lossfn(self):
        self.smoothl1_loss = nn.SmoothL1Loss()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

        
    def join_loss(self,pred_f,pred_c,target_f,target_c):
        mu = self.hparams.mu
        L_f = self.ce_loss(pred_f,target_f)
        L_c = self.mse_loss(pred_c,target_c)
        self.log('LOSS_reg',L_c,on_epoch=True,sync_dist=True)
        self.log('LOSS_frame',L_f,on_epoch=True,sync_dist=True)
        Loss = mu*L_f + (1-mu)*L_c
        return Loss
    
    def loss(self,pred_f,pred_c,target_f,target_c):
        loss_name = ''.join(i.capitalize() for i in self.hparams.loss_name)
        if loss_name == 'MSE':
            Loss = self.mse_loss(pred_c,target_c)
        if loss_name == 'CROSSENTROPY':
            Loss = self.ce_loss(pred_f,target_f)
        if loss_name == 'JOIN_LOSS':
            Loss = self.join_loss(pred_f,pred_c,target_f,target_c)
        return Loss

    def load_for_training(self,training_layer,frame_checkpoint):
        checkpoint = torch.load(frame_checkpoint)
        try:
            self.model.load_state_dict(checkpoint['state_dict'])
        except RuntimeError:
            print('exception loading')
            keys = checkpoint['state_dict'].keys()
            for k in list(keys):
                checkpoint['state_dict'][k.replace('model.','')] = checkpoint['state_dict'].pop(k)
            missing_keys = self.model.load_state_dict(checkpoint['state_dict'],strict = False)
        for name,param in self.model.named_parameters():
            param.requires_grad = False if training_layer not in name else True
        print(f'loading completed, missing keys {missing_keys[0]}, {training_layer} requires grad is {getattr(self.model,training_layer).weight.requires_grad}')

        
    def training_step(self,train_batch,batch_idx):
        x,(y,y_c,mask) = train_batch
        pred_f,pred_c = self(x)
        pred_cmask,y_cmask = self.apply_mask(pred_c,y_c,mask)
        loss = self.loss(pred_f,pred_cmask,y,y_cmask)
        self.log('train_loss',loss,on_step=False,on_epoch=True,prog_bar=True) #sync_dist=True
        return loss
    
    def validation_step(self,val_batch,batch_idx):
        x,(y,y_c,mask) = val_batch
        pred_f,pred_c = self(x)
        pred_cmask,y_cmask = self.apply_mask(pred_c,y_c,mask)
        loss = self.loss(pred_f,pred_cmask,y,y_cmask)
        pred_f = F.softmax(pred_f,1)
        pred_f = (pred_f[:,1,:]>=self.hparams.pred_thresh).int()
        self.val_acc(pred_f,y)
        self.recall(pred_f,y)
        self.val_precision(pred_f,y)
        self.f1_valid(pred_f,y)
        self.val_mae(pred_cmask,y_cmask)
        self.log('valid_acc_frame',self.val_acc,on_step=False,on_epoch=True)
        self.log('valid_precision_frame',self.val_acc,on_step=False,on_epoch=True)
        self.log('valid_recall_frame',self.recall,on_step=False,on_epoch=True)
        self.log('valid_f1_frame',self.f1_valid,on_step=False,on_epoch=True)
        self.log('valid_MAE_reg',self.val_mae,on_step=False,on_epoch=True,prog_bar=True)
        self.validation_step_outputs.append({'loss':loss,'preds_f': pred_f,'preds_c': pred_c})
        return loss
    
    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        loss = torch.stack([out['loss'] for out in outputs]).mean()
        preds_f = torch.cat([out['preds_f'] for out in outputs])
        preds_c = torch.cat([out['preds_c'] for out in outputs])
        P,R,F1 = self.event_eval(preds_f,preds_c)
        self.log('valid_loss',loss,on_step=False,on_epoch=True,prog_bar=True)#sync_dist=True
        self.log('valid_precision_event',P,on_epoch=True)#sync_dist=True
        self.log('valid_recall_event',R,on_epoch=True)#sync_dist=True
        self.log('valid_F1_event',F1,on_epoch=True,prog_bar=True) #sync_dist=True
        if loss < self.valid_best_loss:
            self.valid_best_f1 = F1
            self.valid_best_loss = loss
        self.validation_step_outputs = []

    
    def test_step(self,test_batch,batch_idx):
        x,(y,y_c,mask) = test_batch
        pred_f,pred_c = self(x)
        pred_cmask,y_cmask = self.apply_mask(pred_c,y_c,mask)
        loss = self.loss(pred_f,pred_cmask,y,y_cmask)
        pred_f = F.softmax(pred_f,1)
        pred_f = (pred_f[:,1,:]>=self.hparams.pred_thresh).int()
        self.test_step_outputs.append({'loss':loss,'preds_f': pred_f,'preds_c': pred_c,'targets_f':y,'preds_cmask':pred_cmask,'targets_cmask':y_cmask})
        return loss
    

    
    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        metrics = {}
        loss = torch.stack([out['loss'] for out in outputs]).mean()
        targets_f = torch.cat([out['targets_f'] for out in outputs])
        targets_cmask = torch.cat([out['targets_cmask'] for out in outputs])
        preds_cmask = torch.cat([out['preds_cmask'] for out in outputs])
        preds_f = torch.cat([out['preds_f'] for out in outputs])
        preds_c = torch.cat([out['preds_c'] for out in outputs])
        P,R,F1 = self.event_eval(preds_f,preds_c)
        metrics['f1_frame'] = self.f1(preds_f,targets_f).item()
        metrics['recall_frame'] = self.test_recall(preds_f,targets_f).item()
        metrics['precision_frame'] = self.test_precision(preds_f,targets_f).item()
        metrics['acc_frame'] = self.test_acc(preds_f,targets_f).item()
        metrics['reg_MAE'] = self.test_mae(preds_cmask,targets_cmask).item()
        metrics['event_precision'] = P
        metrics['event_recall'] = R
        metrics['event_F1'] = F1
        metrics['test_loss_frame'] = loss
        self.outputs = outputs
        self.metrics = metrics
        
        
    def predict_step(self,batch,batch_idx):
        pred_f,pred_c = self(batch)
        pred_f = F.softmax(pred_f,1)
        pred_f = (pred_f[:,1,:]>=self.hparams.pred_thresh).int()
        if pred_f.any():
            pred_centers = self.combine_preds(pred_f,pred_c)
        else:
            pred_centers = []
        return pred_centers

    def event_eval(self,preds_f,preds_c):
        P = R = F1 = 0
        if preds_f.any():
            y_boundry = copy.deepcopy(self.eval_boundary)
            pred_centers = self.combine_preds(preds_f,preds_c)
            self.pred_centers = pred_centers
        tp,fp,fn = self.matching(pred_centers,y_boundry)
        if tp==0:
            tp = 1
        P = tp/(tp+fp)
        R = tp/(tp+fn)
        F1 = 2*P*R/(P+R)
        return P,R,F1
    
    def find_peaks(self,c_sample):
        y = sci.savgol_filter(c_sample, self.hparams.filter_window, self.hparams.filter_order)
        y[y<self.hparams.noise_thresh] = 0
        peaks = sci.find_peaks_cwt(y, self.hparams.grad_win)
        return peaks-2
    
    def sample_centers(self,pos_bounds,peaks,i):
        centers = peaks.tolist()
        if len(pos_bounds):
            for boundary in pos_bounds:
                if (boundary[1] - boundary[0]) < 5:
                    continue
                boundary_contains_center = False
                for center in peaks:
                    if boundary[0] <= center <= boundary[1]:
                        boundary_contains_center = True
                        break
                if boundary_contains_center:
                    continue
                mid = (boundary[0] + boundary[1]) / 2
                centers.append(mid)
        centers.sort()
        return (np.array(centers)*0.02+(i*4.98)).tolist()
    


    def combine_preds(self,preds_f,preds_c):
        pred_centers = []
        for i,f_sample in enumerate(preds_f):
            c_sample = preds_c[i]
            peaks = self.find_peaks(c_sample.cpu().numpy())
            pos_frames = torch.argwhere(f_sample).flatten()
            if not len(pos_frames):
                pos_bounds = []
            else:
                pos_bounds = list(self.group(pos_frames.tolist()))
            pred_centers += self.sample_centers(pos_bounds,peaks,i)

        return pred_centers
    
    def group(self,label):
        s = e = label[0]
        for i in label[1:]:
            if i-1 == e:
                e = i
            else:
                yield [s,e+1]
                s = e = i
        yield [s,e+1]
    
    def matching(self,centers,boundry):
        tp = 0
        LB = len(boundry)
        tp_centers = []
        for c in centers:
            for (s,e) in list(boundry):
                if s<=c<=e:
                    tp+=1
                    tp_centers.append(c)
                    boundry.remove([s,e])
                    break
        self.tp_centers = tp_centers
        return tp, len(centers)-tp,LB-tp
    
    def apply_mask(self,pred,label,mask):
        pred = torch.masked_select(pred, mask)
        label = torch.masked_select(label, mask) 
        return pred,label
    
    def configure_metrics(self):
        avg = 'macro'
        m_a = 'global'
        n_c = 2
        self.test_acc = Accuracy(task='multiclass',num_classes=n_c,mdmc_average = m_a)
        self.val_acc = Accuracy(task='multiclass',num_classes=n_c,mdmc_average = m_a)
        self.train_acc = Accuracy(task='multiclass',num_classes=n_c,mdmc_average = m_a)
        self.test_precision = Precision(task='multiclass',num_classes=n_c,average=avg,mdmc_average = m_a)
        self.val_precision = Precision(task='multiclass',num_classes=n_c,average=avg,mdmc_average = m_a)
        self.train_precision = Precision(task='multiclass',num_classes=n_c,average=avg,mdmc_average = m_a)
        self.recall = Recall(task='multiclass',num_classes=n_c,average=avg,mdmc_average = m_a)
        self.test_recall = Recall(task='multiclass',num_classes=n_c,average=avg,mdmc_average = m_a)
        self.f1_valid = F1Score(task='multiclass',num_classes=n_c,average=avg,mdmc_average = m_a)
        self.auroc = AUROC(task='multiclass',num_classes=n_c,average=avg,mdmc_average = m_a)
        self.f1 = F1Score(task = 'multiclass',num_classes=n_c,average=avg,mdmc_average = m_a)
        self.val_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()


########################################################################################################################################

class HUBERTF_Interface(pl.LightningModule):
    
    def __init__(self,lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.configure_lossfn()
        self.configure_metrics()
        self.test_step_outputs = []
    def forward(self,x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=self.hparams.lr,amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,'min',factor=0.4,patience=2,threshold=0.01, threshold_mode='abs')
        return [optimizer], {'scheduler':scheduler,'monitor':'valid_loss_frame'}
    
    def configure_lossfn(self):
        self.loss_fn = nn.CrossEntropyLoss()


    def training_step(self,train_batch,batch_idx):
        x,y = train_batch
        x = self(x)
        loss = self.loss_fn(x,y)
        pred = F.softmax(x,1)
        _, pred = torch.max(pred, dim=1)
        self.train_acc(pred,y)
        self.train_precision(pred,y)
        self.log('train_acc_frame',self.train_acc,on_step=False,on_epoch=True)
        self.log('train_precision_frame',self.train_precision,on_step=False,on_epoch=True)
        self.log('train_loss_frame',loss,on_step=False,on_epoch=True,prog_bar=True,sync_dist=True)

        return loss
    
    
    def validation_step(self,val_batch,batch_idx):
        x,y= val_batch
        x = self(x)
        loss = self.loss_fn(x,y)
        pred = F.softmax(x,1)
        _, pred = torch.max(pred, dim=1)
        self.val_acc(pred,y)
        self.recall(pred,y)
        self.val_precision(pred,y)
        self.f1_valid(pred,y)
        self.log('valid_acc_frame',self.val_acc,on_step=False,on_epoch=True)
        self.log('valid_precision_frame',self.val_acc,on_step=False,on_epoch=True)
        self.log('valid_recall_frame',self.recall,on_step=False,on_epoch=True)
        self.log('valid_f1_frame',self.f1_valid,on_step=False,on_epoch=True,prog_bar=True)
        self.log('valid_loss_frame',loss,on_step=False,on_epoch=True,prog_bar=True,sync_dist=True)
        
        return loss
    
    def test_step(self,test_batch,batch_idx):
        x,y = test_batch
        x = self(x)
        loss = self.loss_fn(x,y)
        y = y.long()
        self.test_step_outputs.append({'loss':loss,'outputs':x,'targets':y})
        return loss
    

    
    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        metrics = {}
        loss = torch.stack([out['loss'] for out in outputs]).mean()
        targets = torch.cat([out['targets'] for out in outputs])
        outs = torch.cat([out['outputs'] for out in outputs])
        outs = F.softmax(outs,1)
        _,preds = torch.max(outs, dim=1)
        metrics['f1_frame'] = self.f1(preds,targets).item()
        metrics['recall_frame'] = self.test_recall(preds,targets).item()
        metrics['precision_frame'] = self.test_precision(preds,targets).item()
        metrics['acc_frame'] = self.test_acc(preds,targets).item()
        metrics['test_loss_frame'] = loss
        self.outputs = outputs
        self.metrics = metrics
        
    def load_model(self):
        try:
            Model = getattr(importlib.import_module('models'),'HUBERTF')
        except:
            raise ValueError(f'Invalid Module File Name or Invalid Class Name !')
        self.model = Model()

        
    def configure_metrics(self):
        m_a = 'global'
        n_c = 2
        avg = 'macro'
        self.test_acc = Accuracy(task='multiclass',num_classes=n_c,mdmc_average = m_a)
        self.val_acc = Accuracy(task='multiclass',num_classes=n_c,mdmc_average = m_a)
        self.train_acc = Accuracy(task='multiclass',num_classes=n_c,mdmc_average = m_a)
        self.test_precision = Precision(task='multiclass',num_classes=n_c,average=avg,mdmc_average = m_a)
        self.val_precision = Precision(task='multiclass',num_classes=n_c,average=avg,mdmc_average = m_a)
        self.train_precision = Precision(task='multiclass',num_classes=n_c,average=avg,mdmc_average = m_a)
        self.recall = Recall(task='multiclass',num_classes=n_c,average=avg,mdmc_average = m_a)
        self.test_recall = Recall(task='multiclass',num_classes=n_c,average=avg,mdmc_average = m_a)
        self.f1_valid = F1Score(task='multiclass',num_classes=n_c,average=avg,mdmc_average = m_a)
        self.auroc = AUROC(task='multiclass',num_classes=n_c,average=avg,mdmc_average = m_a)
        self.f1 = F1Score(task = 'multiclass',num_classes=n_c,average=avg,mdmc_average = m_a)

########################################################################################################################################

class Distillation_Interface(HUBERTF_Interface):
    
    def __init__(self,lr=1e-3,temperature = 5,mu = .6):
        super().__init__(lr)
        self.distill_temperature = temperature
        self.mu = mu
        self.loss_fn_soft = nn.KLDivLoss(reduction="batchmean")
        
    def train_loss(self,preds,soft_target,target):
        mu = self.mu
        factor = mu/(1-mu)
        log_preds = F.log_softmax(preds/self.distill_temperature,dim=1)
        L_soft = self.loss_fn_soft(log_preds,soft_target)
        L_hard = self.loss_fn(preds,target)
        if L_soft/L_hard >= factor:
            Loss = torch.log(L_soft+1) + L_hard
        else:
            Loss = mu*L_soft + (1-mu)*L_hard
        self.log('LOSS_KLD',L_soft,on_epoch=True,sync_dist=True)
        self.log('LOSS_CE',L_hard,on_epoch=True,sync_dist=True)

        return Loss
    
    def load_teacher(self,teacher_checkpoint):
        checkpoint = torch.load(teacher_checkpoint)
        try:
            Model = getattr(importlib.import_module('models'),'HUBERTF')
            model = Model()
        except:
            raise ValueError(f'Invalid Module File Name or Invalid Class Name!')    
        try:
            model.load_state_dict(checkpoint['state_dict'])
        except RuntimeError:
            print('exception loading')
            keys = checkpoint['state_dict'].keys()
            for k in list(keys):
                checkpoint['state_dict'][k.replace('model.','')] = checkpoint['state_dict'].pop(k)
            # checkpoint['state_dict'].pop('loss_fn.weight')
            model.load_state_dict(checkpoint['state_dict'])
        print('loading teacher model complete')
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        self.teacher = model
    
    def soften(self,target):
        target /= self.distill_temperature
        return F.softmax(target,1)
        
    def training_step(self,train_batch,batch_idx):
        x,y = train_batch
        pred = self(x)
        soft_target = self.teacher(x)
        soft_target = self.soften(soft_target)
        loss = self.train_loss(pred,soft_target,y)
        pred = F.softmax(pred,1)
        _, pred = torch.max(pred, dim=1)
        self.train_acc(pred,y)
        self.train_precision(pred,y)
        self.log('train_acc_frame',self.train_acc,on_step=False,on_epoch=True)
        self.log('train_precision_frame',self.train_precision,on_step=False,on_epoch=True)
        self.log('train_loss_frame',loss,on_step=False,on_epoch=True,prog_bar=True,sync_dist=True)

        return loss
