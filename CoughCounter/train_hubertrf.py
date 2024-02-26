import json
import torch
from pl_interface import Detection_Interface
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.strategies.ddp import DDPStrategy
from torch.utils.data import DataLoader,Subset
from dataloaders import CoughSeg_RF



if __name__ == '__main__':

    pl.seed_everything(1314)
    
    if torch.cuda.is_available() and torch.cuda.device_count()>1:
        STRATEGY = DDPStrategy(find_unused_parameters=True)
    else:
        STRATEGY = 'auto'
        
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)["HUBERTRF_Params"]

    check_point = ModelCheckpoint(dirpath = config['CHECKPOINT_DIR'],filename='HUBERT_RF'+'-best-{epoch:02d}-{valid_loss:.4f}',
                                monitor='valid_loss',save_top_k = 1,mode = 'min',save_weights_only = True)
    
    early_stopping = EarlyStopping(monitor = 'valid_loss', mode = 'min',patience = 5)
    
    
    model = Detection_Interface(lr = config['LR'],loss_name=config['LOSS_NAME'],
                                filter_window=config['FILTER_WINDOW'],filter_order=config['FILTER_ORDER'],
                                grad_win=config['GRAD_WIN'],noise_thresh=config['NOISE_THRESH'])
    model.load_for_training(config['TRAINING_LAYER'],config['FRAME_CHECKPOINT'])
    logger = TensorBoardLogger(config['LOG_DIR'], name='HUBERT_RF')

    trainer = Trainer(strategy=STRATEGY,
                      logger = logger,max_epochs=config['EPOCHS'],log_every_n_steps=config['LOG_EVERY_N_STEPS'],
                      callbacks = [check_point,early_stopping],profiler='simple',
                     check_val_every_n_epoch=config['CHECK_VAL'])
    
    coughseg_train = CoughSeg_RF(config['ANNOTATIONS_FILE'],num_frames=config["NUM_FRAMES"],audio_duration=config['AUDIO_DURATION'])
    coughseg_valid = CoughSeg_RF(config['ANNOTATIONS_FILE'],num_frames=config["NUM_FRAMES"],audio_duration=config['AUDIO_DURATION'])
    coughseg_test = CoughSeg_RF(config['ANNOTATIONS_FILE'],num_frames=config["NUM_FRAMES"],audio_duration=config['AUDIO_DURATION'])
    train_idx,valid_idx,test_idx = coughseg_train.train_test_split()
    coughseg_train.labels = coughseg_train.labels.iloc[train_idx].reset_index(drop=True)
    coughseg_valid.labels = coughseg_valid.labels.iloc[valid_idx].reset_index(drop=True)
    coughseg_test.labels = coughseg_test.labels.iloc[test_idx].reset_index(drop=True)

    train_dataloader = DataLoader(coughseg_train, batch_size = config['BATCH_SIZE'],shuffle = True)
    

    valid_dataloader = DataLoader(coughseg_valid, batch_size = config['BATCH_SIZE'])
    model.eval_boundary = coughseg_valid.gen_boundary()
    trainer.fit(model,train_dataloader,valid_dataloader)

    test_dataloader = DataLoader(coughseg_test, batch_size = config['BATCH_SIZE'])
    model.eval_boundary = coughseg_test.gen_boundary()
    trainer.test(dataloaders=test_dataloader)
    print(model.metrics)