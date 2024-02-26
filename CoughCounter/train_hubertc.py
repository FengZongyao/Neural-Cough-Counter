import json
import torch
from pl_interface import Classification_Interface
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.strategies.ddp import DDPStrategy
from torch.utils.data import DataLoader,Subset
from dataloaders import AudioSegTrain

if __name__ == '__main__':

    pl.seed_everything(1314)
    
    if torch.cuda.is_available() and torch.cuda.device_count()>1:
        STRATEGY = DDPStrategy(find_unused_parameters=True)
    else:
        STRATEGY = 'auto'
        
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)["HUBERTC_Params"]

    check_point = ModelCheckpoint(dirpath = config['CHECKPOINT_DIR'],filename='HUBERTC'+'-best-{epoch:02d}-{valid_loss:.4f}',
                                monitor='valid_loss',save_top_k = 1,mode = 'min',save_weights_only = True)
    
    early_stopping = EarlyStopping(monitor = 'valid_loss', mode = 'min',patience = 5)
    
    
    model = Classification_Interface(lr = config['LR'],LossWeight = config['LOSS_WEIGHT'])
    
    logger = TensorBoardLogger(config['LOG_DIR'], name='HUBERTC')

    trainer = Trainer(strategy=STRATEGY,
                      logger = logger,max_epochs=config['EPOCHS'],log_every_n_steps=config['LOG_EVERY_N_STEPS'],
                      callbacks = [check_point,early_stopping],profiler='simple',
                     check_val_every_n_epoch=config['CHECK_VAL'])
    
    audiosesg = AudioSegTrain(config['ANNOTATIONS_FILE'],audio_duration=config["AUDIO_DURATION"])

    train_idx,valid_idx,test_idx = audiosesg.train_test_split()

    train_dataloader = DataLoader(Subset(audiosesg,train_idx), batch_size = config['BATCH_SIZE'],shuffle = True)
    valid_dataloader = DataLoader(Subset(audiosesg,valid_idx), batch_size = config['BATCH_SIZE'])
    test_dataloader = DataLoader(Subset(audiosesg,test_idx), batch_size = config['BATCH_SIZE'])

    trainer.fit(model,train_dataloader,valid_dataloader)

    trainer.test(dataloaders=test_dataloader)
    print(model.metrics)