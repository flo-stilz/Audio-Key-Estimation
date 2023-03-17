import matplotlib.pyplot as plt

from KeyDataset import *
from torch.utils.data import DataLoader
from models import *
import os
from pathlib import Path
import argparse
from torch.multiprocessing import Pool, Process, set_start_method, cpu_count
import pandas as pd
import torch
import sys

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)


def train(opt):
    # ==================================================================================================================
    # =====  INITIALIZE ALL DATASETLOADERS
    # ==================================================================================================================
    file_path = Path(os.path.dirname(os.path.abspath(os.getcwd())))
    
    giantsteps_key     = DatasetGiantStepsKeyLoader(os.path.join(file_path, 'Data/giantsteps-key-dataset'))
    giantsteps_mtg_key = DatasetGiantStepsMTGKeyLoader(os.path.join(file_path, 'Data/giantsteps-mtg-key-dataset'))
    winterreise        = DatasetSchubertWinterreiseLoader(os.path.join(file_path, 'Data/Schubert_Winterreise_Dataset_v1-1'), opt.local)
    gtzan              = DatasetGTZANLoader(os.path.join(file_path, 'Data/GTZAN'))
    guitar_set         = DatasetGuitarSetLoader(os.path.join(file_path, 'Data/GuitarSet'))
    FSL10K             = DatasetFSL10KLoader(os.path.join(file_path, 'Data/FSL10K'))
    tonality           = DatasetTonalityClassicalDBLoader(os.path.join(file_path, 'Data/Tonality'))
    key_finder         = DatasetKeyFinderLoader(os.path.join(file_path, "Data/KeyFinder"))
    beatles            = DatasetBeatlesLoader(os.path.join(file_path, "Data/Beatles_Isophonics"))
    king_carole        = DatasetKingCaroleLoader(os.path.join(file_path, "Data/King_Carole_Isophonics"))
    queen              = DatasetQueenLoader(os.path.join(file_path, "Data/Queen_Isophonics"))
    zweieck            = DatasetZweieckLoader(os.path.join(file_path, "Data/Zweieck_Isophonics"))
    ultimate_songs      = DatasetUltimateSongsLoader(os.path.join(file_path, "Data/UltimateSongs"))
    mcbillboard       = DatasetMcGillBillboardLoader(os.path.join(file_path, "Data/McGill-Billboard"))
    
    # ==================================================================================================================
    # =====  SPLIT DATA INTO A TRAIN- VALIDATION- AND TEST-SET
    # ==================================================================================================================
    # The DatasetLoaderMultiDS includes shuffling of the all passed datasets
    train_data = KeyDataset(genre=opt.genre, opt=opt)
    val_data = KeyDataset(genre=opt.genre, opt=opt)
    
    if opt.debug:
        debug_d = DatasetGiantStepsMTGKeyLoader(os.path.join(file_path, 'Data/giantsteps-mtg-key-dataset'), data_type="debug")
        train_data.import_data(
            debug_d)
        val_data.import_data(
            debug_d)
    else:
        train_data.import_data(
            giantsteps_mtg_key,
            gtzan,
            key_finder,
            tonality,
            guitar_set,
            ultimate_songs,
        )
    
        val_data.import_data(
            winterreise,
            giantsteps_key
        )
    if not opt.no_test and not opt.debug:
        # Initialize test datasets seperate to extract results for each individually
        test_beatles = KeyDataset(genre=opt.genre, opt=opt)
        test_beatles.import_data(beatles)
        test_winterreise = KeyDataset(genre=opt.genre, opt=opt)
        test_winterreise.import_data(winterreise)
        test_giantsteps_key = KeyDataset(genre=opt.genre, opt=opt)
        test_giantsteps_key.import_data(giantsteps_key)
        test_mcbillboard = KeyDataset(genre=opt.genre, opt=opt)
        test_mcbillboard.import_data(mcbillboard)

    # ==================================================================================================================
    # =====  Model Training
    # ==================================================================================================================
    #Pack data in Dataloader:
    train = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, pin_memory=True, num_workers=12)
    val = DataLoader(val_data, batch_size=opt.batch_size, shuffle=False, drop_last=True, pin_memory=True, num_workers=12)
    # Initialize model
    os.environ["CUDA_VISIBLE_DEVICES"]=str(opt.gpu)
    
    if not opt.only_semitones:
        shape = opt.octaves*36
    elif opt.only_semitones:
        shape = opt.octaves*12
    else:
        shape = 360
    
    if opt.multi_scale:
        shape1=opt.octaves*36
        shape2=opt.octaves*36
        model = PitchClassNet_Multi(shape1, shape2, 12, opt.num_layers, opt.kernel_size, opt=opt, window_size=opt.window_size, batch_size=opt.batch_size, train_set=train_data, val_set=val_data).double().cuda()
    else:
        model = PitchClassNet(shape, 12, opt.num_layers, opt.kernel_size, opt=opt, window_size = opt.window_size, batch_size=opt.batch_size, train_set=train_data, val_set=val_data).double().cuda()
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = model.to(device)
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=10, verbose=False, mode="min")
    best_checkpoint = ModelCheckpoint(monitor='val_mirex_score', save_top_k=1, mode="max")
    
    logger = TensorBoardLogger("Model_logs")
    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        gpus=1 if torch.cuda.is_available() else None,
        callbacks=[early_stop_callback, best_checkpoint],
        num_sanity_val_steps = 0,
        logger=logger,
        accumulate_grad_batches=opt.acc_grad,
    )
    trainer.fit(model) # train the standard classifier 
    final_model_path = trainer.checkpoint_callback.best_model_path # load best model checkpoint
    result = trainer.validate(ckpt_path=final_model_path, dataloaders=val)
    

    hp_results['val_acc'] = result[0]['val_accuracy']
    hp_results['val_acc_tonic'] = result[0]['val_accuracy_tonic']
    if opt.genre:
        hp_results['val_acc_genre'] = result[0]['val_accuracy_genre']
    hp_results['val_loss'] = result[0]['val_loss']
    hp_results['val_mirex'] = result[0]['val_mirex_score']
    hp_results['val_correct'] = result[0]['val_correct']
    hp_results['val_fifths'] = result[0]['val_fifths']
    hp_results['val_relative'] = result[0]['val_relative']
    hp_results['val_parallel'] = result[0]['val_parallel']
    hp_results['val_other'] = result[0]['val_other']
    hp_results['lr'] = opt.lr
    hp_results['num_layers'] = opt.num_layers
    hp_results['kernel_size'] = opt.kernel_size
    hp_results['conv_layers'] = opt.conv_layers
    hp_results['n_filters'] = opt.n_filters
    hp_results['resblock'] = opt.resblock
    hp_results['denseblock'] = opt.denseblock
    hp_results['head_layers'] = opt.head_layers
    hp_results['effective_batch_size'] = opt.batch_size*opt.acc_grad
    hp_results['tonic_loss_weight'] = opt.tonic_weight
    hp_results['genre_loss_weight'] = opt.genre_weight
    hp_results['time_pool_size'] = opt.time_pool_size
    results.append(hp_results)
    tuning_value = "Experiment_1"
    res_frame = pd.DataFrame(results)
    res_frame.to_csv(str(os.getcwd()) + '/Tuning_results_'+str(tuning_value)+'.csv', index=False)
               
        
    return results
        
        
def init_parser():
    # Settings
    parser = argparse.ArgumentParser(
        description='Hyperparameters for Key estimation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--batch_size', type=int, required=False, default=8,
                        help='Batch size!')
    parser.add_argument('--lr', type=float, required=False, default=3e-4,
                        help='Learning Rate!')
    parser.add_argument('--drop', type=float, required=False, default=0.0,
                        help='Dropout of model')
    parser.add_argument('--reg', type=float, required=False, default=0,
                        help='Weight decay')
    parser.add_argument('--gamma', type=float, required=False, default=0.96,
                        help='Gamma value for Exponential LR-Decay')
    parser.add_argument('--acc_grad', type=int, required=False, default=8,
                        help='Set to 1 if no gradient accumulation is desired else enter desired value for accumulated batches before gradient step')
    parser.add_argument('--epochs', type=int, required=False, default=100,
                        help='Set maxs number of epochs.')
    parser.add_argument('--window_size', type=int, required=False, default=592,
                        help='Set window size on cqt!')
    parser.add_argument('--local', action="store_true",
                        help='Train on local key estimation')
    parser.add_argument('--gpu', type=int, required=True,
                        help='Set the name of the GPU in the system')
    parser.add_argument('--octaves', type=int, required=False, default=8,
                        help='How many octaves to consider for CQT')
    parser.add_argument('--conv_layers', type=int, required=False, default=3,
                        help='How many convolutional layers per PitchClassNet layer')
    parser.add_argument('--n_filters', type=int, required=False, default=4,
                        help='standard number of filters within PitchClassNet')
    parser.add_argument('--num_layers', type=int, required=False, default=2,
                        help='Number of Layers within PitchClassNet')
    parser.add_argument('--kernel_size', type=int, required=False, default=7,
                        help='Standard kernel size for PitchClassNet')
    parser.add_argument('--key_weight', type=float, required=False, default=1.0,
                        help='Loss weight for key loss')
    parser.add_argument('--tonic_weight', type=float, required=False, default=1.0,
                        help='Loss weight for tonic loss')
    parser.add_argument('--genre_weight', type=float, required=False, default=0.1,
                        help='Loss weight for genre loss')
    parser.add_argument('--resblock', action="store_true",
                        help='Use Resblocks instead of basic Convs')
    parser.add_argument('--denseblock', action="store_true",
                        help='Use Dense blocks instead of basic Convs')
    parser.add_argument('--frames', type=int, required=False, default=5,
                        help='Sets Hop_Length for cqts to represent 1 sec in song in exactly that many frames')
    parser.add_argument('--genre', action="store_true",
                        help='Train also on Genre!')
    parser.add_argument('--stay_sixth', action="store_true",
                        help='Immediately downsize to semitone representation in CQT!')
    parser.add_argument('--p2pc_conv', action="store_true",
                        help='Use Conv to downsample from pitch level to pitch class level! If false then use max pool')
    parser.add_argument('--head_layers', type=int, required=False, default=2,
                        help='Number of Conv Layers in classification heads at the end of PitchClassNet')
    parser.add_argument('--loc_window_size', type=int, required=False, default=10,
                        help='Amount of sec. shall be considered by local Key estimation per prediction')
    parser.add_argument('--time_pool_size', type=int, required=False, default=2,
                        help='Pooling size along time dimension for each layer')
    parser.add_argument('--only_semitones', action="store_true",
                        help='Preprocess CQTs in semitones meaning 12 bins per octave else 36(12*3) bins per octave')
    parser.add_argument('--multi_scale', action="store_true",
                        help='Preprocess CQTs in semitones and only tones and run two models and merge predictions')
    parser.add_argument('--no_test', action="store_true",
                        help='Do not evaluate on test data')
    parser.add_argument('--debug', action="store_true",
                        help='Use limited amount of data for fast run through')
    parser.add_argument('--linear_reg_multi', action="store_true",
                        help='Use linear regression for output combination of two scale models')
    parser.add_argument('--use_cos', action="store_true",
                        help='Use Cosine Similarity as additional loss term for Key Estimation')
    parser.add_argument('--pc2p_mem', action="store_true",
                        help='Use Memory-efficient variant for upsampling pitc class wise features to pitch wise features')
    parser.add_argument('--no_ckpt', action="store_true",
                        help='Do not save best model!')
    
    opt = parser.parse_args()
    
    
    return opt

if __name__ == "__main__":
    opt = init_parser()
    train(opt)
