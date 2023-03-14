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

def eval(opt):
    # ==================================================================================================================
    # =====  INITIALIZE ALL DATASET LOADERS
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
    train_data = KeyDataset(genre=opt.genre, opt=opt)
    val_data = KeyDataset(genre=opt.genre, opt=opt)
    
    # only for debug -> use only a handful samples
    if opt.debug:
        debug_d = DatasetGiantStepsMTGKeyLoader(os.path.join(file_path, 'Data/giantsteps-mtg-key-dataset'), data_type="debug")
        train_data.import_data(
            debug_d)
        val_data.import_data(
            debug_d)
    else:
        train_data.import_data(
            king_carole,
        )
    
        val_data.import_data(
            winterreise,
            giantsteps_key
        )
    if not opt.no_test and not opt.debug:
        # Initialize test datasets seperate to extract results for each individually
        test_beatles = KeyDataset(genre=opt.genre, opt=opt)
        test_beatles.import_data(beatles)
        test_isophonics = KeyDataset(genre=opt.genre, opt=opt)
        test_isophonics.import_data(beatles, king_carole, queen, zweieck)
        test_winterreise = KeyDataset(genre=opt.genre, opt=opt)
        test_winterreise.import_data(winterreise)
        test_giantsteps_key = KeyDataset(genre=opt.genre, opt=opt)
        test_giantsteps_key.import_data(giantsteps_key)
        test_mcbillboard = KeyDataset(genre=opt.genre, opt=opt)
        test_mcbillboard.import_data(mcbillboard)

    # ==================================================================================================================
    # =====  Model Evaluation
    # ==================================================================================================================

    # Pack data in Dataloader:
    train = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, pin_memory=True)
    val = DataLoader(val_data, batch_size=opt.batch_size, shuffle=False, drop_last=True, pin_memory=True)
    if not opt.no_test and not opt.debug:
        test_w = DataLoader(test_winterreise, batch_size=opt.batch_size, shuffle=False, drop_last=False, pin_memory=True)
        test_g = DataLoader(test_giantsteps_key, batch_size=opt.batch_size, shuffle=False, drop_last=False, pin_memory=True)
        test_b = DataLoader(test_beatles, batch_size=opt.batch_size, shuffle=False, drop_last=False, pin_memory=True)
        test_isophonics = DataLoader(test_isophonics, batch_size=opt.batch_size, shuffle=False, drop_last=False, pin_memory=True)
        test_mc = DataLoader(test_mcbillboard, batch_size=opt.batch_size, shuffle=False, drop_last=False, pin_memory=True)
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
    
    logger = TensorBoardLogger("Model_logs")
    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        gpus=1 if torch.cuda.is_available() else None,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        logger=False,
    )
    
    model_path = find_path(opt)
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    print("Result of Validation set")
    trainer.validate(model, dataloaders=val)
    if not opt.no_test and not opt.debug:
        print("Result of Winterreise set")
        result_winterreise = trainer.validate(model, dataloaders=test_w)
        print("Result of Giantsteps set")
        result_giantsteps_key = trainer.validate(model, dataloaders=test_g)
        print("Result of Beatles set")
        result_beatles = trainer.validate(model, dataloaders=test_b)
        print("Result of McGillBillboard set")
        result_mcbillboard = trainer.validate(model, dataloaders=test_mc)
        print("Result of Isophonics set")
        result_isophonics = trainer.validate(model, dataloaders=test_isophonics)
        

def find_path(opt):
    parent_path = os.path.join(os.getcwd(), "Model_logs/lightning_logs/version_"+str(opt.version))
    model_path = os.path.join(parent_path, "best_model.pt")
    
    return model_path
        
        
def init_parser():
    # Settings
    parser = argparse.ArgumentParser(
        description='Hyperparameters for Key estimation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--batch_size', type=int, required=False, default=1,
                        help='Batch size!')
    parser.add_argument('--lr', type=float, required=False, default=3e-4,
                        help='Learning Rate!')
    parser.add_argument('--drop', type=float, required=False, default=0.1,
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
    parser.add_argument('--version', type=int, required=True,
                        help='Trained model version number which shall be evaluated')
    parser.add_argument('--no_ckpt', action="store_false",
                        help='Do not save best model! Default deactivated')
    
    opt = parser.parse_args()
    
    
    return opt

if __name__ == "__main__":
    opt = init_parser()
    eval(opt)
