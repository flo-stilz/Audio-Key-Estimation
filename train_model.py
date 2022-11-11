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

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)


def train(opt):
    # ==================================================================================================================
    # =====  INITIALIZE ALL DATASET LOADERS
    # ==================================================================================================================
    file_path = Path(os.path.dirname(os.path.abspath(os.getcwd())))
    
    giantsteps_key     = DatasetGiantStepsKeyLoader(os.path.join(file_path, 'Data/giantsteps-key-dataset'))
    giantsteps_mtg_key = DatasetGiantStepsMTGKeyLoader(os.path.join(file_path, 'Data/giantsteps-mtg-key-dataset'))
    winterreise        = DatasetSchubertWinterreiseLoader(os.path.join(file_path, 'Data/Schubert_Winterreise_Dataset_v1-1'))
    gtzan              = DatasetGTZANLoader(os.path.join(file_path, 'Data/GTZAN'))
    guitar_set         = DatasetGuitarSetLoader(os.path.join(file_path, 'Data/GuitarSet'))
    FSL10K             = DatasetFSL10KLoader(os.path.join(file_path, 'Data/FSL10K'))
    tonality           = DatasetTonalityClassicalDBLoader(os.path.join(file_path, 'Data/Tonality'))
    key_finder         = DatasetKeyFinderLoader(os.path.join(file_path, "Data/KeyFinder"))
    beatles            = DatasetBeatlesLoader(os.path.join(file_path, "Data/Beatles_Isophonics"))
    king_carole        = DatasetKingCaroleLoader(os.path.join(file_path, "Data/King_Carole_Isophonics"))
    queen              = DatasetQueenLoader(os.path.join(file_path, "Data/Queen_Isophonics"))
    zweieck            = DatasetZweieckLoader(os.path.join(file_path, "Data/Zweieck_Isophonics"))
    popular_songs      = DatasetPopularSongsLoader(os.path.join(file_path, "Data/PopularSongs"))
    
    # Test Data Distribution:
    gmk_train = DatasetGiantStepsMTGKeyLoader(os.path.join(file_path, 'Data/giantsteps-mtg-key-dataset'), data_type="train")
    gmk_val = DatasetGiantStepsMTGKeyLoader(os.path.join(file_path, 'Data/giantsteps-mtg-key-dataset'), data_type="val")

    # ==================================================================================================================
    # =====  DISPLAY THE CLASS DISTRIBUTION
    # ==================================================================================================================

    """entire_ds = key_finder_ds
    keys = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for spec, label_id in entire_ds:
        keys[label_id] = keys[label_id] + 1

    key_signatures = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
    sets = keys
    plt.bar(key_signatures, sets)
    plt.title('Dataset Class Distribution')
    plt.xlabel('Key Signature')
    plt.show()"""

    # ==================================================================================================================
    # =====  SPLIT DATA INTO A TRAIN- VALIDATION- AND TEST-SET
    # ==================================================================================================================
    genre = False
    # The DatasetLoaderMultiDS includes shuffling of the all passed datasets
    train_data = KeyDataset(genre=genre, opt=opt)
    val_data = KeyDataset(genre=genre, opt=opt)
    
    train_data.import_data(
        giantsteps_mtg_key,
        #gtzan,
        #key_finder,
        #tonality,
        #FSL10K,
        #guitar_set,
        #beatles,
        #king_carole,
        #queen,
        #zweieck,
        #popular_songs,
        #gmk_train
    )

    val_data.import_data(
        winterreise,
        #gmk_val,
        giantsteps_key
        #king_carole,
    )
    
    # test set and val set identical atm: must be changed once all data is gathered to avoid data leakage
    test_data = val_data # TODO: val and test shall not be the same
    '''
    winterreise_ds = dataset_loader_multi.import_data(
            winterreise)
    
    giantsteps_key_ds = dataset_loader_multi.import_data(
            giantsteps_key)
    '''
    '''
    test_ds = dataset_loader_multi.import_data(
        giantsteps_key,
        #gtzan,
        winterreise,
        #key_finder,
        #tonality,
        #guitar_set
    )
    '''
    # TODO: make batch_size, learning_rate, decay_rate, #epochs into console arguments (argv).
    
    #print(train.__getitem__(0))

    # ==================================================================================================================
    # =====  SET-UP MODEL AND IT'S OPTIMIZERS
    # ==================================================================================================================
    
    results = []
    #lrs = [0.004, 0.002, 0.001, 0.0005, 0.0003, 0.0002, 0.0001]
    lrs = [1e-2, 3e-3,1e-3, 3e-4,1e-4, 3e-5,1e-5, 3e-6,1e-6,3e-7,1e-7]
    lrs = [3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
    #lrs = [1e-1]
    nl = [8, 9, 10]
    nl = [1, 2, 3, 4]
    #nl = [1]
    kernel_sizes = [1, 3, 5, 7, 9, 11, 13]
    kernel_sizes = [3]
    conv_layers = [1, 2, 3]
    #conv_layers = [1]
    number_filters = [2, 4, 8, 16, 32, 64, 128]
    number_filters = [4]
    resblocks = [False]
    denseblocks = [False]
    
    g_w = [0.005] # genre loss weights for final loss function
    #nf = [4]
    #for hp in range(len(g_w)):
    for n in range(len(nl)):
        for res in range(len(resblocks)):
            for dense in range(len(denseblocks)):
                for ker in range(len(kernel_sizes)):
                    for conv in range(len(conv_layers)):
                        for nf in range(len(number_filters)):
                            for hp in range(len(lrs)):
                                '''
                                if n==0 and ker==0:
                                    continue
                                '''
                                genre_loss_weight = g_w[0]
                                lr=lrs[hp]
                                opt.lr = lr
                                num_layers = nl[n]
                                opt.num_layers = num_layers
                                kernel_size = kernel_sizes[ker]
                                opt.kernel_size = kernel_size
                                conv_layer = conv_layers[conv]
                                opt.conv_layers = conv_layer
                                n_filters = number_filters[nf]
                                opt.n_filters = n_filters
                                opt.resblock = resblocks[res]
                                opt.denseblock = denseblocks[dense]
                                print("LR: "+str(opt.lr))
                                print("Number of Filters: "+str(opt.n_filters))
                                print("Kernel size: "+str(opt.kernel_size))
                                print("Amount of conv layers: "+str(opt.conv_layers))
                                print("Resblocks used: "+str(opt.resblock))
                                print("Denseblocks used: "+str(opt.denseblock))
                                '''
                                steps_in_epoch = train_ds.cardinality().numpy()
                                lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                                    initial_learning_rate,
                                    decay_steps=steps_in_epoch*40,
                                    decay_rate=0.9,
                                    staircase=True)
                                '''
                                #Pack data in Dataloader:
                                train = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, pin_memory=True, num_workers=12)
                                val = DataLoader(val_data, batch_size=opt.batch_size, shuffle=False, drop_last=True, pin_memory=True, num_workers=12)
                                # Initialize model
                                os.environ["CUDA_VISIBLE_DEVICES"]=str(opt.gpu)
                                
                                if opt.octaves==5:
                                    shape = 180
                                elif opt.octaves==7:
                                    shape = 252
                                elif opt.octaves==8:
                                    shape = 288
                                else:
                                    shape = 360
                                
                                model = AttentionPitchClassNet(shape, 12, opt.num_layers, opt.kernel_size, opt=opt, window_size = opt.window_size, batch_size=opt.batch_size, train_set=train_data, val_set=val_data).double().cuda()
                                #model = TestNet(shape, 12, opt.num_layers, opt.kernel_size, opt=opt, window_size = opt.window_size, batch_size=opt.batch_size, train_set=train_data, val_set=val_data).double().cuda()
                                #model = JXC1(shape, 12, opt.num_layers, opt.kernel_size, opt=opt, window_size = opt.window_size, batch_size=opt.batch_size, train_set=train_data, val_set=val_data).double().cuda()
                                #model = EquivariantDilatedModel(nf=opt.n_filters, batch_size=opt.batch_size, opt=opt, train_set=train_data, val_set=val_data)
                                #model = AttentionPitchClassNetTest(shape, 12, num_layers, kernel_size, opt=opt, window_size = opt.window_size, batch_size=opt.batch_size, train_set=train_data, val_set=val_data).double().cuda()
                                #model = TestCNN(180, 12, num_layers, kernel_size=5, opt=opt, window_size=opt.window_size, batch_size=opt.batch_size, train_set=train_data, val_set=val_data).double().cuda()
                                #model = EqCNNKey(shape, 12, num_layers, 5, opt=opt, window_size = opt.window_size, batch_size=opt.batch_size, train_set=train_data, val_set=val_data).double().cuda()
                                
                                # initialize weights:
                                #model.apply(weights_init)
                                
                                
                                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                                print(device)
                                model = model.to(device)
                                #print(model)
                                early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=100, verbose=False, mode="min")
                                #best_checkpoint = ModelCheckpoint(monitor='val_accuracy', save_top_k=1, mode="max")
                                best_checkpoint = ModelCheckpoint(monitor='val_mirex_score', save_top_k=1, mode="max")
                                #best_checkpoint = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode="min")
                                
                                logger = TensorBoardLogger("Model_logs")
                                trainer = pl.Trainer(
                                    max_epochs=opt.epochs,
                                    gpus=1 if torch.cuda.is_available() else None,
                                    #distributed_backend='dp',
                                    callbacks=[early_stop_callback, best_checkpoint],
                                    num_sanity_val_steps = 0,
                                    #logger=True,
                                    #log_every_n_steps=50,
                                    logger=logger,
                                    accumulate_grad_batches=opt.acc_grad,
                                )
                                #trainer.validate(model, dataloaders=val) # only needed when validation curve shall start on x=0
                                trainer.fit(model) # train the standard classifier 
                                final_model_path = trainer.checkpoint_callback.best_model_path # load best model checkpoint
                                result = trainer.validate(ckpt_path=final_model_path, dataloaders=val)
                                result_train = trainer.validate(ckpt_path=final_model_path, dataloaders=train)
                                #model.load_state_dict(torch.load(final_model_path)['state_dict'], strict=True)
                                #trainer.validate(model, dataloaders=val)
                                hp_results = {}
                        
                                hp_results['val_acc'] = result[0]['val_accuracy']
                                hp_results['val_acc_tonic'] = result[0]['val_accuracy_tonic']
                                hp_results['val_loss'] = result[0]['val_loss']
                                hp_results['val_mirex'] = result[0]['val_mirex_score']
                                hp_results['val_correct'] = result[0]['val_correct']
                                hp_results['val_fifths'] = result[0]['val_fifths']
                                hp_results['val_relative'] = result[0]['val_relative']
                                hp_results['val_parallel'] = result[0]['val_parallel']
                                hp_results['val_other'] = result[0]['val_other']
                                hp_results['train_loss'] = result_train[0]['val_loss']
                                hp_results['train_acc'] = result_train[0]['val_accuracy']
                                hp_results['train_mirex'] = result_train[0]['val_mirex_score']
                                hp_results['lr'] = opt.lr
                                hp_results['num_layers'] = opt.num_layers
                                hp_results['kernel_size'] = opt.kernel_size
                                hp_results['conv_layers'] = opt.conv_layers
                                hp_results['n_filters'] = opt.n_filters
                                hp_results['resblock'] = opt.resblock
                                hp_results['denseblock'] = opt.denseblock
                                hp_results['effective_batch_size'] = opt.batch_size*opt.acc_grad
                                print(result[0]['val_accuracy'])
                                results.append(hp_results)
                                tuning_value = "Basic_N_Filters"#"PitchClassNet_Layer_1_Big_Test"#"PitchClassNet-Mirex_Score_ConvSemitoneBothDirections_More_Data"
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
    parser.add_argument('--lr', type=float, required=False, default=1e-3,
                        help='Learning Rate!')
    parser.add_argument('--drop', type=float, required=False, default=0.1,
                        help='Dropout of model')
    parser.add_argument('--reg', type=float, required=False, default=0,
                        help='Weight decay')
    parser.add_argument('--gamma', type=float, required=False, default=0.95,
                        help='Gamma value for Exponential LR-Decay')
    parser.add_argument('--acc_grad', type=int, required=False, default=1,
                        help='Set to 1 if no gradient accumulation is desired else enter desired value for accumulated batches before gradient step')
    parser.add_argument('--genre', type=bool, required=False, default=False,
                        help='Use genre loss as proxy loss')
    parser.add_argument('--epochs', type=int, required=False, default=100,
                        help='Set maxs number of epochs.')
    parser.add_argument('--window_size', type=int, required=False, default=592,
                        help='Set window size on cqt!')
    parser.add_argument('--local', type=int, required=False, default=False,
                        help='Train on local key estimation')
    parser.add_argument('--gpu', type=int, required=True,
                        help='Set the name of the GPU in the system')
    parser.add_argument('--octaves', type=int, required=False, default=10,
                        help='How many octaves to consider for CQT')
    parser.add_argument('--conv_layers', type=int, required=False, default=1,
                        help='How many convolutional layers per PitchClassNet layer')
    parser.add_argument('--n_filters', type=int, required=False, default=1,
                        help='standard number of filters within PitchClassNet')
    parser.add_argument('--num_layers', type=int, required=False, default=1,
                        help='Number of Layers within PitchClassNet')
    parser.add_argument('--kernel_size', type=int, required=False, default=3,
                        help='Standard kernel size for PitchClassNet')
    parser.add_argument('--tonic_weight', type=float, required=False, default=1.0,
                        help='Loss weight for tonic loss')
    parser.add_argument('--resblock', action="store_true",
                        help='Use Resblocks instead of basic Convs')
    parser.add_argument('--denseblock', action="store_true",
                        help='Use Dense blocks instead of basic Convs')
    parser.add_argument('--hop_length', type=int, required=False, default=0,
                        help='Hop_Length for cqts')
    opt = parser.parse_args()
    
    return opt

if __name__ == "__main__":
    opt = init_parser()
    train(opt)
