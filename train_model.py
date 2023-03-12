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
    popular_songs      = DatasetPopularSongsLoader(os.path.join(file_path, "Data/UltimateSongs"))
    mcbillboard       = DatasetMcGillBillboardLoader(os.path.join(file_path, "Data/McGill-Billboard"))
    
    #gmk_train = DatasetGiantStepsMTGKeyLoader(os.path.join(file_path, 'Data/giantsteps-mtg-key-dataset'), data_type="train")
    #gmk_val = DatasetGiantStepsMTGKeyLoader(os.path.join(file_path, 'Data/giantsteps-mtg-key-dataset'), data_type="val")

    # ==================================================================================================================
    # =====  SPLIT DATA INTO A TRAIN- VALIDATION- AND TEST-SET
    # ==================================================================================================================
    # The DatasetLoaderMultiDS includes shuffling of the all passed datasets
    train_data = KeyDataset(genre=opt.genre, opt=opt)
    val_data = KeyDataset(genre=opt.genre, opt=opt)
    
    if opt.debug:
        debug_d = DatasetGiantStepsMTGKeyLoader(os.path.join(file_path, 'Data/giantsteps-mtg-key-dataset'), data_type="debug")
        train_data.import_data(
            #winterreise)
            debug_d)
        val_data.import_data(
            #winterreise)
            debug_d)
    else:
        train_data.import_data(
            giantsteps_mtg_key,
            gtzan,
            key_finder,
            tonality,
            #FSL10K,
            guitar_set,
            #beatles,
            #king_carole,
            #queen,
            #zweieck,
            popular_songs,
            #gmk_train,
            #winterreise,
            #mcbillboard,
            #giantsteps_key,
        )
    
        val_data.import_data(
            winterreise,
            #gmk_val,
            giantsteps_key
            #king_carole,
        )
    if not opt.no_test and not opt.debug:
        # Initialize test datasets seperate to extract results for each individually
        test_beatles = KeyDataset(genre=opt.genre, opt=opt)
        test_beatles.import_data(beatles)
        #test_isophonics = KeyDataset(genre=opt.genre, opt=opt)
        #test_isophonics.import_data(beatles, king_carole, queen, zweieck)
        test_winterreise = KeyDataset(genre=opt.genre, opt=opt)
        test_winterreise.import_data(winterreise)
        test_giantsteps_key = KeyDataset(genre=opt.genre, opt=opt)
        test_giantsteps_key.import_data(giantsteps_key)
        test_mcbillboard = KeyDataset(genre=opt.genre, opt=opt)
        test_mcbillboard.import_data(mcbillboard)

    # ==================================================================================================================
    # =====  Model Training
    # ==================================================================================================================
    '''
    import torch.nn as nn
    count = 0
    genre = []
    keys = np.zeros(len(KEY_SIGNATURE_MAP))
    t = {}
    cosinesim = nn.CosineSimilarity(dim=1)
    for i in np.arange(12):
        t[str(i)] = [-1,0,-1,0]
    for data in train_data:
        #print(data['key_labels'])
        #print(data['tonic_labels'])
        key_id = torch.argmax(cosinesim(data['key_labels'].reshape(1,data['key_labels'].shape[0]), torch.tensor(KEY_SIGNATURE_MAP.numpy()))).item()
        keys[key_id] += 1
        idx = torch.argmax(data['tonic_labels']).item()
        if t[str(idx)][0] == -1:
            t[str(idx)][0] = key_id
            t[str(idx)][1] = 1
        elif t[str(idx)][0] == key_id:
            t[str(idx)][1] += 1
        elif t[str(idx)][2] == -1:
            t[str(idx)][2] = key_id
            t[str(idx)][3] = 1
        elif t[str(idx)][2] == key_id:
            t[str(idx)][3] += 1
        if count==0:
            genre = list(data['genre'])
        else:    
            genre = [sum(i) for i in zip(genre, list(data['genre']) )]
        count+=1
        if count==len(train_data):
            break
    print(count)
    print(genre)
    print(t)
    sys.exit()
    '''
    results = []
    lrs = [3e-4]
    nl = [2]
    kernel_sizes = [7]
    conv_layers = [3]
    number_filters = [4]
    resblocks = [False]
    denseblocks = [False]
    head_layers = [2]
    time_pool_size = [2]
    batch_size = [1]#[1, 2, 4]
    gammas = [0.96]
    
    #g_w = [0.25, 0.1, 0.05, 0.02, 0.01, 0.005] # genre loss weights for final loss function
    g_w = [0.1, 0.5]
    t_w = [1]
    #nf = [4]
    #for hp in range(len(g_w)):
    for bs in range(len(batch_size)):
        for n in range(len(nl)):
            for tw in range(len(t_w)):
                for gw in range(len(g_w)):
                    for res in range(len(resblocks)):
                        for dense in range(len(denseblocks)):
                            for tp in range(len(time_pool_size)):
                                for hl in range(len(head_layers)):
                                    for ker in range(len(kernel_sizes)):
                                        for conv in range(len(conv_layers)):
                                            for nf in range(len(number_filters)):
                                                for ga in range(len(gammas)):
                                                    for hp in range(len(lrs)):
                                                        
                                                        #if nf==0 and conv==0 and ker==0:
                                                        #    continue
                                                        
                                                        genre_loss_weight = g_w[gw]
                                                        opt.genre_weight = genre_loss_weight
                                                        opt.tonic_weight = t_w[tw]
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
                                                        opt.head_layers = head_layers[hl]
                                                        opt.time_pool_size = time_pool_size[tp]
                                                        opt.gamma = gammas[ga]
                                                        #opt.acc_grad = batch_size[bs]
                                                        print("LR: "+str(opt.lr))
                                                        print("Number of Filters: "+str(opt.n_filters))
                                                        print("Kernel size: "+str(opt.kernel_size))
                                                        print("Amount of conv layers: "+str(opt.conv_layers))
                                                        print("Resblocks used: "+str(opt.resblock))
                                                        print("Denseblocks used: "+str(opt.denseblock))
                                                        print("Amount of layers within head: "+str(opt.head_layers))
                                                        print("Tonic Loss weight: "+str(opt.tonic_weight))
                                                        print("Genre Loss weight: "+str(opt.genre_weight))
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
                                                        if not opt.no_test and not opt.debug:
                                                            test_w = DataLoader(test_winterreise, batch_size=opt.batch_size, shuffle=False, drop_last=True, pin_memory=True, num_workers=12)
                                                            test_g = DataLoader(test_giantsteps_key, batch_size=opt.batch_size, shuffle=False, drop_last=True, pin_memory=True, num_workers=12)
                                                            test_b = DataLoader(test_beatles, batch_size=opt.batch_size, shuffle=False, drop_last=True, pin_memory=True, num_workers=12)
                                                            #test_isophonics = DataLoader(test_isophonics, batch_size=opt.batch_size, shuffle=False, drop_last=True, pin_memory=True, num_workers=12)
                                                            test_mc = DataLoader(test_mcbillboard, batch_size=opt.batch_size, shuffle=False, drop_last=True, pin_memory=True, num_workers=12)
                                                        # Initialize model
                                                        os.environ["CUDA_VISIBLE_DEVICES"]=str(opt.gpu)
                                                        
                                                        if not opt.no_semitones:
                                                            shape = opt.octaves*36
                                                        elif opt.no_semitones:
                                                            shape = opt.octaves*12
                                                        else:
                                                            shape = 360
                                                        
                                                        if opt.multi_scale:
                                                            shape1=opt.octaves*36
                                                            shape2=opt.octaves*36
                                                            model = PitchClassNet_Multi(shape1, shape2, 12, opt.num_layers, opt.kernel_size, opt=opt, window_size=opt.window_size, batch_size=opt.batch_size, train_set=train_data, val_set=val_data).double().cuda()
                                                        else:
                                                            model = PitchClassNet(shape, 12, opt.num_layers, opt.kernel_size, opt=opt, window_size = opt.window_size, batch_size=opt.batch_size, train_set=train_data, val_set=val_data).double().cuda()
                                                            #model = TestNet(shape, 12, opt.num_layers, opt.kernel_size, opt=opt, window_size = opt.window_size, batch_size=opt.batch_size, train_set=train_data, val_set=val_data).double().cuda()
                                                            #model = JXC1(shape, 12, opt.num_layers, opt.kernel_size, opt=opt, window_size = opt.window_size, batch_size=opt.batch_size, train_set=train_data, val_set=val_data).double().cuda()
                                                            #model = EquivariantDilatedModel(nf=opt.n_filters, batch_size=opt.batch_size, opt=opt, train_set=train_data, val_set=val_data)
                                                            #model = AttentionPitchClassNetTest(shape, 12, num_layers, kernel_size, opt=opt, window_size = opt.window_size, batch_size=opt.batch_size, train_set=train_data, val_set=val_data).double().cuda()
                                                            #model = TestCNN(180, 12, num_layers, kernel_size=5, opt=opt, window_size=opt.window_size, batch_size=opt.batch_size, train_set=train_data, val_set=val_data).double().cuda()
                                                            #model = EqCNNKey(shape, 12, num_layers, 5, opt=opt, window_size = opt.window_size, batch_size=opt.batch_size, train_set=train_data, val_set=val_data).double().cuda()
                                                        
                                                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                                                        print(device)
                                                        model = model.to(device)
                                                        #print(model)
                                                        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=5, verbose=False, mode="min")
                                                        #best_checkpoint = ModelCheckpoint(monitor='val_accuracy', save_top_k=1, mode="max")
                                                        best_checkpoint = ModelCheckpoint(monitor='val_mirex_score', save_top_k=1, mode="max")
                                                        #best_checkpoint = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode="min")
                                                        
                                                        logger = TensorBoardLogger("Model_logs")
                                                        trainer = pl.Trainer(
                                                            #precision=16,
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
                                                        #result_train = trainer.validate(ckpt_path=final_model_path, dataloaders=train)
                                                        if not opt.no_test and not opt.debug:
                                                            result_winterreise = trainer.validate(ckpt_path=final_model_path, dataloaders=test_w)
                                                            result_giantsteps_key = trainer.validate(ckpt_path=final_model_path, dataloaders=test_g)
                                                            result_beatles = trainer.validate(ckpt_path=final_model_path, dataloaders=test_b)
                                                            result_mcbillboard = trainer.validate(ckpt_path=final_model_path, dataloaders=test_mc)
                                                        #result_isophonics = trainer.validate(ckpt_path=final_model_path, dataloaders=test_isophonics)
                                                        #model.load_state_dict(torch.load(final_model_path)['state_dict'], strict=True)
                                                        #trainer.validate(model, dataloaders=val)
                                                        hp_results = {}
                                                
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
                                                        #hp_results['train_loss'] = result_train[0]['val_loss']
                                                        #hp_results['train_acc'] = result_train[0]['val_accuracy']
                                                        #hp_results['train_mirex'] = result_train[0]['val_mirex_score']
                                                        if not opt.no_test and not opt.debug:
                                                            # winterreise
                                                            hp_results['winter_acc'] = result_winterreise[0]['val_accuracy']
                                                            hp_results['winter_acc_tonic'] = result_winterreise[0]['val_accuracy_tonic']
                                                            hp_results['winter_mirex'] = result_winterreise[0]['val_mirex_score']
                                                            if opt.genre:
                                                                hp_results['winter_acc_genre'] = result_winterreise[0]['val_accuracy_genre']
                                                            hp_results['winter_correct'] = result_winterreise[0]['val_correct']
                                                            hp_results['winter_fifths'] = result_winterreise[0]['val_fifths']
                                                            hp_results['winter_relative'] = result_winterreise[0]['val_relative']
                                                            hp_results['winter_parallel'] = result_winterreise[0]['val_parallel']
                                                            hp_results['winter_other'] = result_winterreise[0]['val_other']
                                                            # giantsteps
                                                            hp_results['giantsteps_acc'] = result_giantsteps_key[0]['val_accuracy']
                                                            hp_results['giantsteps_acc_tonic'] = result_giantsteps_key[0]['val_accuracy_tonic']
                                                            hp_results['giantsteps_mirex'] = result_giantsteps_key[0]['val_mirex_score']
                                                            if opt.genre:
                                                                hp_results['giantsteps_acc_genre'] = result_giantsteps_key[0]['val_accuracy_genre']
                                                            hp_results['giantsteps_correct'] = result_giantsteps_key[0]['val_correct']
                                                            hp_results['giantsteps_fifths'] = result_giantsteps_key[0]['val_fifths']
                                                            hp_results['giantsteps_relative'] = result_giantsteps_key[0]['val_relative']
                                                            hp_results['giantsteps_parallel'] = result_giantsteps_key[0]['val_parallel']
                                                            hp_results['giantsteps_other'] = result_giantsteps_key[0]['val_other']
                                                            # beatles
                                                            hp_results['beatles_acc'] = result_beatles[0]['val_accuracy']
                                                            hp_results['beatles_acc_tonic'] = result_beatles[0]['val_accuracy_tonic']
                                                            hp_results['beatles_mirex'] = result_beatles[0]['val_mirex_score']
                                                            if opt.genre:
                                                                hp_results['beatles_acc_genre'] = result_beatles[0]['val_accuracy_genre']
                                                            hp_results['beatles_correct'] = result_beatles[0]['val_correct']
                                                            hp_results['beatles_fifths'] = result_beatles[0]['val_fifths']
                                                            hp_results['beatles_relative'] = result_beatles[0]['val_relative']
                                                            hp_results['beatles_parallel'] = result_beatles[0]['val_parallel']
                                                            hp_results['beatles_other'] = result_beatles[0]['val_other']
                                                            # Isophonics
                                                            '''
                                                            hp_results['isophonics_acc'] = result_isophonics[0]['val_accuracy']
                                                            hp_results['isophonics_acc_tonic'] = result_isophonics[0]['val_accuracy_tonic']
                                                            hp_results['isophonics_mirex'] = result_isophonics[0]['val_mirex_score']
                                                            if opt.genre:
                                                                hp_results['isophonics_acc_genre'] = result_isophonics[0]['val_accuracy_genre']
                                                            hp_results['isophonics_correct'] = result_isophonics[0]['val_correct']
                                                            hp_results['isophonics_fifths'] = result_isophonics[0]['val_fifths']
                                                            hp_results['isophonics_relative'] = result_isophonics[0]['val_relative']
                                                            hp_results['isophonics_parallel'] = result_isophonics[0]['val_parallel']
                                                            hp_results['isophonics_other'] = result_isophonics[0]['val_other']
                                                            '''
                                                            # McBillboard
                                                            hp_results['billboard_acc'] = result_mcbillboard[0]['val_accuracy']
                                                            hp_results['billboard_acc_tonic'] = result_mcbillboard[0]['val_accuracy_tonic']
                                                            hp_results['billboard_mirex'] = result_mcbillboard[0]['val_mirex_score']
                                                            if opt.genre:
                                                                hp_results['billboard_acc_genre'] = result_mcbillboard[0]['val_accuracy_genre']
                                                            hp_results['billboard_correct'] = result_mcbillboard[0]['val_correct']
                                                            hp_results['billboard_fifths'] = result_mcbillboard[0]['val_fifths']
                                                            hp_results['billboard_relative'] = result_mcbillboard[0]['val_relative']
                                                            hp_results['billboard_parallel'] = result_mcbillboard[0]['val_parallel']
                                                            hp_results['billboard_other'] = result_mcbillboard[0]['val_other']
                                                            
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
                                                        print(result[0]['val_accuracy'])
                                                        results.append(hp_results)
                                                        tuning_value = "Big_Genre_Last"#"Normal_L2_BS"#"Sub_Data_TP_1"#"Big_Data_Normal_L2_Genre_3"
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
    parser.add_argument('--gamma', type=float, required=False, default=0.96,
                        help='Gamma value for Exponential LR-Decay')
    parser.add_argument('--acc_grad', type=int, required=False, default=1,
                        help='Set to 1 if no gradient accumulation is desired else enter desired value for accumulated batches before gradient step')
    parser.add_argument('--epochs', type=int, required=False, default=100,
                        help='Set maxs number of epochs.')
    parser.add_argument('--window_size', type=int, required=False, default=592,
                        help='Set window size on cqt!')
    parser.add_argument('--local', action="store_true",
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
    parser.add_argument('--key_weight', type=float, required=False, default=1.0,
                        help='Loss weight for key loss')
    parser.add_argument('--tonic_weight', type=float, required=False, default=1.0,
                        help='Loss weight for tonic loss')
    parser.add_argument('--genre_weight', type=float, required=False, default=1.0,
                        help='Loss weight for genre loss')
    parser.add_argument('--resblock', action="store_true",
                        help='Use Resblocks instead of basic Convs')
    parser.add_argument('--denseblock', action="store_true",
                        help='Use Dense blocks instead of basic Convs')
    parser.add_argument('--frames', type=int, required=False, default=0,
                        help='Sets Hop_Length for cqts to represent 1 sec in song in exactly that many frames')
    parser.add_argument('--genre', action="store_true",
                        help='Train also on Genre!')
    parser.add_argument('--stay_sixth', action="store_true",
                        help='Immediately downsize to semitone representation in CQT!')
    parser.add_argument('--p2pc_conv', action="store_true",
                        help='Use Conv to downsample from pitch level to pitch class level! If false then use max pool')
    parser.add_argument('--head_layers', type=int, required=False, default=1,
                        help='Number of Conv Layers in classification heads at the end of PitchClassNet')
    parser.add_argument('--loc_window_size', type=int, required=False, default=10,
                        help='Amount of sec. shall be considered by local Key estimation per prediction')
    parser.add_argument('--time_pool_size', type=int, required=False, default=2,
                        help='Pooling size along time dimension for each layer')
    parser.add_argument('--no_semitones', action="store_true",
                        help='Preprocess CQTs in semitones meaning 36(12*3) bins per octave else 12 bins per octave')
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
