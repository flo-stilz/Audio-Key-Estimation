# Test Model Equivariance
import matplotlib.pyplot as plt

from KeyDataset import *
from models import *
import os
from pathlib import Path
import librosa.display
import librosa
import torchaudio
import sys
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def init_parser():
    # Settings
    parser = argparse.ArgumentParser(
        description='Hyperparameters for Publication Year Prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--batch_size', type=int, required=False, default=8,
                        help='Batch size!')
    parser.add_argument('--lr', type=float, required=False, default=1e-3,
                        help='Learning Rate!')
    parser.add_argument('--drop', type=float, required=False, default=0.1,
                        help='Dropout of model')
    parser.add_argument('--reg', type=float, required=False, default=0,
                        help='Weight decay')
    parser.add_argument('--gamma', type=float, required=False, default=1,
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
                        help='How many octaves to consider for CQT! NOTE: actual amount of octaves = octaves-2 due to padding so if 8 octaves are desired enter 10')
    parser.add_argument('--conv_layers', type=int, required=False, default=3,
                        help='How many convolutional layers per PitchClassNet layer')
    parser.add_argument('--n_filters', type=int, required=False, default=4,
                        help='standard number of filters within PitchClassNet')
    parser.add_argument('--num_layers', type=int, required=False, default=2,
                        help='Number of layers for PitchClassNet')
    parser.add_argument('--kernel_size', type=int, required=False, default=7,
                        help='Standard kernel size for PitchClassNet')
    parser.add_argument('--custom_cqt', action="store_true",
                        help='Cqt without border values when false else actual song')
    parser.add_argument('--resblock', action="store_true",
                        help='Use Resblocks instead of basic Convs')
    parser.add_argument('--denseblock', action="store_true",
                        help='Use Dense blocks instead of basic Convs')
    parser.add_argument('--frames', type=int, required=False, default=5,
                        help='Sets Hop_Length for CQTs to represent each sec. in song in the amount of frames')
    parser.add_argument('--stay_sixth', action="store_true",
                        help='Immediately downsize to semitone representation in CQT!')
    parser.add_argument('--p2pc_conv', action="store_true",
                        help='Use Conv to downsample from pitch level to pitch class level! If false then use max pool')
    parser.add_argument('--head_layers', type=int, required=False, default=2,
                        help='Number of Conv Layers in classification heads at the end of PitchClassNet')
    parser.add_argument('--cqt_with_border', action="store_true",
                        help='When creating custom cqt decides on whether border points contained!')
    parser.add_argument('--loc_window_size', type=int, required=False, default=10,
                        help='Amount of sec. shall we considered by local Key estimation per prediction')
    parser.add_argument('--time_pool_size', type=int, required=False, default=2,
                        help='Pooling size along time dimension for each layer')
    parser.add_argument('--only_semitones', action="store_true",
                        help='Preprocess CQTs in semitones meaning 12 bins per octave else 36(12*3) bins per octave')
    parser.add_argument('--multi_scale', action="store_true",
                        help='Preprocess CQTs in semitones and only tones and run two models and merge predictions')
    parser.add_argument('--linear_reg_multi', action="store_true",
                        help='Use linear regression for output combination of two scale models')
    parser.add_argument('--use_cos', action="store_true",
                        help='Use Cosine Similarity as additional loss term for Key Estimation')
    parser.add_argument('--pc2p_mem', action="store_true",
                        help='Use Memory-efficient variant for upsampling pitc class wise features to pitch wise features')
    parser.add_argument('--no_ckpt', action="store_false",
                        help='Do save best model!')
    parser.add_argument('--max_pool', action="store_true",
                        help='Perform Global MaxPooling')
    opt = parser.parse_args()
    
    return opt

opt = init_parser()

# load song or sound:
# Different examples are commented out
file_p = Path(os.path.dirname(os.path.abspath(os.getcwd())))
#file = os.path.join(file_p, "Data/giantsteps-key-dataset/audio")
file = os.path.join(file_p, "Data/GTZAN/genres_original/blues")
#file = os.path.join(file_p, "Data/UltimateSongs/Rock/SubA")
#file = os.path.join(file_p, "Data/Queen_Isophonics")
#file_path = tf.io.read_file(file+"/10089.LOFI.mp3")

#file_path = bytes.decode(file_path.numpy()) # convert tf to string
#waveform, rate = torchaudio.load(file+"/10089.LOFI.wav")
waveform, rate = torchaudio.load(file+"/blues.00000.wav")
#waveform, rate = torchaudio.load(file+"/4Am.mp3")
#waveform, rate = torchaudio.load(file+"/A_Farewell_To_Kings.mp3")
#waveform, rate = torchaudio.load(file+"/A_Firm_Kick.mp3")
#waveform, rate = torchaudio.load(file+"/Queen_Kind_Of_Magic.mp3")
print(waveform.shape[1]/rate)
print(waveform.shape)
w_length = waveform.shape[1]
print(rate)
waveform = waveform[0]
    
print("Adjusted Wave: "+str(waveform.shape))

def mel_shifting_up(mel, semitones):
    # to adjust for that semitones are split into 3 bins
    # if mel is on semitone level then remove 3
    steps = 3*semitones
    mel_shift = mel.clone()
    for i in range(mel.shape[0]-1,-1,-1):
        if i==steps-1:
            mel_shift[0:steps] = torch.zeros([steps, mel.shape[1]])
            break
        else:
            mel_shift[i,:] = mel[i-steps,:]
    return mel_shift

def mel_shifting_down(mel, semitones):
    # to adjust for that semitones are split into 3 bins
    # if mel is on semitone level then remove 3
    steps = 3*semitones
    mel_shift = mel.clone()
    for i in range(mel.shape[0]-1):
        if i==mel.shape[0]-steps:
            mel_shift[mel.shape[0]-steps:] = torch.zeros([steps, mel.shape[1]])
            break
        else:
            mel_shift[i,:] = mel[i+steps,:]
    return mel_shift


shift = 1

# Display the different waveforms:
librosa.display.waveshow(waveform.numpy(),sr=rate)

if not opt.custom_cqt:
    time_interval = 1 # means 1 sec
    song_length = math.ceil(w_length/rate) # song length in seconds
    hop_length = round(rate/(opt.frames if opt.frames>0 else 1)) # hop_length is including sample rate so that it yields exactly the desired amount of frames per second in the song
    window_size = math.ceil(rate/hop_length)
    
    if not opt.only_semitones:
        melspectrogram = librosa.cqt(y=waveform.numpy(), sr=rate, hop_length=(w_length // 592 if opt.frames==0 else hop_length),
                                     bins_per_octave=12 * 3, n_bins=12 * 3 * (opt.octaves-2))
    else:
        melspectrogram = librosa.cqt(y=waveform.numpy(), sr=rate, hop_length=(w_length // 592 if opt.frames==0 else hop_length),
                                     bins_per_octave=12 * 1, n_bins=12 * 1 * (opt.octaves-2))
    
    mel = torch.tensor(melspectrogram)
    mel = torch.abs(mel)
    mel = torch.log(1 + mel)  # log of the intensity
    mel_shift = mel_shifting_up(mel, shift)

def shift_and_stack(mel):
    # pad input!
    mel = torch.cat((mel,torch.zeros(12*3, mel.shape[1])))
    mel = torch.cat((torch.zeros(12*3, mel.shape[1]), mel))

    shape = opt.octaves*12*3
    model = PitchClassNet(shape, 12, num_layers=opt.num_layers, kernel_size=opt.kernel_size, opt=opt, window_size=opt.window_size).double().cuda()
    # shift up every semitone up to exactly 1 octave
    for i in range(0,13):
        if i>0:
            mel_shift = mel_shifting_up(mel,i)
        else:
            mel_shift = mel
        if opt.frames==0:
            if mel_shift.shape[1] > opt.window_size:
                mel_shift = mel_shift[:, :opt.window_size]
        keys_pred_shift, tonic = model.forward(mel_shift.reshape(1, 1, mel_shift.shape[0], mel_shift.shape[1]).double().cuda(), torch.tensor(mel_shift.shape[1]).reshape(1,1).cuda() if opt.frames>0 else None)
        
        if i>0:
            out = torch.vstack((keys_pred_shift,out))
        else:
            out = keys_pred_shift
            
    # shift up every semitone down to exactly 1 octave
    for i in range(1,13):
        mel_shift = mel_shifting_down(mel,i)
        if opt.frames==0:
            if mel_shift.shape[1] > opt.window_size:
                mel_shift = mel_shift[:, :opt.window_size]
        keys_pred_shift, tonic = model.forward(mel_shift.reshape(1, 1, mel_shift.shape[0], mel_shift.shape[1]).double().cuda(), torch.tensor(mel_shift.shape[1]).reshape(1,1).cuda() if opt.frames>0 else None)
        
        out = torch.vstack((out,keys_pred_shift))
    
    return out

def rotateArray(arr, n, d):
    temp = []
    i = 0
    while (i < d):
        temp.append(arr[i])
        i = i + 1
    i = 0
    while (d < n):
        arr[i] = arr[d]
        i = i + 1
        d = d + 1
    arr[:] = arr[: i] + temp
    return arr

def rightRotate(lists, num):
    output_list = []
 
    # Will add values from n to the new list
    for item in range(len(lists) - num, len(lists)):
        output_list.append(lists[item])
 
    # Will add the values before
    # n to the end of new list
    for item in range(0, len(lists) - num):
        output_list.append(lists[item])
 
    return output_list

def display_heat_map(results):
    results = results.detach()
    plt.figure(figsize=(25,12))
    heat_map = sns.heatmap( results, linewidth = 1 , annot = True, xticklabels=["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"], yticklabels=["12","11","10","9","8","7","6","5","4","3","2","1","0","-1","-2","-3","-4","-5","-6","-7","-8","-9","-10","-11","-12"])
    heat_map.set(xlabel='Pitch Classes', ylabel='Semitone Shifts')
    plt.title( "Model Pitch Shift Results" )
    plt.show()
    
def display_heat_map_adj(results):
    results = results.detach()
    for i in range(12):
        results[i] = torch.tensor(rotateArray(list(results[i]), len(results[i]), 12-i))
    for i in range(13, 25):
        results[i] = torch.tensor(rightRotate(list(results[i]), i-12))
    plt.figure(figsize=(25,12))
    heat_map = sns.heatmap( results, linewidth = 1 , annot = True, xticklabels=["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"], yticklabels=["12","11","10","9","8","7","6","5","4","3","2","1","0","-1","-2","-3","-4","-5","-6","-7","-8","-9","-10","-11","-12"])
    heat_map.set(xlabel='Pitch Classes (Reversed Transposition on Output)', ylabel='Semitone Shifts')
    plt.title( "Model Pitch Shift Results" )
    plt.show()

def load_results(name):
    results = torch.load(name,map_location=torch.device('cpu'))
    
    return results

def evaluate():
    name = "Equivariance_Test.pt"
    results = load_results(name)
    display_heat_map(results)
    display_heat_map_adj(results)
    
def custom_cqt(with_border=True):
    shape = opt.octaves*3*12
    mel = torch.zeros([shape,592])
    mel[100:150, 20:50] = torch.ones([50,30])
    if with_border:
        mel[30:40, 400] = torch.ones([10])*10
        mel[10:15, 200] = torch.ones([5])*8
    mel[50,320:350] = torch.ones([30])*20
    
    mel_shift = mel_shifting_up(mel, shift)
    
    return mel, mel_shift

# file _path to save:
name = "Equivariance_Test.pt"
#does it for 1 octave shift up
if opt.custom_cqt:
    mel,_ = custom_cqt(opt.cqt_with_border)
out = shift_and_stack(mel)
torch.save(out,name)
fig, ax = plt.subplots()
img = librosa.display.specshow(mel.numpy(),#librosa.amplitude_to_db(mel.numpy(),ref=np.max),
                               x_axis='time',
                         y_axis='cqt_note', sr=rate, ax=ax, hop_length=hop_length, bins_per_octave=12*3)
fig.colorbar(img, ax=ax, format='%+2.1f dB')
ax.set(title='Mel-frequency spectrogram')

plt.savefig('CQT.png')
