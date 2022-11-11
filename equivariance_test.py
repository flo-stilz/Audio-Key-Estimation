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
#import sounddevice as sd

# for playing audio:
# sd.play(waveform[0],rate)

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
    parser.add_argument('--octaves', type=int, required=False, default=5,
                        help='How many octaves to consider for CQT')
    parser.add_argument('--conv_layers', type=int, required=False, default=1,
                        help='How many convolutional layers per PitchClassNet layer')
    parser.add_argument('--n_filters', type=int, required=False, default=1,
                        help='standard number of filters within PitchClassNet')
    parser.add_argument('--num_layers', type=int, required=False, default=1,
                        help='Number of layers for PitchClassNet')
    parser.add_argument('--kernel_size', type=int, required=False, default=3,
                        help='Standard kernel size for PitchClassNet')
    parser.add_argument('--custom_cqt', action="store_true",
                        help='Cqt without border values when false else actual song')
    parser.add_argument('--resblock', action="store_true",
                        help='Use Resblocks instead of basic Convs')
    parser.add_argument('--denseblock', action="store_true",
                        help='Use Dense blocks instead of basic Convs')
    opt = parser.parse_args()
    
    return opt

opt = init_parser()

# load song or sound:
file_p = Path(os.path.dirname(os.path.abspath(os.getcwd())))
#file = os.path.join(file_p, "Data/giantsteps-key-dataset/audio")
file = os.path.join(file_p, "Data/GTZAN/genres_original/blues")
#file = os.path.join(file_p, "Data/PopularSongs/Rock/SubA")
#audio_binary = tf.io.read_file("C:/Users/Philipp/Downloads/Robbie-Williams_I-tried-love.wav")
#file_path = tf.io.read_file(file+"/10089.LOFI.mp3")

#file_path = bytes.decode(file_path.numpy()) # convert tf to string
#waveform, rate = torchaudio.load(file+"/10089.LOFI.wav")
waveform, rate = torchaudio.load(file+"/blues.00000.wav")
#waveform, rate = torchaudio.load(file+"/4Am.mp3")
print(waveform.shape[1]/rate)
#waveform = waveform[:,:rate]
print(waveform.shape)
w_length = waveform.shape[1]
print(rate)

def mel_shifting_up(mel, semitones):
    steps = 3*semitones
    mel_shift = mel.clone()
    print(torch.sum(mel[-1]))
    for i in range(mel.shape[0]-1,-1,-1):
        if i==steps-1:
            mel_shift[0:steps] = torch.zeros([steps, mel.shape[1]])
            break
        else:
            mel_shift[i,:] = mel[i-steps,:]
    return mel_shift

def mel_shifting_down(mel, semitones):
    steps = 3*semitones
    mel_shift = mel.clone()
    print(torch.sum(mel[0]))
    for i in range(mel.shape[0]-1):
        if i==mel.shape[0]-steps:
            mel_shift[mel.shape[0]-steps:] = torch.zeros([steps, mel.shape[1]])
            break
        else:
            mel_shift[i,:] = mel[i+steps,:]
    return mel_shift


shift = 1

#waveform_shift = test_pitch_shift_up(waveform, rate, shift)

# Display the different waveforms:
librosa.display.waveshow(waveform.numpy())

time_interval = 1 # means 1 sec
song_length = math.ceil(w_length/rate) # song length in seconds
hop_length = round(rate/22050 * 1000) # hop_length is 100 for sample rate 22050 else adjusted accordingly
window_size = math.ceil(rate/hop_length)
start_time = time.time()

melspectrogram = librosa.cqt(y=waveform.numpy(), sr=rate, hop_length=w_length // 592,# fmin=10,
                             bins_per_octave=12 * 3, n_bins=12 * 3 * opt.octaves)
# 44100Hz / 512(hop_size) => feature rate of ~86.1Hz
print("Mel_Shape: "+str(melspectrogram.shape))
end_time = time.time()-start_time
#melspectrogram = librosa.feature.melspectrogram(y=waveform.numpy(), sr=rate, hop_length=100000)
#print(melspectrogram.shape)
if melspectrogram.shape[0]==2:
    melspectrogram = melspectrogram[0]
else:
    melspectrogram = melspectrogram.reshape(melspectrogram.shape[1], melspectrogram.shape[2])


mel = torch.tensor(melspectrogram)
mel = torch.abs(mel)
mel = torch.log(1 + mel)  # log of the intensity

# most of the time we get 593 in the time-axis because the hop_length has to be an integer.
# Therefore, discard the last column
# TODO: don't discard the last column, but one from the middle! Reason: the last chord gives hints to the tonic!

mel_shift = mel_shifting_up(mel, shift)

def shift_and_stack(mel):
    #out = mel
    shape = opt.octaves*12*3
    model = AttentionPitchClassNet(shape, 12, num_layers=opt.num_layers, kernel_size=opt.kernel_size, opt=opt, window_size=opt.window_size).double().cuda()
    #model = TestNet(shape, 12, num_layers=opt.num_layers, kernel_size=opt.kernel_size, opt=opt, window_size=opt.window_size).double().cuda()
    #model = JXC1(shape, 12, num_layers=opt.num_layers, kernel_size=opt.kernel_size, opt=opt, window_size=opt.window_size).double().cuda()
    #model = EquivariantDilatedModel(4)
    # shift up every semitone up to exactly 1 octave
    for i in range(0,13):
        if i>0:
            mel_shift = mel_shifting_up(mel,i)
        else:
            mel_shift = mel
        
        if mel_shift.shape[1] > opt.window_size:
            mel_shift = mel_shift[:, :opt.window_size]
            
        keys_pred_shift, tonic = model.forward(mel_shift.reshape(1, 1, mel_shift.shape[0], mel_shift.shape[1]).double().cuda())
        
        if i>0:
            out = torch.vstack((keys_pred_shift,out))
        else:
            out = keys_pred_shift
            
    # shift up every semitone down to exactly 1 octave
    for i in range(1,13):
        mel_shift = mel_shifting_down(mel,i)
        
        if mel_shift.shape[1] > opt.window_size:
            mel_shift = mel_shift[:, :opt.window_size]
            
        keys_pred_shift, tonic = model.forward(mel_shift.reshape(1, 1, mel_shift.shape[0], mel_shift.shape[1]).double().cuda())
        
        out = torch.vstack((out,keys_pred_shift))
    
    return out

def display_heat_map(results):
    results = results.detach()
    plt.figure(figsize=(25,12))
    heat_map = sns.heatmap( results, linewidth = 1 , annot = True, xticklabels=["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"], yticklabels=["12","11","10","9","8","7","6","5","4","3","2","1","0","-1","-2","-3","-4","-5","-6","-7","-8","-9","-10","-11","-12"])
    heat_map.set(xlabel='Pitch Classes', ylabel='Semitone Shifts')
    plt.title( "Model Pitch Shift Results" )
    plt.show()

def load_results(name):
    results = torch.load(name,map_location=torch.device('cpu'))
    
    return results

def evaluate():
    name = "Equivariance_Test.pt"
    results = load_results(name)
    display_heat_map(results)
    
def custom_cqt():
    #mel[100, 20:50] = torch.ones(30)
    shape = opt.octaves*3*12
    mel = torch.zeros([shape,592])
    mel[100:150, 20:50] = torch.ones([50,30])
    mel[30:40, 400] = torch.ones([10])*10
    mel[50,320:350] = torch.ones([30])*20
    #mel[200:220, 100:110] = torch.ones([20,10])
    
    mel_shift = mel_shifting_up(mel, shift)
    
    return mel, mel_shift

# file _path to save:
name = "Equivariance_Test.pt"
#does it for 2 octave shifts up
if opt.custom_cqt:
    mel,_ = custom_cqt()
out = shift_and_stack(mel)
torch.save(out,name)
print(end_time)
sys.exit()
fig, ax = plt.subplots()
img = librosa.display.specshow(mel.numpy(), x_axis='time',
                         y_axis='cqt_note', sr=rate, ax=ax)  # TODO: remove min/max # fmin: 63.57
fig.colorbar(img, ax=ax, format='%+2.0f dB')
#ax.axhline(y=104, color='r', linestyle='-')
ax.set(title='Mel-frequency spectrogram')

fig2, ax2 = plt.subplots()
img2 = librosa.display.specshow(mel_shift.numpy(), x_axis='time',
                         y_axis='cqt_note', sr=rate, ax=ax2)  # TODO: remove min/max
fig2.colorbar(img2, ax=ax2, format='%+2.0f dB')
#ax.axhline(y=104, color='r', linestyle='-')
ax2.set(title='Mel-frequency spectrogram with one semitone shift up')

mel = mel.reshape(mel.shape[0], mel.shape[1], 1)
mel_shift = mel_shift.reshape(mel_shift.shape[0], mel_shift.shape[1], 1)

#mel, mel_shift = custom_spectrogram()

if mel.shape[1] > opt.window_size:
    mel = mel[:, :opt.window_size]
    
if mel_shift.shape[1] > opt.window_size:
    mel_shift = mel_shift[:, :opt.window_size]

'''
print(mel.shape)
print(mel_shift.shape)
# Initialize model with random weights:
#model = AlignedCNNModel("elu", [120, 592, 1],d_rate=0.1)
model = AttentionPitchClassNet(180, 12, num_layers=2, kernel_size=5, opt=opt, window_size=opt.window_size).double()
#model = TestCNN(360, 12, 4, 5, opt=opt, window_size=opt.window_size).double()
#model = EqCNNKey(180, 12, num_layers=4, kernel_size=13, opt=opt, window_size=opt.window_size).double()

#model = AttentionPitchNet_Old("elu", [180, 592, 1],d_rate=0.1, num_layers=2)
keys_pred = model.forward(mel.reshape(1, 1, mel.shape[0], mel.shape[1]).double().cuda())

if opt.local:
    print(keys_pred[:,:,0])
else:
    print(keys_pred)

# 1 semitone up
keys_pred_shift = model.forward(mel_shift.reshape(1, 1, mel_shift.shape[0], mel_shift.shape[1]).double().cuda())

if opt.local:
    print(keys_pred_shift[:,:,0])
else:
    print(keys_pred_shift)
'''
sys.exit()


print(tf.cast(keys_pred_shift >= tf.sort(keys_pred_shift)[-7], tf.float32))
#print(tf.round(keys_pred))
print(tf.cast(keys_pred_shift >= 0.1, tf.float32))
print(tf.reduce_sum(keys_pred_shift))
print("Difference: \n")
print(keys_pred_shift-keys_pred)
print(tf.cast(keys_pred>=tf.sort(keys_pred)[-7], tf.float32))
print(tf.cast((keys_pred_shift-keys_pred)>0.0, tf.float32))

'''
keys_pred = model.predict(mel[tf.newaxis, ...])[0]

print(keys_pred)
print(tf.cast(keys_pred >= tf.sort(keys_pred)[-7], tf.float32))
#print(tf.round(keys_pred))
print(tf.cast(keys_pred >= 0.1, tf.float32))
print(tf.reduce_sum(keys_pred))
'''
'''
# 1 octave up
keys_pred = model.predict(mel[tf.newaxis, ...])[0]

print(keys_pred)
print(tf.cast(keys_pred >= tf.sort(keys_pred)[-7], tf.float32))
#print(tf.round(keys_pred))
print(tf.cast(keys_pred >= 0.1, tf.float32))
print(tf.reduce_sum(keys_pred))
'''