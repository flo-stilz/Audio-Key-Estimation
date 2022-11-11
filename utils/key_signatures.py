import tensorflow as tf
import torch as nn
import torch

"""
An array that contains tensors for each key signature. E.g. C major would be encoded as:
   [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1] where 1 means that key is included in the said key signature.
The array corresponds to the 12 keys in an octave ordered by increasing frequency. This means C/B# < C#/Db < D < ...

The choice to predict the 12 keys from a key signature instead of assigning each key signature a number
    (to one-hot encode) was so that the network can also predict non-diatonic keys
    e.g. minor pentatonic (1b345b7), major pentatonic (12356).

The following arrays correspond to the keys within a key signature.
            [C,  C#, D, D#, E,  F,  F#, G, G#, A, A#, B ]
            [B#, Db, D, Eb, Fb, E#, Gb, G, Ab, A, Bb, Cb]

"""
KEY_SIGNATURE_MAP = tf.convert_to_tensor([
    tf.cast([0,  1,  0, 1,  1,  0,  1,  0, 1,  0, 1,  1], tf.float32),  # Cb_major, Ab_minor
    tf.cast([0,  1,  0, 1,  0,  1,  1,  0, 1,  0, 1,  1], tf.float32),  # Gb_major, Eb_minor
    tf.cast([1,  1,  0, 1,  0,  1,  1,  0, 1,  0, 1,  0], tf.float32),  # Db_major, Bb_minor
    tf.cast([1,  1,  0, 1,  0,  1,  0,  1, 1,  0, 1,  0], tf.float32),  # Ab_major, F_minor
    tf.cast([1,  0,  1, 1,  0,  1,  0,  1, 1,  0, 1,  0], tf.float32),  # Eb_major, C_minor
    tf.cast([1,  0,  1, 1,  0,  1,  0,  1, 0,  1, 1,  0], tf.float32),  # Bb_major, G_minor
    tf.cast([1,  0,  1, 0,  1,  1,  0,  1, 0,  1, 1,  0], tf.float32),  # F_major,  D_minor
    tf.cast([1,  0,  1, 0,  1,  1,  0,  1, 0,  1, 0,  1], tf.float32),  # C_major,  A_minor
    tf.cast([1,  0,  1, 0,  1,  0,  1,  1, 0,  1, 0,  1], tf.float32),  # G_major,  E_minor
    tf.cast([0,  1,  1, 0,  1,  0,  1,  1, 0,  1, 0,  1], tf.float32),  # D_major,  B_minor
    tf.cast([0,  1,  1, 0,  1,  0,  1,  0, 1,  1, 0,  1], tf.float32),  # A_major,  F#_minor
    tf.cast([0,  1,  0, 1,  1,  0,  1,  0, 1,  1, 0,  1], tf.float32),  # E_major,  C#_minor
    tf.cast([0,  1,  0, 1,  1,  0,  1,  0, 1,  0, 1,  1], tf.float32),  # B_major,  G#_minor
    tf.cast([0,  1,  0, 1,  0,  1,  1,  0, 1,  0, 1,  1], tf.float32),  # F#_major, D#_minor
    tf.cast([1,  1,  0, 1,  0,  1,  1,  0, 1,  0, 1,  0], tf.float32),  # C#_major, A#_minor
    # THEORETICAL KEYS:
    tf.cast([0,  1,  1, 0,  1,  0,  1,  1, 0,  1, 0,  1], tf.float32),  # Cb minor, direct enharmonic equivalent: B minor
    tf.cast([0,  1,  0, 1,  1,  0,  1,  0, 1,  1, 0,  1], tf.float32),  # Db minor, direct enharmonic equivalent: C# minor
    tf.cast([0,  1,  1, 0,  1,  0,  1,  0, 1,  1, 0,  1], tf.float32),  # Gb minor, direct enharmonic equivalent: F# minor
    tf.cast([1,  0,  1, 1,  0,  1,  0,  1, 1,  0, 1,  0], tf.float32),  # D# major, direct enharmonic equivalent: Eb major
    tf.cast([1,  1,  0, 1,  0,  1,  0,  1, 1,  0, 1,  0], tf.float32),  # G# major, direct enharmonic equivalent: Ab major
    tf.cast([1,  0,  1, 1,  0,  1,  0,  1, 0,  1, 1,  0], tf.float32),  # A# major, direct enharmonic equivalent: Bb major
])

    
def eval_mode(pred_keys,pred_tonic):
    # mode == 1 --> major
    # mode == 0 --> minor
    
    # no theoretical keys included so far
    
    if pred_keys.all()==torch.tensor([0,  1,  0, 1,  1,  0,  1,  0, 1,  0, 1,  1]).all():
        if pred_tonic==torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]): # Cb_major
            mode = 1
        elif pred_tonic==torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]): # Ab_minor
            mode = 0
    
    elif pred_keys.all()==torch.tensor([0,  1,  0, 1,  0,  1,  1,  0, 1,  0, 1,  1]).all():
        if pred_tonic==torch.tensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]): # Gb_major
            mode = 1
        elif pred_tonic==torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]): # Eb_minor
            mode = 0
        
    elif pred_keys.all()==torch.tensor([1,  1,  0, 1,  0,  1,  1,  0, 1,  0, 1,  0]).all():
        if pred_tonic==torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]): # Db_major
            mode = 1
        elif pred_tonic==torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]): # Bb_minor
            mode = 0
        
    elif pred_keys.all()==torch.tensor([1,  1,  0, 1,  0,  1,  0,  1, 1,  0, 1,  0]).all():
        if pred_tonic==torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]): # Ab_major
            mode = 1
        elif pred_tonic==torch.tensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]): # F_minor
            mode = 0
        
    elif pred_keys.all()==torch.tensor([1,  0,  1, 1,  0,  1,  0,  1, 1,  0, 1,  0]).all():
        if pred_tonic==torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]): # Eb_major
            mode = 1
        elif pred_tonic==torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]): # C_minor
            mode = 0
            
    elif pred_keys.all()==torch.tensor([1,  0,  1, 1,  0,  1,  0,  1, 0,  1, 1,  0]).all():
        if pred_tonic==torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]): # Bb_major
            mode = 1
        elif pred_tonic==torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]): # G_minor
            mode = 0
        
    elif pred_keys.all()==torch.tensor([1,  0,  1, 0,  1,  1,  0,  1, 0,  1, 1,  0]).all():
        if pred_tonic==torch.tensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]): # F_major
            mode = 1
        elif pred_tonic==torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]): # D_minor
            mode = 0
            
    elif pred_keys.all()==torch.tensor([1,  0,  1, 0,  1,  1,  0,  1, 0,  1, 0,  1]).all():
        if pred_tonic==torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]): # C_major
            mode = 1
        elif pred_tonic==torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]): # A_minor
            mode = 0
    
    elif pred_keys.all()==torch.tensor([1,  0,  1, 0,  1,  0,  1,  1, 0,  1, 0,  1]).all():
        if pred_tonic==torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]): # G_major
            mode = 1
        elif pred_tonic==torch.tensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]): # E_minor
            mode = 0
            
    elif pred_keys.all()==torch.tensor([0,  1,  1, 0,  1,  0,  1,  1, 0,  1, 0,  1]).all():
        if pred_tonic==torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]): # D_major
            mode = 1
        elif pred_tonic==torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]): # B_minor
            mode = 0
            
    elif pred_keys.all()==torch.tensor([0,  1,  1, 0,  1,  0,  1,  0, 1,  1, 0,  1]).all():
        if pred_tonic==torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]): # A_major
            mode = 1
        elif pred_tonic==torch.tensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]): # F#_minor
            mode = 0
            
    elif pred_keys.all()==torch.tensor([0,  1,  0, 1,  1,  0,  1,  0, 1,  1, 0,  1]).all():
        if pred_tonic==torch.tensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]): # E_major
            mode = 1
        elif pred_tonic==torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]): # C#_minor
            mode = 0
            
    elif pred_keys.all()==torch.tensor([0,  1,  0, 1,  1,  0,  1,  0, 1,  0, 1,  1]).all():
        if pred_tonic==torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]): # B_major
            mode = 1
        elif pred_tonic==torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]): # G#_minor
            mode = 0
            
    elif pred_keys.all()==torch.tensor([0,  1,  0, 1,  0,  1,  1,  0, 1,  0, 1,  1]).all():
        if pred_tonic==torch.tensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]): # F#_major
            mode = 1
        elif pred_tonic==torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]): # D#_minor
            mode = 0
            
    elif pred_keys.all()==torch.tensor([1,  1,  0, 1,  0,  1,  1,  0, 1,  0, 1,  0]).all():
        if pred_tonic==torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]): # C#_major
            mode = 1
        elif pred_tonic==torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]): # A#_minor
            mode = 0
        
        
    
    return mode

# import tensorflow as tf
# import numpy as np
#
# from enum import IntEnum
#
# class Key(IntEnum):
#     C, BIS = 0, 0
#     CIS, DB = 1, 1
#     D = 2
#     DIS, EB = 3, 3
#     E, FB = 4, 4
#     F, EIS = 5, 5
#     FIS, GB = 6, 6
#     G = 7
#     GIS, AB = 8, 8
#     A = 9
#     AIS, BB = 10, 10
#     B, CB = 11, 11
#     # double sharps, double flats
#     BBB = A
#     EBB = D
#     CX = D
#     FX = G
#     GX = A
#
#
# def generate_key_tensor(*args):
#     keys = np.zeros(12)
#     for key in args:
#         keys[int(key)] = 1
#     return tf.convert_to_tensor(keys, dtype=tf.float32)
#
#
# """
# An array that contains tensors for each key signature. E.g. C major would be encoded as:
#    [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1] where 1 means that key is included in the said key signature.
# The array corresponds to the 12 keys in an octave ordered by increasing frequency. This means C/B# < C#/Db < D < ...
#
# The choice to predict the 12 keys from a key signature instead of assigning each key signature a number
#     (to one-hot encode) was so that the network can also predict non-diatonic keys
#     e.g. minor pentatonic (1b345b7), major pentatonic (12356).
# """
# KEY_SIGNATURE_MAP = [
#     generate_key_tensor(Key.CB,  Key.DB,  Key.EB,  Key.FB,  Key.GB,  Key.AB,  Key.BB),   # Cb_major, Ab_minor
#     generate_key_tensor(Key.CB,  Key.DB,  Key.EB,  Key.F,   Key.GB,  Key.AB,  Key.BB),   # Gb_major, Eb_minor
#     generate_key_tensor(Key.C,   Key.DB,  Key.EB,  Key.F,   Key.GB,  Key.AB,  Key.BB),   # Db_major, Bb_minor
#     generate_key_tensor(Key.C,   Key.DB,  Key.EB,  Key.F,   Key.G,   Key.AB,  Key.BB),   # Ab_major, F_minor
#     generate_key_tensor(Key.C,   Key.D,   Key.EB,  Key.F,   Key.G,   Key.AB,  Key.BB),   # Eb_major, C_minor
#     generate_key_tensor(Key.C,   Key.D,   Key.EB,  Key.F,   Key.G,   Key.A,   Key.BB),   # Bb_major, G_minor
#     generate_key_tensor(Key.C,   Key.D,   Key.E,   Key.F,   Key.G,   Key.A,   Key.BB),   # F_major,  D_minor
#     generate_key_tensor(Key.C,   Key.D,   Key.E,   Key.F,   Key.G,   Key.A,   Key.B),    # C_major,  A_minor
#     generate_key_tensor(Key.C,   Key.D,   Key.E,   Key.FIS, Key.G,   Key.A,   Key.B),    # G_major,  E_minor
#     generate_key_tensor(Key.CIS, Key.D,   Key.E,   Key.FIS, Key.G,   Key.A,   Key.B),    # D_major,  B_minor
#     generate_key_tensor(Key.CIS, Key.D,   Key.E,   Key.FIS, Key.GIS, Key.A,   Key.B),    # A_major,  F#_minor
#     generate_key_tensor(Key.CIS, Key.DIS, Key.E,   Key.FIS, Key.GIS, Key.A,   Key.B),    # E_major,  C#_minor
#     generate_key_tensor(Key.CIS, Key.DIS, Key.E,   Key.FIS, Key.GIS, Key.AIS, Key.B),    # B_major,  G#_minor
#     generate_key_tensor(Key.CIS, Key.DIS, Key.EIS, Key.FIS, Key.GIS, Key.AIS, Key.B),    # F#_major, D#_minor
#     generate_key_tensor(Key.CIS, Key.DIS, Key.EIS, Key.FIS, Key.GIS, Key.AIS, Key.BIS),  # C#_major, A#_minor
#     # there are also theoretically possible major key signatures e.g. G# major, which are non diatonic because
#     #   they use the same key twice: G + G#
#
#     # THEORETICAL KEYS
#     # Db minor,  which is equal to its direct enharmonic equivalent: C# minor
#     generate_key_tensor(Key.CB,  Key.DB,  Key.EB,  Key.FB,  Key.GB,  Key.AB,  Key.BBB),
#     # Gb minor,  which is equal to its direct enharmonic equivalent: F# minor
#     generate_key_tensor(Key.CB,  Key.DB,  Key.EBB, Key.FB,  Key.GB,  Key.AB,  Key.BBB),
#     # D# major,  which is equal to its direct enharmonic equivalent: Eb major
#     generate_key_tensor(Key.CX,  Key.DIS, Key.EIS, Key.FX,  Key.GIS, Key.AIS, Key.BIS),
#     # G# major,  which is equal to its direct enharmonic equivalent: Ab major
#     generate_key_tensor(Key.CIS, Key.DIS, Key.EIS, Key.FX,  Key.GIS, Key.AIS, Key.BIS),
#     # A# major,  which is equal to its direct enharmonic equivalent: Bb major
#     generate_key_tensor(Key.CX,  Key.DIS, Key.EIS, Key.FX,  Key.GX,  Key.AIS, Key.BIS),
# ]
