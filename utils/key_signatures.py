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
