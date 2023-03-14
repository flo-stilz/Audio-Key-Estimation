# Audio-Key-Estimation
This project is a global key signature estimation for audio files. It additionally predicts the tonic as well as the genre of a music piece.

##Introduction and Content

The project includes:
- Scraper tool that downloads audio files from Youtube and is also based on a similarity score to save only relevant matches
- Dataset classes for 14 different datasets + additional dataloaders
- Different PitchClassNet architectures
- training script
- fast model evaluation script
- Equivariance testing script to prove transcription equivariance by design

