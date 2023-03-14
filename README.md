# Audio-Key-Estimation

## Introduction

This project is a global key signature estimation for audio files. It additionally predicts the tonic as well as the genre of a music piece. The final architecture is called PitchClassNet and is music transposition equivariant by design.

## Content

The project includes:
- Scraper tool that downloads audio files from Youtube and is also based on a similarity score to save only relevant matches
- Dataset classes for 14 different datasets + additional dataloaders
- Different PitchClassNet architectures
- training script
- fast model evaluation script
- Equivariance testing script to prove transcription equivariance by design

## Setup & Datasets

This project makes use of the following datasets except FSL10K. The following table only shows the sample amounts used: 

<table>
    <col>
    <col>
    <col>
    <col>
    <tr>
        <th rowspan=1>Dataset</th>
        <th rowspan=1>Amount of Samples</th>
        <th rowspan=1>Genre Annotations</th>
        <th rowspan=1>Folder Locations</th>
    </tr>
    <tr>
        <td>GiantstepsMTG Key</td>
        <td>1486</td>
        <td>Yes</td>
        <td>Data/giantsteps-mtg-key-dataset</td>
    </tr>
    <tr>
        <td>GiantSteps Key</td>
        <td>604</td>
        <td>Yes</td>
        <td>Data/giantsteps-key-dataset</td>
    </tr>
    <tr>
        <td>Tonality classicalDB</td>
        <td>342</td>
        <td>Yes</td>
        <td>Data/Tonality</td>
    </tr>
    <tr>
        <td>Schubert - Winterreise</td>
        <td>48</td>
        <td>Yes</td>
        <td>Data/Schubert_Winterreise_Dataset_v1-1</td>
    </tr>
    <tr>
        <td>Isophonics - Queen</td>
        <td>19</td>
        <td>Yes</td>
        <td>Data/Queen_Isophonics</td>
    </tr>
    <tr>
        <td>Isophonics - Zweieck</td>
        <td>18</td>
        <td>Yes</td>
        <td>Data/Zweieck_Isophonics</td>
    </tr>
    <tr>
        <td>Isophonics - The Beatles</td>
        <td>177</td>
        <td>Yes</td>
        <td>Data/Beatles_Isophonics</td>
    </tr>
    <tr>
        <td>Isophonics - King Carole</td>
        <td>7</td>
        <td>Yes</td>
        <td>Data/King_Carole_Isophonics</td>
    </tr>
    <tr>
        <td>GuitarSet</td>
        <td>360</td>
        <td>No</td>
        <td>Data/GuitarSet</td>
    </tr>
    <tr>
        <td>GTZAN</td>
        <td>837</td>
        <td>No</td>
        <td>Data/GTZAN</td>
    </tr>
    <tr>
        <td>McGill Billboard</td>
        <td>734</td>
        <td>No</td>
        <td>Data/McGill-Billboard</td>
    </tr>
    <tr>
        <td>FSL10K</td>
        <td>9486</td>
        <td>No</td>
        <td>Data/FSL10K</td>
    </tr>
    <tr>
        <td>UltimateSongs</td>
        <td>25412</td>
        <td>Yes</td>
        <td>Data/UltimateSongs</td>
    </tr>
    <tr>
        <td>KeyFinder</td>
        <td>841</td>
        <td>No</td>
        <td>Data/KeyFinder</td>
    </tr>

</table>

The following Code is compatible with PyTorch 1.8. Please create a new conda environment via running the following cell after cloning this repository:
<pre lang="shell">conda create --name <env> --file requirements.txt</pre>

Please, also make sure to store the Data in a folder called "Data" in the same main folder than contains this project e.g. main_folder/Audio-Key-Estimation -> main_folder/Data.

To download the data use the attached links to the datasets and if no audio files are contained use the youtube_scraper.py:
<pre lang="shell">python youtube_scraper --source <'song_list.txt'> --destination <'Dataset name'></pre>
Make sure that 

## Training & Evaluation

You can train your model via running the following command:
<pre lang="shell">python train_model.py --gpu <'GPU-NUMBER'></pre>

To evaluate a trained model, look up the lightning_logs version number of the specific model and run the following command:
<pre lang="shell">python eval.py --gpu <'GPU-NUMBER'> --version <'VERSION-NUMBER'></pre>
Note: Make sure that any architecture changes that deviate from default in train_model.py are also entered for eval.py.

When testing the architecture for transposition equivariance, just run the following command:
<pre lang="shell">python equivariance_test.py</pre>

For each of the scripts additional console commands exist which either change input dimensions, architecture designs, learning hyperparameters and so on. You can find the precise descriptions within the respective scripts.

## Architecture
The following figure displays the final model architecture:


## Results


