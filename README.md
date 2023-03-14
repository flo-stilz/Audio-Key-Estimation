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

This project makes use of the following datasets:

<table>
    <col>
    <col>
    <col>
    <tr>
        <th rowspan=2>Dataset</th>
        <th rowspan=2>Amount of Samples</th>
        <th rowspan=2>Genre Annotations</th>
    </tr>
    <tr>
        <td>GiantSteps Key</td>
        <td>604</td>
        <td>Yes</td>
    </tr>
    <tr>
        <td>Tonality classicalDB</td>
        <td>342</td>
        <td>Yes</td>
    </tr>
    <tr>
        <td>Schubert - Winterreise</td>
        <td>48</td>
        <td>Yes</td>
    </tr>
    <tr>
        <td>Isophonics - Queen</td>
        <td>19</td>
        <td>Yes</td>
    </tr>
    <tr>
        <td>Isophonics - Zweieck</td>
        <td>18</td>
        <td>Yes</td>
    </tr>
    <tr>
        <td>Isophonics - The Beatles</td>
        <td>177</td>
        <td>Yes</td>
    </tr>
    <tr>
        <td>Isophonics - King Carole</td>
        <td>7</td>
        <td>Yes</td>
    </tr>
    <tr>
        <td>GuitarSet</td>
        <td>360</td>
        <td>No</td>
    </tr>
    <tr>
        <td>GTZAN</td>
        <td>837</td>
        <td>No</td>
    </tr>
    <tr>
        <td>McGill Billboard</td>
        <td>734</td>
        <td>No</td>
    </tr>
    <tr>
        <td>FSL10K</td>
        <td>9486</td>
        <td>No</td>
    </tr>
    <tr>
        <td>UltimateSongs</td>
        <td>?</td>
        <td>Yes</td>
    </tr>
    <tr>
        <td>KeyFinder</td>
        <td>841</td>
        <td>No</td>
    </tr>

</table>

Please create a new conda environment via

