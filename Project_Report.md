# Final Project 
## Course - CS 418 (Introduction to Data Science)
## Term - Fall 2020

## Team - 
- Saurabh Sangwan
- Rakshitha Jayarame Gowda
- Keshvi Srivastava

## Topic - Music Genre Classification and Similarity Identification

### 1. Introduction
In this project we aim to compare machine learning algorithms 
in their ability to automatically classify a musical genre to a song
clip/snippet. We will also use clustering techniques to 
analyze the similarities between different genres.

### 2. Dataset
The dataset we used is **FMA : a dataset for music analysis**[1]. The original dataset has 106,574 tracks. 
For our project we used a small version of this dataset which has 8,000 tracks of 30 seconds each, 
these tracks categorise into 8 balanced genres. 
In addition to these audio tracks, it also has per track metadata such as ID, title, artist, genres, tags and play counts, 
for all 106,574 tracks.

### 3. Feature Selection
Machine learning typically requires each data example to be represented numerically, often as a vector of relevant information 
or ‘feature vector. For this, we used librosa[2] -  a python package for music and audio analysis 
to extract features from the audio clips. This data includes information about the sound frequencies, bandwidth and rhythm. 

The following features were extracted from the audio clip for each song.

#### 3.1. MFCCs (timbral texture feature) - 

In sound processing, the mel-frequency cepstrum (MFC) is a representation of the short-term power spectrum of a sound. 
While Mel-frequency cepstral coefficients (MFCCs) are coefficients that collectively make up an MFC.
MFCCs are used in music genre classification to detect timbre , 
which can be defined as the ‘quality’ or ‘colour’ of a sound.
For this project, we extracted 20 MFCCs and further took their average across 
the MFCC sequence for each MFCC.

#### 3.2. Zero Crossing Rate (timbral texture feature) - 

The ZCR of a signal is defined as the rate at which a signal changes from pos- itive to negative or negative to positive.
In the context of audio, this refers to the number of times the amplitude of the signal passes through a value of zero, within a given time interval.
Like MFCCs, ZCR is another feature that can be used to understand the timbre of an audio file.

#### 3.3. Chroma Features (pitch content feature) -
Chroma-based features concern the musical notes that are played throughout a song.
For our project we extracted different chroma based features across 12 chroma bins and took their average - 

- chroma_cqt - Constant-Q chromagram
- chroma_cens - Computes the chroma variant “Chroma Energy Normalized” (CENS)
- chroma_stft - Compute a chromagram from a waveform or power spectrogram.

#### 3.4. Spectral Spread (timbral texture feature) - 
Spectral low-level features aim at describing the structure of
(frame or) sound spectra using a single quantity.
Different spectral features extracted - 

- Spectral Centroid (spectral_centroid)
- Spectral Bandwidth (spectral_bandwidth)
- Spectral Contrast (spectral_contrast)
- Spectral roll-off frequency (spectral_rolloff)

#### 3.5. Other Features -

- rms - root-mean-square value for each frame, either from the audio samples.
- tonal centroid features - tonnetz.

### 4. Data Exploration

#### 4.2 Feature Analysis

![](images/feature_scores.png)


### 5. Classification Algorithms

#### 5.1 Support-Vector Machine



Note - Values for feature extraction 

  1.  Sampling rate(the number of samples per second of audio) - 22050 Hz. 
  2.  Hop length (number of samples between successive frames for different features) - 512.
  3.  Frame Length (Length of the frame over which to compute different features) - 2048.

### 6. Clustering Inferences

One of the our interesting research was to identify overlapping of genres. We approached this problem using Clustering. We analysed which features that would contirbute to the model and in hand identify which genres would overlap or disappear altogether.

To begin with, we identified the important features from their F-score. Two of the most informative features were chroma_cens and chroma_cqt. For our study we calculated the averaged values of various features such as - chroma_cens_avg, chroma_cqt_avg, chroma_stft_avg, and spectral_contrast_avg. We then checked the correleation between different features. From this study we analysed which mfcc features could be dropped to satisfy model assumptions of no collinearity between the features. After this pre processing, we fed the data to our K-means model and analysed which features stood out, overlapped and/or dissappeared.

1. Feature Selection:

Our above tables on F-score helps identify the important features to be considered. Apart from this, we also need to identify the relation between the features. To do so we check the correlation between all features. While, most were non-related a few stood out with a pearson correlation coefficient greater than 0.3. which we dropped. These included - ['mfcc7', 'mfcc5', 'mfcc19', 'mfcc3', 'mfcc4']. The correlation table is given as below:

![](images/pearson_correlation.png)

2. Studying the cluster features:

We analysed how many clusters were actually required for the data itself. We checked the SSE value for various number of clusters ( the data ideally should move towards 8 clusters as we have 8 balanced genres ). We plot the SSE value against the number of clusters and get the following plot:

![](images/ssevsk.png)

The plot shows that having 8 clusters definitey minimises our error. Which is exactly as it would because there are 8 possible clusters (genre) to categorise our data into. We also compared the sillhoette coeffficient against the number of clusters as shown below. Again, the silhouette coefficient is seen to be the lowest at the higher cluster values (k=8). This means our data is clustering correctly or very close to its ideal value.

![](images/scvsk.png)

3. Cluster Confusion Matrix:

Our most interesting study was to see how some genres may be completely overshadowed by their counterparts. We observed this when plotting the confusion matrix of our clustering model. As one can see, the value for Electronic, Experimental and International was zero. This would mean that all songs classified into these genres can easily be represented in other genres too, maybe even more correctly. This would constitute to our theory of overlapping genres which is what makes music genre classification a very difficult task generally.

![](images/kconfmatrix.png)

4. Conclusion:

It is very interesting to note how clustering can help identify special cases even when not used for the purpose of classification specifically. Our k-means model if used for identifying which cluster test data may fall into is definitely not a good model. It gives an accuracy of approximately 7%, which although better than random case probability, is much lower than our above classification models. However, our k-means model helps identify various other parts of the genre classification. Using clustering, we were able to identify which genres may overlap and overshadow the others. It would also help to note that there are more features which may help in improving the clustering of these overlapped genres. Features such as danceability, pitch, etc can also help but were not used here due to unavailability for the studied tracks.

### References
1. https://www.loc.gov/item/2018655052
2. https://medium.com/latinxinai/discovering-descriptive-music-genres-using-k-means-clustering-d19bdea5e443
3. https://towardsdatascience.com/discovering-similarities-across-my-spotify-music-using-data-clustering-and-visualization-52b58e6f547b
4. https://towardsdatascience.com/a-feature-selection-tool-for-machine-learning-in-python-b64dd23710f0
