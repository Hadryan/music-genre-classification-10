# Music Genre Classification

<h3>TEAM MEMBERS<h3>
 <h6> <h6>

|Names                       |NetID            |UIN           |Role              |
|---                         |---              |---           |---               |
|Saurabh Sangwan            |ssangw2@uic.edu   |658548774     |Time Keeper       |
|Rakshitha Jayarame Gowda   |rjayar4@uic.edu   |667917639     |Scribe            |
|Keshvi Srivastava          |ksriva5@uic.edu   |652825616     |Project Manager   |


<h3>CONTRIBUTIONS<h3>


<h5>Rakshitha Jayarame Gowda<h6>
<h6> <h6>

* Tracking of project tasks and documentation of duties
* Data exploration and studying random forest classification model

<h5>Keshvi Srivastava<h5>
<h6> <h6>

* Succesfully guided and monitored the team as a project manager
* Feature Extraction and studying clustering models

<h5>Saurabh Sangwan<h5>
 <h6> <h6>

* Time management in tasks completion
* Data Collection, preparation, and studying SVM classification model

<h3>INSTRUCTIONS<h3>
 <h6> <h6>


 <b>Software</b> - Jupyter Notebook <br>

 <b>Libraries</b> - pandas, numpy, scipy, plotly, seaborn, librosa. <br>

 <b>Installation</b> -   
    
    pip install pandas==1.0.5 
    pip install librosa==0.6.3

  <b>Datasets</b> -

- final_tracks.csv - contains title, track_id and genre of tracks/audio clips extracted from tracks.csv in FMA metadata dataset by limiting to the 8000 tracks in the FMA small dataset. 

- features.csv - contains the track_id and all the features that were extracted from audio clips in FMA small dataset using python library - 'librosa'.

- merged_data.csv - combined data of final_tracks.csv and features.csv using inner join on track_id.

  

  <b>Python Notebooks</b> -


  <b>Run</b> - `jupyter notebook`
