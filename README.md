# Steady-State Visual Evoked Potential (SSVEP) Task Using Canonical Correlation Analysis (CCA)

This project was conducted during the _Ibn Sina Neurotech Summer School_, where we collected and analyzed SSVEP data using the Unicorn EEG headset. Subjects focused on one of four squares on a screen, each flickering at a different frequency. When observing such flickering, the human brain generates brainwaves with frequencies similar to the observed— fascinating right?!

We applied Canonical Correlation Analysis (CCA) to classify the frequencies based on the EEG signals. The aim was to enhance control applications like brain-computer interfaces (BCIs). Data and analysis methods used in this project are available in the repository.

## Data Acquisition 
Data was self-curated from Seven test subjects (Participants of ISN). The session was 3-minute long per participant. The following table contains the critical information.

![IMG_2508](https://github.com/user-attachments/assets/37851841-3bf4-492d-84f7-0d08a60a57c8)
(Figure 1: One of the participants recording the task)

![image](https://github.com/user-attachments/assets/b9656f04-bb7b-43fe-8635-258090753172)

(Table 1: The experiment information)


## Preliminary results

![image](https://github.com/user-attachments/assets/6190949f-f964-47ef-8992-9dac9dd597fd)

- Overall Accuracy Across Subjects: 26.79
- Removing Subject 3 woul increase the overall accuracy since his data seems noisy/corrupted.

## Files
- The Dataset: Dataset_7 Subjects_SSVEP_Unicorn.rar 
- The Main Script: cca_ssvep_UnicorN_Team.ipynb 
- Secondery/Imported Functions: utilss.py
- 
## Credits
- Nada O. Salah & Farida Sharaf (Official Team)
- Dr. Nour Eldin Elmadany (Assisted with conceptualization and debugging)
- Kareem Akram Hagi (Curated, and tested the interface codes)
