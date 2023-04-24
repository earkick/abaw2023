# abaw2023

The code in this repository produces results described in the paper:
*Analysis of Emotion Annotation Strength Improves Generalization of Speech Emotion Recognition Models*
accepted as a workshop paper at the ABAW workshop at CVPR 2023.
---
## Annotations
We re-annotated 1% of the audio emotion data, namely, 317 files from a combined dataset contained CREMA-D, RAVDESS, CMU MOSEI and AFFWILD2.

The annotations are included in assessments.csv. The relevant columns are: id, happy, sad, fear, disgust, anger, surprise.
Each emotion has four levels in increasing emotion strength: None (0), A little (1), Moderate (2), Extremely (3). These levels are on a Likert scale on [0, 3] as done in the CMU MOSEI dataset. The audio files under "audio" column can be mapped to the file names in the original dataset. Original labels are not included.