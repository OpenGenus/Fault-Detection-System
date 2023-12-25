# Fault Detection System

A repository for CNN based binary classification model for the task of detecting defective solar module cells.

Developed by Cherifi Imane; ([Step by step documentation](https://iq.opengenus.org/fault-detection-system-predict-defective-solar-module-cells/))

## Execution
All the codes have been run on Google Colab.
The codes are coded in the python 3.x version

Additional Libraries are:
    - run "!pip install tf-keras-vis" in a cell to use Score-Cam
## Dataset
The dataset used in this project can be downloaded from [this repo](https://github.com/zae-bayern/elpv-dataset)
It is composed of 2,624 samples of 300x300 pixels 8-bit grayscale images of functional and defective solar cells with varying degree of degradations extracted from 44 different solar modules.

## Code
The notebook "fault_detection.ipynb" builds a binary classification model using Transfer Learning from a pre-trained EfficientNetv2B2 on ImageNet Dataset.
The notebook also shows how to use ScoreCam to explain the predictions of the model.

The folder utils have utility function to load the dataset.

The models folder is where the best trained model will be saved
## Notes
* To reproduce this work:
    - download the dataset using the following command "git clone https://github.com/zae-bayern/elpv-dataset.git"
    - Move the images of the dataset to the images folder of this repository


## References:

[1] Buerhop-Lutz, C.; Deitsch, S.; Maier, A.; Gallwitz, F.; Berger, S.; Doll, B.; Hauch, J.; Camus, C. & Brabec, C. J. A Benchmark for Visual Identification of Defective Solar Cells in Electroluminescence Imagery. European PV Solar Energy Conference and Exhibition (EU PVSEC), 2018. DOI: 10.4229/35thEUPVSEC20182018-5CV.3.15

[2] Deitsch, S., Buerhop-Lutz, C., Sovetkin, E., Steland, A., Maier, A., Gallwitz, F., & Riess, C. (2021). Segmentation of photovoltaic module cells in uncalibrated electroluminescence images. Machine Vision and Applications, 32(4). DOI: 10.1007/s00138-021-01191-9
