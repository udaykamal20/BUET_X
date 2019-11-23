# BUET_X

This repository contains the code and submission files of Team BUET_X (FPGA section) for Design Automation Conference System Design Contest  2019. The goal was embedded system implementation of neural network based object detection for drones. We placed 8th position in this competition. 

![alt text](https://github.com/udday2014/BUET_X/blob/master/dac_hdc_ranking.png)

The competition details and the dataset can be found from the following link: http://www.cse.cuhk.edu.hk/~byu/2019-DAC-SDC/

We built a Yolo-like light wieght object detection model, where spatial convolution was replaced by the Zero flop, Zero parameter shift layer. The model was built using pytorch and trained on the provided dataset. Then the model was quantized to 8bit precision, and suitable HLS code was developed to implement the model on the Xilinx Ultra96 FPGA board. 

# Repository Details:

* Deploy Folder: Contains the .ipynb file to be run on the pynq and the BUET_X.bin file, which are to be placed in 
  the 'BUET_X/params' folder.
* HLS Folder: Contains all the required HLS scripts and files to synthesize the core. 
* Overlay : Contains the bitstream, tcl, and block design file of the design. (also a pdf version of the block design)
* Training Folder : Contains all the pytorch scripts to train the model and also the trained weight under 
 'Training Folder/weights' directory.

