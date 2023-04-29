This repo is an adaptation of https://github.com/wangyirui/AD_Prediction

To run a model run the following on the terminal 

``python main_alexnet.py --optimizer Adam --learning_rate 4e-5 --save AlexNet2D --batch_size 16 --epochs 100 --gpuid 0``

For plotting the losses figure, set the corresponding boolean in plot_data.py to true. 

For plotting the results of your model, run plot_data.py and be sure to set the according target filepath for saving in the code.

A seed value is initialized in the main_alexnet.py file.
For the figures constructed, a seed of 11 was used. For the five subsequent runs, a seed of 1-5 was used.

For the figures constructed, an early stoppage of 1e-2 was used. For the five subsequent runs, an early stoppage of 5e-2 was used.

Dependencies
- torch 1.13.0
- numpy 1.21.5
- skimage 0.19.2
- pillow 9.2.0
- torchvision 0.14.0