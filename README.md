# CycleGAN with Self-Attention Layer
<img align="center" alt="architecture" src="./images/Generator.png" /><br>
In this repository, I have developed a CycleGAN architecture with embedded Self-Attention Layers, that could solve three different complex tasks. Here the same principle Neural Network architecture has been used to solve the three different task. Although truth be told, my model has not exceeded any state of the art performances for the given task, but the architecture was powerful enough to understand the task that has been given to solve and produce considerably good results.

# About the architecture
The concept of CycleGAN used in this project is the same as the original. The novel approach that I have added is adding the self-attention layers to the U-net generator and discriminator. The concept of self attention is inspired from the research paper <a href="https://arxiv.org/pdf/1805.08318v2.pdf">Self-Attention Generative Adversarial Networks</a>. I have modified the self-attention layer discussed in the research paper for better results. In my case, the base formula for attention is shown below.
<img align="center" alt="attention" src="./images/att.png" />
The base code for the self-attention layer is built around this formula. The self-attention layers added at the bottleneck of the u-net and right before the output Convolution Layer. In the case of the discriminator, the self-attention layers are added right before the zero-padding layer and right before the output layer.

## Technologies used:
1. The entire architecture is built with tensorflow. 
2. Matplotlib has been used for visualization. 
3. Numpy has been used for mathematical operations. 
4. OpenCV have used for the processing of images.

# Tasks Solved by the Architecture
I have trained and validated the model with an image size of 256 and trained over 800 epochs. The default parameters mentioned in the config.py file are the baseline parameters used for training over three different tasks.

## Colorize Sketch
The given task is to colorize a input facial sketch image.<br>
Over Training examples<br>
<img align="center" alt="sketch1" src="./images/Sketch/sketch1.png" />
<img align="center" alt="sketch2" src="./images/Sketch/sketch2.png" />
<img align="center" alt="sketch3" src="./images/Sketch/sketch3.png" />
<img align="center" alt="sketch4" src="./images/Sketch/sketch4.png" />
<img align="center" alt="sketch5" src="./images/Sketch/sketch5.png" />
<img align="center" alt="sketch6" src="./images/Sketch/sketch6.png" />
<img align="center" alt="sketch7" src="./images/Sketch/sketch7.png" />
<img align="center" alt="sketch8" src="./images/Sketch/sketch8.png" />
<img align="center" alt="sketch9" src="./images/Sketch/sketch9.png" />
<img align="center" alt="sketch10" src="./images/Sketch/sketch10.png" />
<br>Over Validation examples<br>
<img align="center" alt="sketchx" src="./images/Sketch/sketchx.jpg" />
<img align="center" alt="sketchy" src="./images/Sketch/sketchy.jpg" />
<img align="center" alt="sketch11" src="./images/Sketch/sketch11.png" />
<img align="center" alt="sketch12" src="./images/Sketch/sketch12.png" />
<img align="center" alt="sketch13" src="./images/Sketch/sketch13.png" />
<img align="center" alt="sketch14" src="./images/Sketch/sketch14.png" />
<img align="center" alt="sketch15" src="./images/Sketch/sketch15.png" />
<img align="center" alt="sketch16" src="./images/Sketch/sketch16.png" />
<img align="center" alt="sketch17" src="./images/Sketch/sketch17.png" />
<img align="center" alt="sketch18" src="./images/Sketch/sketch18.png" />
<img align="center" alt="sketch19" src="./images/Sketch/sketch19.png" />
<img align="center" alt="sketch20" src="./images/Sketch/sketch20.png" />
<img align="center" alt="sketch21" src="./images/Sketch/sketch21.png" />
<img align="center" alt="sketch22" src="./images/Sketch/sketch22.png" />
<img align="center" alt="sketch23" src="./images/Sketch/sketch23.png" />
<img align="center" alt="sketch24" src="./images/Sketch/sketch24.png" />

## Gender Bender
The given task is to Transform a Male face into a female face.<br>
Over Training examples<br>
<img align="center" alt="gender1" src="./images/Gender/gender1.png" />
<img align="center" alt="gender2" src="./images/Gender/gender2.png" />
<img align="center" alt="gender3" src="./images/Gender/gender3.png" />
<img align="center" alt="gender4" src="./images/Gender/gender4.png" />
<img align="center" alt="gender5" src="./images/Gender/gender5.png" />
<img align="center" alt="gender6" src="./images/Gender/gender6.png" />
<img align="center" alt="gender7" src="./images/Gender/gender7.png" />
<img align="center" alt="gender8" src="./images/Gender/gender8.png" />
<img align="center" alt="gender9" src="./images/Gender/gender9.png" />
<img align="center" alt="gender10" src="./images/Gender/gender10.png" />
<br>Over Validation examples<br>
<img align="center" alt="gender11" src="./images/Gender/gender11.png" />
<img align="center" alt="gender12" src="./images/Gender/gender12.png" />
<img align="center" alt="gender13" src="./images/Gender/gender13.png" />
<img align="center" alt="gender14" src="./images/Gender/gender14.png" />
<img align="center" alt="gender15" src="./images/Gender/gender15.png" />
<img align="center" alt="gender16" src="./images/Gender/gender16.png" />
<img align="center" alt="gender17" src="./images/Gender/gender17.png" />
<img align="center" alt="gender18" src="./images/Gender/gender18.png" />
<img align="center" alt="gender19" src="./images/Gender/gender19.png" />
<img align="center" alt="gender20" src="./images/Gender/gender20.png" />
<img align="center" alt="gender21" src="./images/Gender/gender21.png" />
<img align="center" alt="gender22" src="./images/Gender/gender22.png" />
<img align="center" alt="gender23" src="./images/Gender/gender23.png" />
<img align="center" alt="gender24" src="./images/Gender/gender24.png" />
<img align="center" alt="gender25" src="./images/Gender/gender25.png" />
<img align="center" alt="gender26" src="./images/Gender/gender26.png" />
<img align="center" alt="gender27" src="./images/Gender/gender27.png" />
<img align="center" alt="gender28" src="./images/Gender/gender28.png" />

## Shades and Glass Remover
The given task is to remove glass and sun-glass from an input facial image. While training the model to solve this task the alpha parameter of LeakyReLU was set to 0.4 instead of the default 0.1 for the above two tasks.<br>
Over Training examples<br>
<img align="center" alt="glass1" src="./images/Glass/glass1.png" />
<img align="center" alt="glass2" src="./images/Glass/glass2.png" />
<img align="center" alt="glass3" src="./images/Glass/glass3.png" />
<img align="center" alt="glass4" src="./images/Glass/glass4.png" />
<img align="center" alt="glass5" src="./images/Glass/glass5.png" />
<img align="center" alt="glass6" src="./images/Glass/glass6.png" />
<img align="center" alt="glass7" src="./images/Glass/glass7.png" />
<img align="center" alt="glass8" src="./images/Glass/glass8.png" />
<img align="center" alt="glass9" src="./images/Glass/glass9.png" />
<img align="center" alt="glass10" src="./images/Glass/glass10.png" />
<br>Over Validation examples<br>
<img align="center" alt="glass11" src="./images/Glass/glass11.png" />
<img align="center" alt="glass12" src="./images/Glass/glass12.png" />
<img align="center" alt="glass13" src="./images/Glass/glass13.png" />
<img align="center" alt="glass14" src="./images/Glass/glass14.png" />
<img align="center" alt="glass15" src="./images/Glass/glass15.png" />
<img align="center" alt="glass16" src="./images/Glass/glass16.png" />
<img align="center" alt="glass17" src="./images/Glass/glass17.png" />
<img align="center" alt="glass18" src="./images/Glass/glass18.png" />
<img align="center" alt="glass19" src="./images/Glass/glass19.png" />
<img align="center" alt="glass20" src="./images/Glass/glass20.png" />
<img align="center" alt="glass21" src="./images/Glass/glass21.png" />
<img align="center" alt="glass22" src="./images/Glass/glass22.png" />
<br><br><br>

# Implimentation
## Training
```
python main.py --height 256 --width 256 --epoch 300 --dataset "./dataset/" --subject 1
```
## Validation
```
python main.py --train False --dataset "./validate/" --validate "face-1001.png" --subject 1
```
# Special cases
As I have mentioned above the a principle architecture thave solved all three tasks, but I have also found out that modifying the self-attention layer architecture by 
```python
   def hw_flatten(self, x) :  
        return layers.Reshape(( -1, x.shape[-2], x.shape[-1]))(x)
```
instead of
```python
   def hw_flatten(self, x) :  
        return layers.Reshape(( -1, x.shape[-2]* x.shape[-1]))(x)
```
have improved the outcomes of the model for solving a particular individual case. Also removing the LayerNormalization and Dropout from the self-attention layer have improved the performance for individual cases.

# CycleGAN with attention Architecture
The Self-Attention layer has been used in both generator and discriminator network.
## Generator
<img align="center" alt="gen" src="./images/Glass/gen.png" />

## Discriminator
<img align="center" alt="dis" src="./images/Glass/dis.png" />

# Future Scopes

The model could be further improved with the further turning of Convolution layers or other layers. Creating a deeper u-net architecture could also have helped in improving the performance of the model.

