# U-Net
![u net structure](https://u9qcsa.am.files.1drv.com/y4mfknEQpOtArPojQ2nvN1cOwci_zxLM5FoFynKOV2v9CSRrz21vlZeR7zo1cmn4Tm7fMBo9u_FpMPn-ZRyS2Oxdf800tE0K2BGzb5Dp7CYjYbTzCjSrtd2JyWEqKa7kJW4IX9tO4TIJDdRdPLa2_k7GyBkLyJBfX7Y2ws5bbHUm8qb4fb8j1T_6w8YOHkRy6bxbXlITze31atOQYP_f1v17g?width=823&height=513&cropmode=none)

U-net consists of a contracting
path (left side) and an expansive path (right side).<br>

### Contracting Phase
The contracting path follows
the typical architecture of a convolutional network. It consists of the repeated
application of two 3x3 convolutions (unpadded convolutions), each followed by
a rectied linear unit (ReLU) and a 2x2 max pooling operation with stride 2
for downsampling.<br>
At each downsampling step we double the number of feature
channels.<br><br>

### Expansive Phase
Every step in the expansive path consists of an upsampling of the
feature map followed by a 2x2 convolution ("up-convolution") that halves the number of feature channels, a concatenation with the correspondingly cropped
feature map from the contracting path, and two 3x3 convolutions, each followed by a ReLU.<br>
The cropping is necessary due to the loss of border pixels in
every convolution. At the final layer a 1x1 convolution is used to map each 64-component feature vector to the desired number of classes. In total the network has 23 convolutional layers.<br><br>

To allow a seamless tiling of the output segmentation map (see Figure 2), it is important to select the input tile size such that all 2x2 max-pooling operations are applied to a layer with an even x- and y-size.<br>

## Network Result Example
![example](https://utqcsa.am.files.1drv.com/y4mtuNLTy3W1hyGp1Je9JtrVzQOiqShDAlCcHHkTvfXsH8Au_aBIWnvq65T4zLqrs95xV-mGu3l_rqRLkcOMh1UWRMjWGaVr1-3_BmU52Kfb49PPXSzm4YBgjdCssRA8sHWUY6ctyTyMgMdF-TKdGbFrDkT-QG96OHzIzMX7mD97XYcDJLAjfhqcaG1ooKQZ33NqPWURGMVQ4ABbtKpbWRjvA?width=868&height=399&cropmode=none)


## Implementation
### Environment
In this repo, I implemented U-net using Pytorch and Python 3.6.7.

### Diff with Paper
Except `depth`, all parameters are same as paper. Default value of `depth` in paper is `5` (look at the shape at top of page). So you can have deeper or shallower network by changing `deep` variable.<br>

### Features
- Run on CPU
- Controllable `Depth` of network

## Upcoming Features
- Ability to change runtime device
- Parallelism (Run on multiple GPU)

## Reference
Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al., 2015)
