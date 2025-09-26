# MnistLite-8k
MNist Neural Network
The goal is to achieve 99.4% validation/test accuracy consistently, with less than 15 epochs and 8k parameters

## Model1
We start with first iteration

### Target:
- Get the set-up right
- Set Transforms
- Set Data Loader
- Set Basic Working Code
- Set Basic Training & Test Loop
- Get the basic skeleton right
### Results:
- Parameters: 26.5k
- Best Training Accuracy: 99.64%
- Best Test Accuracy: 99.00%
### Analysis
- In the initial epochs, the model quickly jumps from ~69% test accuracy (Epoch 1) to ~96% (Epoch 2) and ~98%+ by Epoch 3–4. This shows the network is learning the MNIST features very efficiently even with a small parameter count.
- Training accuracy steadily rises and exceeds 99% around Epoch 10, while test accuracy stabilizes between 98.8–99.0%.
- There is no major overfitting observed — the gap between training and test accuracy remains within ~0.5–0.6%, which is acceptable.
- The model is relatively lightweight (26k params) compared to typical CNNs on MNIST, yet it achieves strong performance close to larger networks.
- Slight oscillations in test accuracy after Epoch 10 (98.89% → 99.00% → 98.95%) are normal variance, not a sign of severe overfitting.
- Early training shows very fast convergence. No significant overfitting is visible, though performance seems to have plateaued around 99%, suggesting that improvements will likely come from architectural tweaks rather than longer training.

### Model Architecture

    Net(
      (conv1): Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv2): Conv2d(8, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (conv3): Conv2d(10, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv4): Conv2d(16, 28, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (conv5): Conv2d(28, 32, kernel_size=(3, 3), stride=(1, 1))
      (conv6): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1))
      (conv7): Conv2d(32, 10, kernel_size=(3, 3), stride=(1, 1))
    )

### Model Parameters

    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1            [-1, 8, 28, 28]              80
                Conv2d-2           [-1, 10, 28, 28]             730
             MaxPool2d-3           [-1, 10, 14, 14]               0
                Conv2d-4           [-1, 16, 14, 14]           1,456
                Conv2d-5           [-1, 28, 14, 14]           4,060
             MaxPool2d-6             [-1, 28, 7, 7]               0
                Conv2d-7             [-1, 32, 5, 5]           8,096
                Conv2d-8             [-1, 32, 3, 3]           9,248
                Conv2d-9             [-1, 10, 1, 1]           2,890
    ================================================================
    Total params: 26,560
    Trainable params: 26,560
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.00
    Forward/backward pass size (MB): 0.21
    Params size (MB): 0.10
    Estimated Total Size (MB): 0.31
    ----------------------------------------------------------------


### Training Logs


## Model2

### Target:
- Make the model lighter (reduce parameter count, stay < 8k)
- Use Global Average Pooling (GAP) → reduces overfitting, removes need for large dense layer
- Add BatchNorm after convs → stabilize training, faster convergence, better generalization
### Results:
- Parameters: 5.6k
- Best Training Accuracy: 99.48%
- Best Test Accuracy: By Epoch 13–15: ~99.0–99.18%.
### Analysis
- lighter model + GAP + BN gave better stability, fewer params, and accuracy that scales above 99% by 13–15 epochs.
- still a bit short of the 99.4% milestone (peaked at ~99.18%).
- Notice how training loss is stable and doesn’t collapse/oscillate.
- This iteration is clearly more efficient and better aligned with our goal compared to the first model.

### Model Architecture

    Net(
      (conv1): Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1))
      (bn1): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(8, 10, kernel_size=(3, 3), stride=(1, 1))
      (bn2): BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (conv3): Conv2d(10, 8, kernel_size=(1, 1), stride=(1, 1))
      (bn3): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv4): Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1))
      (bn4): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv5): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1))
      (bn5): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (conv6): Conv2d(16, 10, kernel_size=(1, 1), stride=(1, 1))
      (bn6): BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv7): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1))
      (gap): AdaptiveAvgPool2d(output_size=(1, 1))
    )

### Model Parameters

    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1            [-1, 8, 26, 26]              80
           BatchNorm2d-2            [-1, 8, 26, 26]              16
                Conv2d-3           [-1, 10, 24, 24]             730
           BatchNorm2d-4           [-1, 10, 24, 24]              20
             MaxPool2d-5           [-1, 10, 12, 12]               0
                Conv2d-6            [-1, 8, 12, 12]              88
           BatchNorm2d-7            [-1, 8, 12, 12]              16
                Conv2d-8           [-1, 16, 10, 10]           1,168
           BatchNorm2d-9           [-1, 16, 10, 10]              32
               Conv2d-10             [-1, 16, 8, 8]           2,320
          BatchNorm2d-11             [-1, 16, 8, 8]              32
            MaxPool2d-12             [-1, 16, 4, 4]               0
               Conv2d-13             [-1, 10, 4, 4]             170
          BatchNorm2d-14             [-1, 10, 4, 4]              20
               Conv2d-15             [-1, 10, 2, 2]             910
    AdaptiveAvgPool2d-16             [-1, 10, 1, 1]               0
    ================================================================
    Total params: 5,602
    Trainable params: 5,602
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.00
    Forward/backward pass size (MB): 0.24
    Params size (MB): 0.02
    Estimated Total Size (MB): 0.27
    ----------------------------------------------------------------


### Training Logs


## Model3

### Target:
- Make the model lighter (reduce parameter count, stay < 8k)
- Use Global Average Pooling (GAP) → reduces overfitting, removes need for large dense layer
- Add BatchNorm after convs → stabilize training, faster convergence, better generalization
### Results:
- Parameters: 5.6k
- Best Training Accuracy: 99.48%
- Best Test Accuracy: By Epoch 13–15: ~99.0–99.18%.
### Analysis
- lighter model + GAP + BN gave better stability, fewer params, and accuracy that scales above 99% by 13–15 epochs.
- still a bit short of the 99.4% milestone (peaked at ~99.18%).
- Notice how training loss is stable and doesn’t collapse/oscillate.
- This iteration is clearly more efficient and better aligned with our goal compared to the first model.

### Model Architecture

    Net(
      (dropout): Dropout(p=0.01, inplace=False)
      (conv1): Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1))
      (bn1): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(8, 10, kernel_size=(3, 3), stride=(1, 1))
      (bn2): BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (conv3): Conv2d(10, 8, kernel_size=(1, 1), stride=(1, 1))
      (bn3): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv4): Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1))
      (bn4): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv5): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1))
      (bn5): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (conv6): Conv2d(32, 8, kernel_size=(1, 1), stride=(1, 1))
      (bn6): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv7): Conv2d(8, 10, kernel_size=(3, 3), stride=(1, 1))
      (gap): AdaptiveAvgPool2d(output_size=(1, 1))
    )

### Model Parameters

    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1            [-1, 8, 26, 26]              80
           BatchNorm2d-2            [-1, 8, 26, 26]              16
               Dropout-3            [-1, 8, 26, 26]               0
                Conv2d-4           [-1, 10, 24, 24]             730
           BatchNorm2d-5           [-1, 10, 24, 24]              20
               Dropout-6           [-1, 10, 24, 24]               0
             MaxPool2d-7           [-1, 10, 12, 12]               0
                Conv2d-8            [-1, 8, 12, 12]              88
           BatchNorm2d-9            [-1, 8, 12, 12]              16
              Dropout-10            [-1, 8, 12, 12]               0
               Conv2d-11           [-1, 16, 10, 10]           1,168
          BatchNorm2d-12           [-1, 16, 10, 10]              32
              Dropout-13           [-1, 16, 10, 10]               0
               Conv2d-14             [-1, 32, 8, 8]           4,640
          BatchNorm2d-15             [-1, 32, 8, 8]              64
              Dropout-16             [-1, 32, 8, 8]               0
            MaxPool2d-17             [-1, 32, 4, 4]               0
               Conv2d-18              [-1, 8, 4, 4]             264
          BatchNorm2d-19              [-1, 8, 4, 4]              16
              Dropout-20              [-1, 8, 4, 4]               0
               Conv2d-21             [-1, 10, 2, 2]             730
    AdaptiveAvgPool2d-22             [-1, 10, 1, 1]               0
    ================================================================
    Total params: 7,864
    Trainable params: 7,864
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.00
    Forward/backward pass size (MB): 0.38
    Params size (MB): 0.03
    Estimated Total Size (MB): 0.42
    ----------------------------------------------------------------


### Training Logs

