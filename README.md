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

    Epoch 1
    Train loss=0.1072 batch_id=468 Accuracy=62.38: 100% 469/469 [00:12<00:00, 38.92it/s]
    Test set: Average loss: 0.1845, Accuracy: 9393/10000 (93.93%)
    
    Epoch 2
    Train loss=0.1953 batch_id=468 Accuracy=96.42: 100% 469/469 [00:12<00:00, 38.63it/s]
    Test set: Average loss: 0.0693, Accuracy: 9774/10000 (97.74%)
    
    Epoch 3
    Train loss=0.0355 batch_id=468 Accuracy=97.80: 100% 469/469 [00:12<00:00, 38.51it/s]
    Test set: Average loss: 0.0575, Accuracy: 9804/10000 (98.04%)
    
    Epoch 4
    Train loss=0.0486 batch_id=468 Accuracy=98.38: 100% 469/469 [00:12<00:00, 38.97it/s]
    Test set: Average loss: 0.0494, Accuracy: 9838/10000 (98.38%)
    
    Epoch 5
    Train loss=0.0223 batch_id=468 Accuracy=98.69: 100% 469/469 [00:11<00:00, 41.22it/s]
    Test set: Average loss: 0.0397, Accuracy: 9871/10000 (98.71%)
    
    Epoch 6
    Train loss=0.0085 batch_id=468 Accuracy=98.92: 100% 469/469 [00:12<00:00, 38.89it/s]
    Test set: Average loss: 0.0430, Accuracy: 9866/10000 (98.66%)
    
    Epoch 7
    Train loss=0.0446 batch_id=468 Accuracy=99.05: 100% 469/469 [00:12<00:00, 38.87it/s]
    Test set: Average loss: 0.0338, Accuracy: 9887/10000 (98.87%)
    
    Epoch 8
    Train loss=0.2567 batch_id=468 Accuracy=99.12: 100% 469/469 [00:12<00:00, 38.64it/s]
    Test set: Average loss: 0.0311, Accuracy: 9906/10000 (99.06%)
    
    Epoch 9
    Train loss=0.0181 batch_id=468 Accuracy=99.28: 100% 469/469 [00:12<00:00, 39.08it/s]
    Test set: Average loss: 0.0388, Accuracy: 9885/10000 (98.85%)
    
    Epoch 10
    Train loss=0.0428 batch_id=468 Accuracy=99.28: 100% 469/469 [00:12<00:00, 38.84it/s]
    Test set: Average loss: 0.0310, Accuracy: 9897/10000 (98.97%)
    
    Epoch 11
    Train loss=0.0022 batch_id=468 Accuracy=99.39: 100% 469/469 [00:12<00:00, 38.89it/s]
    Test set: Average loss: 0.0309, Accuracy: 9912/10000 (99.12%)
    
    Epoch 12
    Train loss=0.0033 batch_id=468 Accuracy=99.44: 100% 469/469 [00:12<00:00, 38.32it/s]
    Test set: Average loss: 0.0324, Accuracy: 9902/10000 (99.02%)
    
    Epoch 13
    Train loss=0.0004 batch_id=468 Accuracy=99.49: 100% 469/469 [00:12<00:00, 38.82it/s]
    Test set: Average loss: 0.0310, Accuracy: 9908/10000 (99.08%)
    
    Epoch 14
    Train loss=0.0033 batch_id=468 Accuracy=99.59: 100% 469/469 [00:11<00:00, 39.25it/s]
    Test set: Average loss: 0.0362, Accuracy: 9904/10000 (99.04%)
    
    Epoch 15
    Train loss=0.0147 batch_id=468 Accuracy=99.58: 100% 469/469 [00:12<00:00, 38.86it/s]
    Test set: Average loss: 0.0380, Accuracy: 9913/10000 (99.13%)

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

    Epoch 1
    Train loss=0.0637 batch_id=468 Accuracy=91.11: 100% 469/469 [00:12<00:00, 37.89it/s]
    Test set: Average loss: 0.0782, Accuracy: 9796/10000 (97.96%)
    
    Epoch 2
    Train loss=0.0363 batch_id=468 Accuracy=98.12: 100% 469/469 [00:12<00:00, 38.95it/s]
    Test set: Average loss: 0.0612, Accuracy: 9843/10000 (98.43%)
    
    Epoch 3
    Train loss=0.0202 batch_id=468 Accuracy=98.53: 100% 469/469 [00:12<00:00, 38.82it/s]
    Test set: Average loss: 0.0428, Accuracy: 9861/10000 (98.61%)
    
    Epoch 4
    Train loss=0.0086 batch_id=468 Accuracy=98.81: 100% 469/469 [00:12<00:00, 38.22it/s]
    Test set: Average loss: 0.0356, Accuracy: 9885/10000 (98.85%)
    
    Epoch 5
    Train loss=0.0729 batch_id=468 Accuracy=98.94: 100% 469/469 [00:12<00:00, 38.02it/s]
    Test set: Average loss: 0.0353, Accuracy: 9885/10000 (98.85%)
    
    Epoch 6
    Train loss=0.0816 batch_id=468 Accuracy=99.08: 100% 469/469 [00:12<00:00, 37.20it/s]
    Test set: Average loss: 0.0423, Accuracy: 9870/10000 (98.70%)
    
    Epoch 7
    Train loss=0.0048 batch_id=468 Accuracy=99.18: 100% 469/469 [00:12<00:00, 37.99it/s]
    Test set: Average loss: 0.0341, Accuracy: 9886/10000 (98.86%)
    
    Epoch 8
    Train loss=0.0203 batch_id=468 Accuracy=99.19: 100% 469/469 [00:12<00:00, 37.81it/s]
    Test set: Average loss: 0.0334, Accuracy: 9895/10000 (98.95%)
    
    Epoch 9
    Train loss=0.0063 batch_id=468 Accuracy=99.31: 100% 469/469 [00:12<00:00, 38.32it/s]
    Test set: Average loss: 0.0290, Accuracy: 9908/10000 (99.08%)
    
    Epoch 10
    Train loss=0.0046 batch_id=468 Accuracy=99.39: 100% 469/469 [00:11<00:00, 39.16it/s]
    Test set: Average loss: 0.0317, Accuracy: 9896/10000 (98.96%)
    
    Epoch 11
    Train loss=0.0148 batch_id=468 Accuracy=99.42: 100% 469/469 [00:12<00:00, 38.17it/s]
    Test set: Average loss: 0.0315, Accuracy: 9894/10000 (98.94%)
    
    Epoch 12
    Train loss=0.0436 batch_id=468 Accuracy=99.48: 100% 469/469 [00:12<00:00, 37.82it/s]
    Test set: Average loss: 0.0294, Accuracy: 9907/10000 (99.07%)
    
    Epoch 13
    Train loss=0.0280 batch_id=468 Accuracy=99.50: 100% 469/469 [00:12<00:00, 37.83it/s]
    Test set: Average loss: 0.0255, Accuracy: 9913/10000 (99.13%)
    
    Epoch 14
    Train loss=0.0367 batch_id=468 Accuracy=99.50: 100% 469/469 [00:12<00:00, 38.01it/s]
    Test set: Average loss: 0.0326, Accuracy: 9898/10000 (98.98%)
    
    Epoch 15
    Train loss=0.0275 batch_id=468 Accuracy=99.57: 100% 469/469 [00:12<00:00, 37.50it/s]
    Test set: Average loss: 0.0298, Accuracy: 9906/10000 (99.06%)


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

    Epoch 1
    Train loss=0.0953 batch_id=468 Accuracy=93.35: 100% 469/469 [00:20<00:00, 23.31it/s]
    Test set: Average loss: 0.0573, Accuracy: 9827/10000 (98.27%)
    
    Epoch 2
    Train loss=0.0164 batch_id=468 Accuracy=97.98: 100% 469/469 [00:21<00:00, 21.79it/s]
    Test set: Average loss: 0.0383, Accuracy: 9875/10000 (98.75%)
    
    Epoch 3
    Train loss=0.0280 batch_id=468 Accuracy=98.38: 100% 469/469 [00:21<00:00, 21.72it/s]
    Test set: Average loss: 0.0275, Accuracy: 9916/10000 (99.16%)
    
    Epoch 4
    Train loss=0.0396 batch_id=468 Accuracy=98.62: 100% 469/469 [00:21<00:00, 22.23it/s]
    Test set: Average loss: 0.0293, Accuracy: 9902/10000 (99.02%)
    
    Epoch 5
    Train loss=0.0102 batch_id=468 Accuracy=98.75: 100% 469/469 [00:19<00:00, 23.73it/s]
    Test set: Average loss: 0.0222, Accuracy: 9919/10000 (99.19%)
    
    Epoch 6
    Train loss=0.0113 batch_id=468 Accuracy=98.85: 100% 469/469 [00:21<00:00, 22.01it/s]
    Test set: Average loss: 0.0256, Accuracy: 9922/10000 (99.22%)
    
    Epoch 7
    Train loss=0.0568 batch_id=468 Accuracy=98.91: 100% 469/469 [00:21<00:00, 22.05it/s]
    Test set: Average loss: 0.0244, Accuracy: 9917/10000 (99.17%)
    
    Epoch 8
    Train loss=0.0426 batch_id=468 Accuracy=99.00: 100% 469/469 [00:21<00:00, 22.12it/s]
    Test set: Average loss: 0.0221, Accuracy: 9927/10000 (99.27%)
    
    Epoch 9
    Train loss=0.0631 batch_id=468 Accuracy=99.12: 100% 469/469 [00:20<00:00, 22.88it/s]
    Test set: Average loss: 0.0195, Accuracy: 9940/10000 (99.40%)
    
    Epoch 10
    Train loss=0.1005 batch_id=468 Accuracy=99.15: 100% 469/469 [00:20<00:00, 23.25it/s]
    Test set: Average loss: 0.0205, Accuracy: 9933/10000 (99.33%)
    
    Epoch 11
    Train loss=0.0061 batch_id=468 Accuracy=99.20: 100% 469/469 [00:21<00:00, 21.62it/s]
    Test set: Average loss: 0.0183, Accuracy: 9941/10000 (99.41%)
    
    Epoch 12
    Train loss=0.0300 batch_id=468 Accuracy=99.28: 100% 469/469 [00:21<00:00, 21.60it/s]
    Test set: Average loss: 0.0181, Accuracy: 9942/10000 (99.42%)
    
    Epoch 13
    Train loss=0.0048 batch_id=468 Accuracy=99.33: 100% 469/469 [00:20<00:00, 22.34it/s]
    Test set: Average loss: 0.0183, Accuracy: 9940/10000 (99.40%)
    
    Epoch 14
    Train loss=0.0475 batch_id=468 Accuracy=99.27: 100% 469/469 [00:20<00:00, 22.97it/s]
    Test set: Average loss: 0.0178, Accuracy: 9940/10000 (99.40%)
    
    Epoch 15
    Train loss=0.0140 batch_id=468 Accuracy=99.36: 100% 469/469 [00:21<00:00, 22.21it/s]
    Test set: Average loss: 0.0180, Accuracy: 9941/10000 (99.41%)

### Plots
<img width="1249" height="836" alt="image" src="https://github.com/user-attachments/assets/a882effe-9549-4cd8-b280-3e2eb2c1e8ec" />
