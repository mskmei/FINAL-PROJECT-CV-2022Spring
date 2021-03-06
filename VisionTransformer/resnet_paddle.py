# Resnet convolutional backbone implementation in paddle.
# Author    : https://github.com/ForeverHaibara 

import paddle

class ConvBNLayer(paddle.nn.Layer):
    """Convolution layer with batch normalization and relu"""
    def __init__(self, in_channels, out_channels, filter_size, stride = 1):
        super().__init__()

        self.conv = paddle.nn.Conv2D(in_channels, out_channels, filter_size, 
                                        stride=stride, padding=(filter_size-1)//2, bias_attr=False)
        self.bn = paddle.nn.BatchNorm2D(out_channels)
        self.act = paddle.nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x



class ResBlock(paddle.nn.Layer):
    """Residual layer in ResNet, also called bottleneck"""
    def __init__(self, in_channels, out_channels, stride = 1):
        super().__init__()

        self.conv1 = ConvBNLayer(in_channels, out_channels // 4, 1, 1)
        self.conv2 = ConvBNLayer(out_channels // 4, out_channels // 4, 3, stride)
        self.conv3 = ConvBNLayer(out_channels // 4, out_channels, 1, 1)

        if in_channels != out_channels * 4:
            # a conv layer that match the input to the filters
            self.short = ConvBNLayer(in_channels, out_channels, 1, stride)
        else:
            self.short = None

        self.act = paddle.nn.LeakyReLU()
            
    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        if self.short is not None:
            # a conv layer that match the input to the filters
            x = self.short(x)
        y += x
        y = self.act(y)
        return y 



class ResLayer(paddle.nn.Layer):
    # A series of residual blocks make up a ResLayer
    def __init__(self, num, out_channels, in_channels = 0, stride = 2):
        super().__init__()
        if in_channels == 0:
            # default: the channel doubles after each ResLayer
            in_channels = out_channels // 2

        self.sequence = paddle.nn.LayerList()

        # in most cases there is a downsampling by stride = 2
        self.sequence.append(
            ResBlock(in_channels, out_channels, stride = stride)
        )

        for i in range(num - 1):
            self.sequence.append(
                ResBlock(out_channels, out_channels, stride = 1)
            )
        
    def forward(self, x):
        for i in range(len(self.sequence)):
            x = self.sequence[i](x)
        return x


    
class ResHead(paddle.nn.Layer):
    """Head of a Resnet"""
    def __init__(self, nums, channels = None):
        super().__init__()
        self.conv1 = ConvBNLayer(3, 64, 7, 2)
        self.pool1 = paddle.nn.MaxPool2D(kernel_size = 3, stride = 2, padding = 1)
        
        if channels is None:
            # default: the channel doubles after each ResLayer
            channels = [64 * 2**i for i in range(len(nums))]

        self.reslayers = paddle.nn.LayerList()
        self.reslayers.append(
            ResLayer(nums[0], channels[0], in_channels = 64, stride = 1)
        )

        for i, c in zip(nums[1:], channels[1:]):
            self.reslayers.append(
                ResLayer(i, c)
            )

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        for i in range(len(self.reslayers)):
            x = self.reslayers[i](x)
        return x