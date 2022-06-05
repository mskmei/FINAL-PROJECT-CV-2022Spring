class CONFIG:
    log_file  = './log'
    
    # # example for paddle 
    paddle_data_file  = None
    paddle_save_path  = './models/model.pdparams'
    paddle_device     = 'gpu:0'

    # # example for torch
    torch_data_file   = None
    torch_save_path   = './models/model.pth'
    torch_device      = 'cuda:0'


    # data augmentation
    resize        = 32            # resize the size of the image, 32 for default
    augmentation  = True
    cut           = 4             # size of Cutout, set to zero to disable
    mix           = True          # enable (True) / disable (False) Mixup
                                  # When cut > 0 and mix == True, then it activates CutMix

    jitter        = (.5,.5,.5,.5) # image jitter, (brightness, contrast, saturation, hue)
    
    # transformer architecture
    channels          = 192       # = embed_dim
    hidden_size       = 768       # hidden size in the MLP layer
    patch_size        = 4         # size of each patch
    num_heads         = 3         # number of heads in multihead attention
    encode_layers     = 12        # number of encoders
    pos_encoding      = True      # whether or not use the positional encoding
    res_head          = None      # list, the size of resnet head in the front
    linear_dropout    = 0.        # dropout rate in MLP
    attention_dropout = 0.        # dropout rate in multihead attention
    encoding_dropout  = 0.        # dropout rate after positional encoding

    # training
    train_size    = 256           # training batch size
    test_size     = 100           # testing batch size
    epochs        = 90
    learning_rate = 3e-4          # learning rate for optimizer
    warmup_iters  = 780           # warmup iterations


