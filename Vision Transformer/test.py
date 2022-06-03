from configs import CONFIG

def CountLayers(model, x):
    # count how many layers x are there in model
    last_index = -1
    for layer_name in list(model.keys())[::-1]:
        if layer_name.startswith(x + '.'):
            last_index = int( layer_name[len(x) + 1:].split('.')[0] )
            break 
    return last_index + 1

def AskConfigs(model, backend = 'paddle'):
    '''
    Because there are different transformer architectures and inputs, 
    some configuration hyperparams are needed.

    ONLY SUPPORT MODEL THAT HAS POSITIONAL ENCODING
    '''

    c = {}

    # read whether there is ResNet blocks
    c['res_head'] = CountLayers(model, 'res_head.reslayers')
    if c['res_head'] == 0:
        c['res_head'] = None
    else:
        res_head_num = []
        for i in range(c['res_head']):
            res_head_num.append(
                CountLayers(model, 'res_head.reslayers.%d.sequence'%i)
                                )
        c['res_head'] = res_head_num

    # read the input size, defaults to 32 when no ResNet blocks but 224 otherwise
    input_size_default = 32 if c['res_head'] is None else 224
    input_size = input('Input the input size of your image, '
                        'defaults to %d\n >> '%input_size_default)
    try:    input_size = int(input_size)
    except: input_size = input_size_default
    if input_size <= 0: input_size = input_size_default
    c['input_size'] = input_size

    # real input size after the ResNet downscaling
    if c['res_head'] is not None:
        input_size = input_size // (2 ** (len(c['res_head']) + 1))


    c['output_size'] = 100

    c['channels'] = model['cls_token'].shape[2]

    c['patch_size'] = input_size // round((model['pos_encoding'].shape[1] - 1)**0.5) 

    # the hidden weight in paddle and torch are transposed
    if backend == 'paddle':
        c['hidden_size'] = model['encoders.0.mlp.dense1.weight'].shape[1]
    elif backend == 'torch':
        c['hidden_size'] = model['encoders.0.mlp.dense1.weight'].shape[0] 
    
    c['encode_layers'] = CountLayers(model, 'encoders')

    c['pos_encoding'] = 'pos_encoding' in model.keys()


    num_heads_default = 3 if c['channels'] <= 200 else 6
    num_heads = input('Input the number of heads, defaults to %d\n >> '%num_heads_default)
    try:    num_heads = int(num_heads)
    except: num_heads = num_heads_default
    c['num_heads'] = num_heads

    print('Network architecture = \n', c, '\n')
    return c




if __name__ == '__main__':

    path = input('Input the path to your vision transformer here\n >> ')
    device = 0
    path = path.strip().strip("'").strip('"')
    if path.endswith('.pdparams'):
        import paddle
        if paddle.fluid.is_compiled_with_cuda():
            device = input('Input the index of GPU you want to test on:').strip()
            if len(device) == 0:
                device = '0'
            if not device.startswith('gpu:'):
                device = 'gpu:' + device 
        else:
            device = 'cpu'

        paddle.set_device(device)

        model = paddle.load(path)
        configs = AskConfigs(model, 'paddle')

        from transformer_paddle import VisionTransformer
        net = VisionTransformer(**configs)
        net.set_state_dict(model)

        
        from dataloader_paddle import preprocessor, test
        data_file = CONFIG.paddle_data_file

    elif path.endswith('.pth'):
        import torch 
        if torch.cuda.is_available():
            device = input('Input the index of CUDA you want to test on:').strip()
            if len(device) == 0:
                device = '0'
            if not device.startswith('cuda:'):
                device = 'cuda:' + device 
        else:
            device = 'cpu'

        model = torch.load(path, map_location = device)
        configs = AskConfigs(model, 'torch')

        from transformer_torch import VisionTransformer
        net = VisionTransformer(**configs)
        net.to(device)
        net.load_state_dict(model)
        

        from dataloader_torch import preprocessor, test
        data_file = CONFIG.torch_data_file
    else:
        raise ValueError('Model format not supported or model directory not found.')




    print('Loading Dataset......')
    (train_x, train_y) , (test_x, test_y) = preprocessor(data_file, 
                            resize = (configs['input_size'],configs['input_size']),
                            only_test = True)
    print('Loading Finished\nStarting Testing......')

    acc = test(net, test_x, test_y, CONFIG.test_size, verbose = True, cuda = device)

    print('Top 1 Acc = %.2f%%\nTop 5 Acc = %.2f%%'%(acc[0] * 100, acc[1] * 100))
