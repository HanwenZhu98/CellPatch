import torch


def load_basic_ckpt(args, model,encoder_only = True):
    """
    加载模型checkpoint
    """
    
    #尝试加载args.ckpt_path
    assert(args.ckpt_path is not None)
    print(f'load ckpt from {args.ckpt_path}')
    ckpt = torch.load(args.ckpt_path,map_location=args.device)

    new_state_dict = {}
    for k, v in ckpt['model_state_dict'].items():
        if k.startswith('module.'):
            k = k[7:]
        if 'PathwayDecoder' not in k:
            new_state_dict[k] = v


    #only load encoder
    # model.load_state_dict(new_state_dict)
    load_info = model.load_state_dict(new_state_dict,strict = False)
    print(load_info)
    return model
