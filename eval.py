import os

import numpy as np
import torch
from PIL.Image import Image

from model import utils
import logging
import arguments


from model import evaluation

def main():
    # Hyper Parameters    
    parser = arguments.get_argument_parser()
    opt = parser.parse_args()
    
    # the path of saving model ckpts and train logs
    if utils.is_main_process():
        opt.model_name = opt.logger_name
    # Set GPU
    if opt.multi_gpu:
        utils.init_distributed_mode(opt)
    else:
        torch.cuda.set_device(opt.gpu_id)

    utils.set_seed(opt.seed)


    if utils.is_main_process() and (not os.path.exists(opt.model_name)):
        os.makedirs(opt.model_name)

    # start eval
    if utils.is_main_process() and opt.eval:
        print('Evaluate the model now.')

        base = opt.logger_name

        model_path = os.path.join(base, 'model_best.pth')

        # Save the final results for computing ensemble results
        save_path = os.path.join(base, 'results_{}.npy'.format(opt.dataset))

        if opt.dataset == 'coco':
            # Evaluate COCO 5-fold 1K
            evaluation.evalrank(model_path, model=None, split='testall', fold5=True)

            # Evaluate COCO 5K
            evaluation.evalrank(model_path, model=None, split='testall', fold5=False, save_path=save_path)

            if opt.evaluate_cxc:
                # Evaluate COCO-trained models on CxC
                evaluation.evalrank(model_path, model=None, split='testall', fold5=True, cxc=True)

        else:
            # Evaluate Flickr30K
            evaluation.evalrank(model_path, model=None, split='test', fold5=False, save_path=save_path)

        print('Evaluation finish!')


if __name__ == '__main__':
    
    main()
