import os
import time
import copy
import numpy as np
#设置环境变量开启expandable_segments，让 PyTorch 更高效利用预留的未分配显存
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
from model import image_caption, utils
from transformers import BertTokenizer
import logging
import arguments
import swanlab

from model import evaluation
from model.model_factory import build_model, build_optimizer

from model.evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data, shard_attn_scores
from torch.nn.utils import clip_grad_norm_


def load_shape_compatible_weights(model, checkpoint_path):
    """Warm-start a new architecture without loading incompatible heads."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Initialization checkpoint does not exist: {checkpoint_path}"
        )
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    source = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    current = model.state_dict()
    compatible = {}
    skipped = []
    for name, value in source.items():
        clean_name = name[7:] if name.startswith("module.") else name
        if clean_name in current and current[clean_name].shape == value.shape:
            compatible[clean_name] = value
        else:
            skipped.append(clean_name)
    if not compatible:
        raise RuntimeError(
            f"No shape-compatible parameters found in {checkpoint_path}"
        )
    model.load_state_dict(compatible, strict=False)
    logging.info(
        "Warm-started %d tensors from %s; skipped %d incompatible tensors",
        len(compatible),
        checkpoint_path,
        len(skipped),
    )


def main():

    # Hyper Parameters    
    parser = arguments.get_argument_parser()
    opt = parser.parse_args()
    # Initialize the process group before checking the rank. Before DDP is
    # initialized, every torchrun worker appears to be rank 0 and would create
    # its own SwanLab run.
    if opt.multi_gpu:
        utils.init_distributed_mode(opt)
        # set seed
        # seed = opt.seed + utils.get_rank()
        # utils.set_seed(seed)
    else:
        torch.cuda.set_device(opt.gpu_id) 

    # Every rank needs the checkpoint/log path, while only the real global
    # rank 0 is allowed to create and write to the SwanLab run.
    opt.model_name = opt.logger_name
    if utils.is_main_process():
        swanlab.init(project="AVSE", name=opt.model_name)

    utils.set_seed(opt.seed)


    if utils.is_main_process() and (not os.path.exists(opt.model_name)):
        os.makedirs(opt.model_name)
    
    if utils.is_main_process():
        # logging.basicConfig(filename=os.path.join(opt.logger_name, 'train.log'), 
                            # filemode='w', format='%(asctime)s %(message)s', level=logging.INFO)
        logging.basicConfig(
            level=logging.INFO,  # 设置日志级别
            format='%(asctime)s - %(levelname)s - %(message)s',  # 设置日志格式
            datefmt='%Y-%m-%d %H:%M:%S',  # 设置时间格式
            handlers=[  # 设置日志处理器列表
                logging.FileHandler(os.path.join(opt.logger_name, 'train.log')),  # 文件日志处理器
                logging.StreamHandler()  # 控制台日志处理器
            ]
        )
    
        logger = logging.getLogger(__name__)

    if utils.is_main_process():
        logger.info(opt)  
        arguments.save_parameters(opt, opt.logger_name)


    # CLIP is supported as a visual backbone only.  All branches keep the same
    # BERT tokenizer/text encoder so backbone comparisons change one modality.
    tokenizer = BertTokenizer.from_pretrained(opt.bert_path)
    # opt.vocab_size = len(tokenizer.vocab)
    # vocab_size of BERT model: 30522
    # print('vocab_size of BERT model:', opt.vocab_size)

    # V5 keeps the empirically successful V2-GL and V3 batch constructions,
    # but joins them inside one uninterrupted run. The first loader contains
    # ordinary image-caption pairs; the second contains two descriptions for
    # each image identity.
    if opt.base_epochs <= 0 or opt.base_epochs > opt.num_epochs:
        raise ValueError(
            "--base_epochs must be between 1 and num_epochs; setting it "
            "equal to num_epochs disables the second stage"
        )
    if opt.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if opt.captions_per_image <= 0:
        raise ValueError("--captions_per_image must be positive")
    if opt.refine_learning_rate <= 0:
        raise ValueError("--refine_learning_rate must be positive")
    if opt.resume and opt.init_checkpoint:
        raise ValueError("--resume and --init_checkpoint cannot be used together")

    second_stage_enabled = opt.base_epochs < opt.num_epochs
    grouped_batch_size = opt.batch_size * opt.captions_per_image
    base_loader_opt = copy.copy(opt)
    base_loader_opt.grouped_descriptions = False
    base_train_loader = image_caption.get_train_loader(
        base_loader_opt,
        opt.data_path,
        tokenizer,
        opt.batch_size,
        opt.workers,
        'train',
    )
    grouped_train_loader = None
    if second_stage_enabled:
        grouped_loader_opt = copy.copy(opt)
        grouped_loader_opt.grouped_descriptions = True
        grouped_train_loader = image_caption.get_train_loader(
            grouped_loader_opt,
            opt.data_path,
            tokenizer,
            grouped_batch_size,
            opt.workers,
            'train',
        )
    train_loader = base_train_loader
    print('Number of images for train-set:', train_loader.dataset.num_images)

    # # test-set
    split = 'testall' if opt.dataset == 'coco' else 'test'
    # Some prepared datasets contain only train/test. An explicit --val_split
    # can override this fallback when a separate development split is present.
    val_split = opt.val_split or split
    val_loader = image_caption.get_test_loader(
        opt,
        opt.data_path,
        tokenizer,
        grouped_batch_size,
        opt.workers,
        val_split,
    )

    # load model
    model = build_model(opt)
    if opt.init_checkpoint:
        load_shape_compatible_weights(model, opt.init_checkpoint)
    model = model.cuda()

    # get the optimizer
    optimizer = build_optimizer(opt, model)

    start_epoch = 0

    # multi-gpu
    if opt.multi_gpu:
        print('use multi gpu')
        model = torch.nn.parallel.DistributedDataParallel(module=model, 
                                                          device_ids=[opt.gpu], 
                                                          output_device=opt.gpu, 
                                                          find_unused_parameters=True,
                                                          )
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    


    best_rsum = 0

    scaler = torch.GradScaler()

    if opt.resume:
        if not os.path.isfile(opt.resume):
            raise FileNotFoundError(
                f"Resume checkpoint does not exist: {opt.resume}"
            )
        checkpoint = torch.load(
            opt.resume, map_location="cpu", weights_only=False
        )
        required = {"model", "optimizer", "scaler", "epoch"}
        missing = sorted(required.difference(checkpoint))
        if missing:
            raise RuntimeError(
                "Resume checkpoint is missing required states: "
                + ", ".join(missing)
            )
        resume_state = {
            (name[7:] if name.startswith("module.") else name): value
            for name, value in checkpoint["model"].items()
        }
        model_without_ddp.load_state_dict(resume_state, strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"])
        best_rsum = float(checkpoint.get("best_rsum", 0.0))
        model_without_ddp.Eiters = int(checkpoint.get("Eiters", 0))
        if start_epoch < 0 or start_epoch >= opt.num_epochs:
            raise ValueError(
                f"Checkpoint resumes at epoch {start_epoch}, but "
                f"--num_epochs is {opt.num_epochs}"
            )
        if utils.is_main_process():
            logger.info(
                "Resumed model, optimizer and AMP scaler from %s at epoch %d "
                "(best rsum %.1f, Eiters %d)",
                opt.resume,
                start_epoch,
                best_rsum,
                model_without_ddp.Eiters,
            )
            resumed_best_path = os.path.join(
                opt.model_name, "model_best.pth"
            )
            if not os.path.exists(resumed_best_path):
                torch.save(
                    {
                        "model": model_without_ddp.state_dict(),
                        "opt": opt,
                        "epoch": start_epoch,
                        "best_rsum": best_rsum,
                        "Eiters": model_without_ddp.Eiters,
                    },
                    resumed_best_path,
                )
                logger.info(
                    "Seeded the resumed run's model_best.pth from the "
                    "resume checkpoint"
                )

    # Train the Model
    for epoch in range(start_epoch, opt.num_epochs):
        set_refinement = False
        if opt.model_version == 'v5':
            set_refinement = (
                second_stage_enabled and epoch >= opt.base_epochs
            )
            train_loader = (
                grouped_train_loader
                if set_refinement
                else base_train_loader
            )
            model_without_ddp.set_grouped_training(set_refinement)

            if epoch == opt.base_epochs:
                refine_opt = copy.copy(opt)
                refine_opt.learning_rate = opt.refine_learning_rate
                optimizer = build_optimizer(refine_opt, model_without_ddp)
                scaler = torch.GradScaler()
                if utils.is_main_process():
                    logger.info(
                        "V5 phase switch at epoch %d: description-set "
                        "batches enabled, optimizer reset to task LR %.2e",
                        epoch,
                        opt.refine_learning_rate,
                    )



        sampler_epoch = epoch
        if opt.model_version == 'v5' and set_refinement:
            # Match the proven V3 continuation's local epoch sequence.
            sampler_epoch = epoch - opt.base_epochs
        if hasattr(train_loader.batch_sampler, 'set_epoch'):
            train_loader.batch_sampler.set_epoch(sampler_epoch)
        elif opt.multi_gpu:
            train_loader.sampler.set_epoch(sampler_epoch)

        if utils.is_main_process() and epoch == 0:
            logger.info('Log saving path: ' + opt.logger_name)
            logger.info('Models saving path: ' + opt.model_name)
            if opt.model_version == 'v5':
                if second_stage_enabled:
                    logger.info(
                        "V5 evidence phase: epochs 0-%d, ordinary batch size %d",
                        opt.base_epochs - 1,
                        opt.batch_size,
                    )
                else:
                    logger.info(
                        "V5 single-stage mode: epochs 0-%d use ordinary "
                        "batches; description-set refinement is disabled",
                        opt.num_epochs - 1,
                    )
        # if opt.dataset == 'coco' and 'swin' in opt.vit_type:
        #     adjust_learning_rate1(opt, optimizer, epoch)
        #     print("hhh")
        # else :
        #     adjust_learning_rate(opt, optimizer, epoch)
        adjust_learning_rate(opt, optimizer, epoch)

        # # set hard negative for vse loss
        if (epoch >= opt.vse_mean_warmup_epochs) and (opt.loss == 'vse'):
            model_without_ddp.set_max_violation(max_violation=True)
            if (
                utils.is_main_process()
                and epoch == opt.vse_mean_warmup_epochs
            ):
                logger.info(
                    "Hard-negative refinement starts at epoch %d",
                    epoch,
                )
            # model_without_ddp.img_token_compress_on()
            # model_without_ddp.set_triplet_loss()
            # pass
        # train for one epoch
        train(
            opt,
            train_loader,
            model,
            model_without_ddp,
            optimizer,
            epoch,
            scaler,
        )

        # test-set
        # split = 'testall' if opt.dataset == 'coco' else 'test'
        # test_loader = image_caption.get_test_loader(opt, opt.data_path, tokenizer, opt.batch_size, opt.workers, split)
        # evaluate on validation set
        rsum = validate(opt, val_loader, model_without_ddp, epoch=epoch)

        if utils.is_main_process():
            # remember best results and save checkpoint
            is_best = rsum > best_rsum
            best_rsum = max(rsum, best_rsum)

            logger.info("Epoch: [{}], Best rsum: {:.1f} \n".format(epoch, best_rsum))
            state = {'model': model_without_ddp.state_dict(), 'opt': opt, 'epoch': epoch + 1,
                     'best_rsum': best_rsum, 'Eiters': model_without_ddp.Eiters}
            file_name = f"model_best_{best_rsum}.pth"
            save_checkpoint(state, is_best, filename=file_name, prefix=opt.model_name)

            # Keep the validated epoch-15 state independently from model_best.
            # The loop uses zero-based epoch numbers, matching the value printed
            # in "Epoch: [15]" in the training log.
            if epoch == 15:
                epoch_state = dict(state)
                epoch_state.update({
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                })
                epoch_checkpoint = os.path.join(
                    opt.model_name, "model_epoch_15.pth"
                )
                torch.save(epoch_state, epoch_checkpoint)
                logger.info(
                    "Saved validated epoch-15 checkpoint with optimizer and "
                    "AMP scaler: %s",
                    epoch_checkpoint,
                )

        # waiting for synchronization
        if opt.multi_gpu:
            torch.distributed.barrier() 
            torch.cuda.empty_cache()   

    # start eval
    if utils.is_main_process() and opt.eval:
        print('Evaluate the model now.')

        base = opt.logger_name
        logging.basicConfig(filename=os.path.join(base, 'eval.log'), filemode='w', 
                            format='%(asctime)s %(message)s', level=logging.INFO, force=True)


        logger = logging.getLogger()
        logger.info('Evaluating {}...'.format(base))

        model_path = os.path.join(base, 'model_best.pth')
        
        # Save the final results for computing ensemble results
        save_path = os.path.join(base, 'results_{}.npy'.format(opt.dataset))

        if opt.dataset == 'coco':
            # Evaluate COCO 5-fold 1K
            evaluation.evalrank(model_path, model=model_without_ddp, split='testall', fold5=True)

            # Evaluate COCO 5K
            evaluation.evalrank(model_path, model=model_without_ddp, split='testall', fold5=False, save_path=save_path)

            if opt.evaluate_cxc:
                # Evaluate COCO-trained models on CxC
                evaluation.evalrank(model_path, model=model_without_ddp, split='testall', fold5=True, cxc=True)
        else:
            # Evaluate Flickr30K
            evaluation.evalrank(model_path, model=model_without_ddp, split='test', fold5=False, save_path=save_path)

        logger.info('Evaluation finish!')    

def description_coverage_scale(opt, epoch, batch_index, num_batches):
    """Return the V5 single-run curriculum scale in [0, 1]."""
    progress = float(epoch) + float(batch_index) / max(1, num_batches)
    start = float(opt.set_coverage_start_epoch)
    warmup = float(opt.set_coverage_warmup_epochs)
    if progress <= start:
        return 0.0
    if warmup <= 0:
        return 1.0
    return min(1.0, max(0.0, (progress - start) / warmup))


def train(
    opt,
    train_loader,
    model,
    model_without_ddp,
    optimizer,
    epoch,
    scaler,
):

    # switch to train mode
    model.train()   

    logger = logging.getLogger(__name__)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    if utils.is_main_process() and epoch == 0:
        logger.info('image encoder trainable parameters: {}M'.format(count_params(model_without_ddp.img_enc)))
        logger.info('txt encoder trainable parameters: {}M'.format(count_params(model_without_ddp.txt_enc)))
        logger.info('criterion trainable parameters: {}M'.format(count_params(model_without_ddp.criterion)))
    n_batch = len(train_loader) 

    end = time.time()

    for i, train_data in enumerate(train_loader):  
        optimizer.zero_grad()

        curriculum_epoch = epoch
        if opt.model_version == 'v5' and epoch >= opt.base_epochs:
            curriculum_epoch = epoch - opt.base_epochs
        # Restart the one-epoch auxiliary ramp when V5 enters refinement,
        # matching the separately validated V3 continuation.
        warmup_alpha = (
            float(i) / n_batch
            if curriculum_epoch == opt.embedding_warmup_epochs
            else 1.
        )
        # Coverage belongs to grouped-description refinement. Keep it exactly
        # zero in the ordinary phase, including full single-stage runs where a
        # batch may contain an accidental repeated image identity.
        coverage_alpha = 0.0
        if not (opt.model_version == 'v5' and epoch < opt.base_epochs):
            coverage_alpha = description_coverage_scale(
                opt, epoch, i, n_batch
            )
        # measure data loading time
        data_time.update(time.time() - end)


        images, captions, lengths, ids, img_ids = train_data

        # to device
        images = images.cuda(non_blocking=True)
        captions = captions.cuda(non_blocking=True)
        lengths = lengths.cuda(non_blocking=True) 
        img_ids = img_ids.cuda(non_blocking=True) 
        batch_weight = images.size(0)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            model_kwargs = {
                'img_ids': img_ids,
                'warmup_alpha': warmup_alpha,
                'isTrain': True,
            }
            model_kwargs['coverage_alpha'] = coverage_alpha
            loss,align_loss,cr_loss = model(
                images, captions, lengths, **model_kwargs
            )

        if torch.isnan(loss) or loss is None:
            print("WARNING: loss is NaN or None!")


        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.zeros([], requires_grad=True, device=images.device)

        scaler.scale(loss).backward()

        if opt.grad_clip > 0:
            # AMP stores scaled gradients after backward(). Unscale them before
            # measuring/clipping the norm so --grad_clip keeps its intended
            # value independently of GradScaler's dynamic scale.
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), opt.grad_clip)


        scaler.step(optimizer)
        scaler.update()

    


        batch_time.update(time.time() - end)
        end = time.time()    

        model_without_ddp.logger = train_logger
        model_without_ddp.logger.update('Iter', model_without_ddp.Eiters)
        model_without_ddp.logger.update('lr', optimizer.param_groups[0]['lr'])   
        model_without_ddp.logger.update('Loss', loss.item(), batch_weight)
        model_without_ddp.logger.update('AlignLoss', align_loss.item(), batch_weight)
        model_without_ddp.logger.update('AuxLoss', cr_loss.item(), batch_weight)
        criterion = getattr(model_without_ddp, 'criterion', None)
        pair_loss = getattr(criterion, 'last_pair_loss', None)
        coverage_loss = getattr(criterion, 'last_coverage_loss', None)
        if pair_loss is not None:
            model_without_ddp.logger.update(
                'PairRank', pair_loss.item(), batch_weight
            )
        if coverage_loss is not None:
            model_without_ddp.logger.update(
                'SetCoverage', coverage_loss.item(), batch_weight
            )
        unique_images = getattr(model_without_ddp, 'last_unique_images', None)
        if unique_images is not None:
            model_without_ddp.logger.update(
                'UniqueImages', float(unique_images), 1
            )
        model_without_ddp.logger.update(
            'TeacherDescriptions',
            float(model_without_ddp.last_teacher_descriptions),
            1,
        )
        model_without_ddp.logger.update(
            'CoverageScale', model_without_ddp.last_coverage_scale, 1
        )
        model_without_ddp.logger.update(
            'SetWeight',
            model_without_ddp.last_coverage_scale * opt.set_coverage_weight,
            1,
        )
        model_without_ddp.Eiters += 1

        if utils.is_main_process():
  
            if model_without_ddp.Eiters % opt.log_step == 0:  
                if curriculum_epoch == opt.embedding_warmup_epochs:
                    logging.info('The first epoch for training backbone, warmup alpha for loss is {}'.format(epoch, warmup_alpha))

                logging.info(
                    'Epoch: [{0}][{1}/{2}]\t'
                    '{e_log}\t'
                    'Batch-Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        .format(epoch, i+1, n_batch, batch_time=batch_time, data_time=data_time, e_log=str(model_without_ddp.logger)))


            # Record logs in tensorboard
            swan_metrics = {
                "loss": loss.detach().item(),
                "align_loss": align_loss.detach().item(),
                "cr_loss": cr_loss.detach().item(),
            }
            swanlab.log(swan_metrics)

        if i > n_batch:
            break

def apply_csls(scores, k=10, block_size=512):
    """Return CSLS-adjusted image-caption scores on CPU."""
    if scores.ndim != 2:
        raise ValueError("CSLS expects a 2-D score matrix")
    if k <= 0:
        raise ValueError("CSLS k must be positive")

    n_images, n_captions = scores.shape
    row_k = min(int(k), n_captions)
    col_k = min(int(k), n_images)
    row_neighbourhood = np.empty(n_images, dtype=np.float32)
    col_neighbourhood = np.empty(n_captions, dtype=np.float32)

    for start in range(0, n_images, block_size):
        end = min(start + block_size, n_images)
        block = scores[start:end]
        row_neighbourhood[start:end] = np.partition(
            block, n_captions - row_k, axis=1
        )[:, -row_k:].mean(axis=1)

    for start in range(0, n_captions, block_size):
        end = min(start + block_size, n_captions)
        block = np.ascontiguousarray(scores[:, start:end].T)
        col_neighbourhood[start:end] = np.partition(
            block, n_images - col_k, axis=1
        )[:, -col_k:].mean(axis=1)

    adjusted = 2.0 * scores
    adjusted -= row_neighbourhood[:, None]
    adjusted -= col_neighbourhood[None, :]
    return adjusted


def validation_csls_enabled(opt, epoch):
    """Enable diagnostic CSLS for CLIP or after V5 refinement starts."""
    if epoch is None or opt.val_csls_k <= 0:
        return False
    is_clip_backbone = "clip" in str(opt.vit_type).lower()
    return is_clip_backbone or epoch >= opt.base_epochs


def validate(opt, val_loader, model, epoch=None):

    logger = logging.getLogger(__name__)
    
    model.eval()

    with torch.no_grad():
       img_embs, cap_embs, cap_lens = encode_data(model, val_loader, opt.log_step, logging.info)

    # have repetitive image features
    if opt.dataset == "coco" or opt.dataset == "f30k":
       img_embs = img_embs[::5]

    start_time = time.time()

    if opt.multi_gpu:         
        sims = torch.zeros((len(img_embs), len(cap_embs))).cuda()
        
        num_tasks = utils.get_world_size()
        rank = utils.get_rank() 

        step = img_embs.size(0) // num_tasks + 1
        start = rank * step
        end = min(img_embs.size(0), start + step)

        sims_part = shard_attn_scores(model, img_embs[start:end], cap_embs, cap_lens, opt, gpu=True)
        sims[start:end] = sims_part

        # wait for synchronization 
        torch.distributed.barrier()
        # Aggregating results on different GPUs
        torch.distributed.all_reduce(sims, op=torch.distributed.ReduceOp.SUM) 
        sims = sims.cpu().numpy()
    else:
        sims = shard_attn_scores(model, img_embs, cap_embs, cap_lens, opt)      
        sims = sims.numpy()



    # compute metric
    if utils.is_main_process():
        
        logging.info("calculate similarity time: %.3f" % float(time.time() - start_time))

        npts = img_embs.shape[0]
        print(npts)
        print(img_embs.shape)
        print(sims.shape)
        # caption retrieval
        (r1, r5, r10, medr, meanr) = i2t(npts, sims,mode=opt.dataset)
        logging.info("Raw Image to text (R@1, R@5, R@10): %.1f, %.1f, %.1f" % (r1, r5, r10))

        # image retrieval
        (r1i, r5i, r10i, medri, meanri) = t2i(npts, sims,mode=opt.dataset)
        logging.info("Raw Text to image (R@1, R@5, R@10): %.1f, %.1f, %.1f" % (r1i, r5i, r10i))

        # sum of recalls to be used for early stopping
        currscore = r1 + r5 + r10 + r1i + r5i + r10i
        logger.info('Raw current rsum is {}'.format(round(currscore, 1)))

        # CLIP runs print the CSLS diagnostic from the first validation because
        # their pretrained embedding space is especially sensitive to hubness.
        # Other backbones keep the original second-stage-only behavior.  CSLS
        # never participates in checkpoint selection or back-propagation.
        csls_enabled = validation_csls_enabled(opt, epoch)
        csls_metrics = None
        if csls_enabled:
            csls_scores = apply_csls(sims, k=opt.val_csls_k)
            csls_i2t = i2t(npts, csls_scores, mode=opt.dataset)
            csls_t2i = t2i(npts, csls_scores, mode=opt.dataset)
            csls_rsum = sum(csls_i2t[:3]) + sum(csls_t2i[:3])
            logging.info(
                "CSLS(k=%d) Image to text (R@1, R@5, R@10): "
                "%.1f, %.1f, %.1f",
                opt.val_csls_k,
                csls_i2t[0],
                csls_i2t[1],
                csls_i2t[2],
            )
            logging.info(
                "CSLS(k=%d) Text to image (R@1, R@5, R@10): "
                "%.1f, %.1f, %.1f",
                opt.val_csls_k,
                csls_t2i[0],
                csls_t2i[1],
                csls_t2i[2],
            )
            logger.info(
                "CSLS(k=%d) diagnostic rsum is %.1f; checkpoint selection "
                "still uses raw rsum %.1f",
                opt.val_csls_k,
                csls_rsum,
                currscore,
            )
            csls_metrics = {
                "csls_r1": csls_i2t[0],
                "csls_r5": csls_i2t[1],
                "csls_r10": csls_i2t[2],
                "csls_r1i": csls_t2i[0],
                "csls_r5i": csls_t2i[1],
                "csls_r10i": csls_t2i[2],
                "csls_rsum": csls_rsum,
            }
            
        # record metrics in tensorboard
        validation_metrics = {"r1": r1,
                     "r5": r5,
                     "r10": r10,
                     "medr": medr,
                     "meanr": meanr,
                     "r1i": r1i,
                     "r5i": r5i,
                     "r10i": r10i,
                     "medri": medri,
                     'meanri': meanri,
                     "rsum": currscore}
        if csls_metrics is not None:
            validation_metrics.update(csls_metrics)
        swanlab.log(validation_metrics)
        return currscore

def save_checkpoint(state, is_best, filename='checkpoint.pth', prefix=''):

    if is_best:
        torch.save(state, os.path.join(prefix, 'model_best.pth'))

def adjust_learning_rate(opt, optimizer, epoch):
    logger = logging.getLogger(__name__)

    decay_rate = opt.decay_rate
    lr_schedules = opt.lr_schedules
    # Sets the learning rate to the initial LR
    if epoch in lr_schedules:
        logger.info('Current epoch num is {}, decrease all lr by {}'.format(epoch, decay_rate))
        for param_group in optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = old_lr * decay_rate
            param_group['lr'] = new_lr
            logger.info('new lr: {}'.format(new_lr))



def count_params(model):

    # The unit is M (million)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    params = round(params/(1024**2), 2)

    return params


if __name__ == '__main__':
    
    main()
