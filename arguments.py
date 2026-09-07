import argparse
import os

def get_argument_parser():
    """
    创建并配置命令行参数解析器

    Returns:
        argparse.ArgumentParser: 配置好的参数解析器
    """

    parser = argparse.ArgumentParser(
        description="视觉语义嵌入模型训练配置参数"
    )
    # ==================== 数据集相关参数 ====================
    parser.add_argument('--data_path', default='./data/', type=str,
                        help='数据集根目录路径')
    parser.add_argument('--dataset', default='f30k',
                        help='使用的数据集类型: coco 或 f30k')

    # ==================== 训练超参数 ====================
    parser.add_argument('--val_split', default='', type=str,
                        help='Checkpoint-selection split; empty uses test for f30k and testall for coco')

    parser.add_argument('--margin', default=0.2, type=float,
                        help='排名损失（Rank loss）的边界值（margin）')
    parser.add_argument('--num_epochs', default=30, type=int,
                        help='训练的总轮次（epochs）')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='基础阶段的图像身份数；分组及测试阶段自动乘 captions_per_image')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='联合嵌入空间的维度大小')
    parser.add_argument('--grad_clip', default=2.0, type=float,
                        help='梯度裁剪的阈值，防止梯度爆炸')
    parser.add_argument('--learning_rate', default=2e-4, type=float,
                        help='初始学习率')



    # ==================== 数据加载与日志参数 ====================
    parser.add_argument('--workers', default=8, type=int,
                        help='数据加载器使用的工作进程数')
    parser.add_argument('--log_step', default=200, type=int,
                        help='每隔多少步记录一次训练日志')
    parser.add_argument('--val_step', default=500, type=int,
                        help='每隔多少步进行一次验证')



    parser.add_argument('--logger_name', default='runs/test',
                        help='Tensorboard日志保存路径')

    # ==================== 损失函数相关参数 ====================
    parser.add_argument('--max_violation', action='store_true',
                        help='在排名损失中使用最大值而非求和（hard negative mining）')
    parser.add_argument('--vse_mean_warmup_epochs', type=int, default=1,
                        help='使用平均VSE损失的预热轮次数')
    parser.add_argument('--embedding_warmup_epochs', type=int, default=0,
                        help='嵌入层预热的轮次数')


    # ==================== 数据集路径参数 ====================
    parser.add_argument('--f30k_img_path', type=str, default='./data/flickr30k-images',
                        help='Flickr30K数据集图像路径')
    parser.add_argument('--coco_img_path', type=str, default='./data/coco-images',
                        help='MS-COCO数据集图像路径')


    # ==================== 视觉Transformer参数 ====================
    parser.add_argument('--img_res', type=int, default=224,
                        help='视觉骨干输入图像的分辨率（高度和宽度）')
    parser.add_argument('--vit_type', type=str, default='./save/vit-base',
                        help='本地Hugging Face视觉骨干路径：ViT、Swin或CLIP Vision')

    # ==================== 分布式训练参数 (DDP) ====================
    parser.add_argument('--multi_gpu', type=int, default=0,
                        help='是否使用多GPU训练：0-否，1-是')
    parser.add_argument('--world_size', type=int, default=1,
                        help='分布式训练的进程总数')
    parser.add_argument("--rank", type=int, default=0,
                        help='分布式训练中的进程排名')
    parser.add_argument("--local_rank", type=int, default=0,
                        help='分布式训练中的本地进程排名')
    parser.add_argument("--token_compress", action='store_true',
                        help='是否使用token压缩技术')


    parser.add_argument('--dist_backend', type=str, default='nccl',
                        help='分布式训练的后端（如nccl, gloo）')
    parser.add_argument('--dist_url', type=str, default='env://',
                        help='分布式训练的初始化URL')
    parser.add_argument('--seed', type=int, default=2026,
                        help='随机种子')

    # ==================== 其他训练参数 ====================
    parser.add_argument('--size_augment', type=int, default=1,
                        help='是否使用尺寸增强：0-否，1-是')
    parser.add_argument('--loss', type=str, default='vse',
                        help='优化目标函数类型：vse, infonce等')
    parser.add_argument('--eval', type=int, default=1,
                        help='训练结束后是否进行评估：0-否，1-是')

    parser.add_argument('--save_results', type=int, default=1,
                        help='是否保存评估结果：0-否，1-是')
    parser.add_argument('--evaluate_cxc', type=int, default=0,
                        help='是否对MS-COCO进行特殊的CxC评估：0-否，1-是')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='运行模型使用的GPU ID')


    # ==================== 文本编码器参数 ====================
    parser.add_argument('--bert_path', type=str, default='./save/bert-base-uncased',
                        help='BERT预训练模型路径')
    parser.add_argument('--hugging_face', action='store_false',
                        help='是否使用HuggingFace实现的Vision Transformer')
    # ==================== 优化器参数 ====================
    parser.add_argument("--lr_schedules", default=[9,15, 26], type=int, nargs="+",
                        help='学习率衰减的轮次计划（epoch schedules）')
    parser.add_argument("--decay_rate", default=0.3, type=float,
                        help='学习率衰减的比率')


    # ==================== 跨模态对齐参数 ====================
    parser.add_argument('--shard_size', type=int, default=256,
                        help='交叉注意力（cross-attention）的分片大小')
    parser.add_argument('--max_word', type=int, default=90,
                        help='词级特征的最大长度')

    parser.add_argument('--aggr_ratio', type=float, default=0.4,
                        help='视觉token的聚合比率（aggregation rate）')
    parser.add_argument('--sparse_ratio', type=float, default=0.5,
                        help='视觉token的稀疏化比率（sparsity rate）')
    parser.add_argument('--attention_weight', type=int, default=0.8,
                        help='用于掩码预测的注意力图权重')
    parser.add_argument('--ratio_weight', type=float, default=2.0,
                        help='如果使用分离（detach）的KT损失时的权重')
    parser.add_argument('--num_centers', type=int, default=2)
    parser.add_argument('--suppression_sigma', type=float, default=2.0,
                        help="TSR 距离抑制")
    # FocusDistill V5: language-guided visual evidence learning.
    parser.add_argument(
        '--model_version',
        choices=['v5'],
        default='v5',
    )
    parser.add_argument('--view_keep_ratio', type=float, default=0.8)
    parser.add_argument('--anchor_dim', type=int, default=256)
    parser.add_argument('--radial_alpha', type=float, default=1.0)
    parser.add_argument('--distill_temperature', type=float, default=0.2)
    parser.add_argument('--grounding_temperature', type=float, default=0.07)
    parser.add_argument('--match_temperature', type=float, default=0.07)
    parser.add_argument('--late_pool_temperature', type=float, default=0.05,
                        help='Soft-max pooling temperature; <=0 restores hard max')
    parser.add_argument('--global_score_weight', type=float, default=0.35,
                        help='Initial query-adaptive weight of the global image token')
    parser.add_argument('--block_dim', type=int, default=256)
    parser.add_argument('--anchor_kd_weight', type=float, default=1.0)
    parser.add_argument('--teacher_grounding_weight', type=float, default=0.2)
    parser.add_argument('--view_overlap_weight', type=float, default=0.05)
    # LAVE v3: grouped descriptions and set-level coverage ranking.
    parser.add_argument('--captions_per_image', type=int, default=2,
                        help='Grouped descriptions sampled for each image')
    parser.add_argument('--set_coverage_weight', type=float, default=0.1,
                        help='Weight of description-set coverage ranking')
    parser.add_argument('--set_temperature', type=float, default=0.1,
                        help='Smooth min/max temperature of set coverage')
    parser.add_argument('--set_coverage_start_epoch', type=float, default=17,
                        help='V4/V5 epoch at which description-set coverage starts')
    parser.add_argument('--set_coverage_warmup_epochs', type=float, default=1.0,
                        help='V4/V5 epochs used to linearly enable set coverage')
    parser.add_argument('--base_epochs', type=int, default=17,
                        help='Ordinary epochs before set refinement; set equal to num_epochs to disable stage two')
    parser.add_argument('--refine_learning_rate', type=float, default=0.00002,
                        help='Task learning rate after switching to set refinement')
    parser.add_argument('--init_checkpoint', type=str, default='',
                        help='Load shape-compatible weights before training')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume model, optimizer and AMP scaler from a training checkpoint')
    parser.add_argument('--val_csls_k', type=int, default=10,
                        help='Print diagnostic CSLS metrics from epoch 0 for CLIP and during second-stage validation for other backbones; <=0 disables it')
    return parser

def save_parameters(opt, save_path):
    """
    将配置参数保存到文件中

    Args:
        opt: 包含所有参数的命名空间或对象
        save_path: 参数保存路径
    """
    # 获取参数字典
    varx = vars(opt)
    base_str = ''

    # 将参数格式化为字符串
    for key in varx:
        base_str += str(key)
        if isinstance(varx[key], dict):
            # 如果参数值是字典，递归处理
            for sub_key, sub_item in varx[key].items():
                base_str += '\n\t' + str(sub_key) + ': ' + str(sub_item)
        else:
            base_str += '\n\t' + str(varx[key])
        base_str += '\n'

    # 写入文件
    with open(os.path.join(save_path, 'Parameters.txt'), 'w') as f:
        f.write(base_str)


if __name__ == '__main__':
    """
    参数解析器示例用法：


    1. 基本使用：
        python arguments.py --dataset coco --batch_size 64 --learning_rate 1e-4

    2. 多GPU训练：
        python arguments.py --multi_gpu 1 --world_size 4

    3. 使用特定ViT模型：
        python arguments.py --vit_type swin --img_res 384

    4. 查看所有参数及其默认值：
        python arguments.py --help
    """
    pass
