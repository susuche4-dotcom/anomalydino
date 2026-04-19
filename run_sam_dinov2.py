"""
SAM+DINOv2 联合异常检测（基于 AnomalyDINO）
使用 SAM 分割前景物体，SAM 失败时 fallback 到 PCA 方法

流程：
1. 尝试 SAM 分割 → 提取物体前景掩码
2. 如果 SAM 失败（置信度低）→ 使用 PCA 方法
3. DINOv2 提取特征 → 构建前景特征记忆库
4. kNN 匹配 → 定位异常
"""

import argparse
import os
from argparse import ArgumentParser
import yaml
from tqdm import trange
import time
import json
import numpy as np
import cv2

import csv
from collections import defaultdict

from src.utils import get_dataset_info
from src.detection import run_anomaly_detection
from src.post_eval import eval_finished_run
from src.visualize import create_sample_plots
from src.backbones import get_model
from src.backbones_sam_prompt import get_sam_model


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MVTec",
                        help="Dataset name: MVTec or VisA")
    parser.add_argument("--dinov2_model", type=str, default="dinov2_vits14",
                        help="DINOv2 model name")
    parser.add_argument("--sam_model_type", type=str, default="vit_b",
                        help="SAM model type: vit_h, vit_l, vit_b")
    parser.add_argument("--data_root", type=str, default="./MVTec",
                        help="Path to the root directory of the dataset.")
    parser.add_argument("--preprocess", type=str, default="agnostic",
                        help="Preprocessing method")
    parser.add_argument("--resolution", type=int, default=448,
                        help="Image resolution (smaller edge size)")
    parser.add_argument("--knn_metric", type=str, default="L2_normalized",
                        help="Distance metric for kNN")
    parser.add_argument("--k_neighbors", type=int, default=1,
                        help="Number of nearest neighbors for kNN search")
    parser.add_argument("--faiss_on_cpu", default=False,
                        action=argparse.BooleanOptionalAction,
                        help="Use CPU for FAISS kNN search")
    parser.add_argument("--shots", nargs='+', type=int, default=[1],
                        help="List of shots to evaluate")
    parser.add_argument("--num_seeds", type=int, default=1,
                        help="Number of random seeds for evaluation")
    parser.add_argument("--mask_ref_images", type=bool, default=True,
                        help="Whether to apply SAM masking on reference images")
    parser.add_argument("--just_seed", type=int, default=None,
                        help="Run only a specific seed")
    parser.add_argument('--save_examples', default=True,
                        action=argparse.BooleanOptionalAction,
                        help="Save example plots")
    parser.add_argument("--eval_clf", default=True,
                        action=argparse.BooleanOptionalAction,
                        help="Evaluate anomaly classification performance")
    parser.add_argument("--eval_segm", default=True,
                        action=argparse.BooleanOptionalAction,
                        help="Evaluate anomaly segmentation performance")
    parser.add_argument("--device", default='cuda:0',
                        help="CUDA device")
    parser.add_argument("--warmup_iters", type=int, default=25,
                        help="Number of warmup iterations for CUDA benchmarking")
    parser.add_argument("--sam_checkpoint", type=str, default=None,
                        help="Path to SAM checkpoint file")
    parser.add_argument("--tag", type=str, default=None,
                        help="Optional tag for the saving directory")
    parser.add_argument("--sam_prompt_type", type=str, default="bbox",
                        choices=['bbox', 'point', 'auto'],
                        help="Type of prompt for SAM: bbox, point, or auto")
    parser.add_argument("--sam_confidence_threshold", type=float, default=0.3,
                        help="Confidence threshold for SAM masks (below this use PCA)")
    parser.add_argument("--verbose_timing", default=True,
                        action=argparse.BooleanOptionalAction,
                        help="Save detailed timing breakdown to JSON")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # 验证 checkpoint 路径
    if args.sam_checkpoint is None:
        print("Error: --sam_checkpoint is required!")
        print("Please download SAM checkpoints from: https://github.com/facebookresearch/segment-anything")
        exit(1)

    if not os.path.exists(args.sam_checkpoint):
        print(f"Error: SAM checkpoint not found at: {args.sam_checkpoint}")
        exit(1)

    print(f"Requested to run {len(args.shots)} (different) shot(s):", args.shots)
    print(f"Requested to repeat the experiments {args.num_seeds} time(s).")

    # 获取数据集信息
    objects, object_anomalies, masking_default, rotation_default = get_dataset_info(
        args.dataset, args.preprocess, data_path=args.data_root)

    # 设置 CUDA 设备
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device[-1])

    # 加载 DINOv2 模型（用于特征提取）
    print("\n" + "="*60)
    print("Loading DINOv2 model for feature extraction...")
    print("="*60)
    dinov2_model = get_model(args.dinov2_model, 'cuda', smaller_edge_size=args.resolution)

    # 加载 SAM 模型（用于前景分割）
    print("\n" + "="*60)
    print("Loading SAM model for foreground segmentation...")
    print("="*60)
    sam_model = get_sam_model(
        model_type=args.sam_model_type,
        device='cuda',
        smaller_edge_size=args.resolution,
        checkpoint_path=args.sam_checkpoint
    )

    # SAM 强制启用 masking
    masking_default = {o: True for o in objects}
    print("\nNote: SAM masking is ENABLED for all objects (foreground extraction)")

    if args.just_seed is not None:
        seeds = [args.just_seed]
    else:
        seeds = range(args.num_seeds)

    for shot in list(args.shots):
        save_examples = args.save_examples

        # 结果保存目录
        model_name = f"sam_{args.sam_model_type}+{args.dinov2_model}_{args.resolution}"
        results_dir = f"E:/part/AnomalyDINO/results_MVTec/{model_name}/{shot}-shot_preprocess={args.preprocess}"

        if args.tag is not None:
            results_dir += "_" + args.tag

        plots_dir = results_dir
        os.makedirs(f"{results_dir}", exist_ok=True)

        # 保存预处理配置
        with open(f"{results_dir}/preprocess.yaml", "w") as f:
            yaml.dump({"masking": masking_default, "rotation": rotation_default}, f)

        # 保存参数配置
        with open(f"{results_dir}/args.yaml", "w") as f:
            yaml.dump(vars(args), f)

        if args.faiss_on_cpu:
            print("Warning: Running similarity search on CPU. Consider using faiss-gpu for faster inference.")

        print("Results will be saved to", results_dir)

        for seed in seeds:
            print(f"\n{'='*70}")
            print(f"Shot = {shot}, Seed = {seed}")
            print(f"{'='*70}")

            if os.path.exists(f"{results_dir}/metrics_seed={seed}.json"):
                print(f"Results for shot {shot}, seed {seed} already exist. Skipping.")
                continue

            measurements_file = results_dir + f"/measurements_seed={seed}.csv"
            time_stats_file = results_dir + f"/time_statistics_seed={seed}.csv"

            # 用于统计时间的数据结构
            time_stats = defaultdict(lambda: defaultdict(list))
            all_inference_times = []

            with open(measurements_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Object", "Anomaly_Type", "Sample", "Anomaly_Score",
                               "MemoryBank_Time", "Inference_Time"])

                for object_name in objects:
                    if save_examples:
                        os.makedirs(f"{plots_dir}/{object_name}", exist_ok=True)
                        os.makedirs(f"{plots_dir}/{object_name}/examples", exist_ok=True)

                    # CUDA warmup
                    for _ in trange(args.warmup_iters, desc="CUDA warmup", leave=False):
                        first_image = os.listdir(f"{args.data_root}/{object_name}/train/good")[0]
                        img_tensor, grid_size = dinov2_model.prepare_image(
                            f"{args.data_root}/{object_name}/train/good/{first_image}")
                        features = dinov2_model.extract_features(img_tensor)

                    # 运行异常检测（使用 SAM 分割 + DINOv2 特征）
                    anomaly_scores, time_memorybank, time_inference = run_anomaly_detection(
                        dinov2_model,
                        object_name,
                        data_root=args.data_root,
                        n_ref_samples=shot,
                        object_anomalies=object_anomalies,
                        plots_dir=plots_dir,
                        save_examples=save_examples,
                        masking=masking_default[object_name],  # SAM 分割的前景掩码
                        mask_ref_images=args.mask_ref_images,
                        rotation=rotation_default[object_name],
                        knn_metric=args.knn_metric,
                        knn_neighbors=args.k_neighbors,
                        faiss_on_cpu=args.faiss_on_cpu,
                        seed=seed,
                        save_patch_dists=args.eval_clf,
                        save_tiffs=args.eval_segm
                    )

                    # 写入 CSV
                    for sample in anomaly_scores.keys():
                        anomaly_score = anomaly_scores[sample]
                        inference_time = time_inference[sample]
                        anomaly_type = sample.split('/')[0] if '/' in sample else 'unknown'

                        writer.writerow([
                            object_name,
                            anomaly_type,
                            sample,
                            f"{anomaly_score:.5f}",
                            f"{time_memorybank:.5f}",
                            f"{inference_time:.5f}"
                        ])

                        # 统计时间
                        time_stats[object_name][anomaly_type].append(inference_time)
                        all_inference_times.append(inference_time)

            # 保存时间统计
            with open(time_stats_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Object", "Anomaly_Type", "Num_Samples",
                               "Avg_Time_Per_Sample(s)", "Total_Time(s)"])

                for object_name in time_stats.keys():
                    object_total_samples = 0
                    object_total_time = 0

                    for anomaly_type in sorted(time_stats[object_name].keys()):
                        times = time_stats[object_name][anomaly_type]
                        num_samples = len(times)
                        avg_time = sum(times) / num_samples if num_samples > 0 else 0
                        total_time = sum(times)

                        writer.writerow([
                            object_name,
                            anomaly_type,
                            num_samples,
                            f"{avg_time:.6f}",
                            f"{total_time:.6f}"
                        ])

                        object_total_samples += num_samples
                        object_total_time += total_time

                    object_avg_time = object_total_time / object_total_samples if object_total_samples > 0 else 0
                    writer.writerow([object_name, "ALL_TYPES", object_total_samples,
                                   f"{object_avg_time:.6f}", f"{object_total_time:.6f}"])
                    writer.writerow([])

                # 计算所有类别的总平均时间
                if all_inference_times:
                    total_avg_time = sum(all_inference_times) / len(all_inference_times)
                    total_samples = len(all_inference_times)
                    total_time = sum(all_inference_times)

                    writer.writerow(["MEAN_ALL_OBJECTS", "ALL_TYPES", total_samples,
                                   f"{total_avg_time:.6f}", f"{total_time:.6f}"])

                    print(f"\nFinished AD for {len(objects)} objects (seed {seed}), "
                          f"mean inference time: {total_avg_time:.5f} s/sample = "
                          f"{total_samples/(total_time+1e-10):.2f} samples/s")

            # 检查是否存在 'good' 测试文件夹
            skip_evaluation = False
            for object_name in objects:
                good_folder = f"{args.data_root}/{object_name}/test/good/"
                if not os.path.exists(good_folder):
                    print(f"Warning: 'good' folder not found for {object_name}!")
                    skip_evaluation = True
                    break

            if not skip_evaluation:
                # 评估
                print(f"\n{'='*70}")
                print(f"Evaluating seed = {seed}...")
                print(f"{'='*70}")

                eval_finished_run(
                    args.dataset,
                    args.data_root,
                    anomaly_maps_dir=results_dir + f"/anomaly_maps/seed={seed}",
                    output_dir=results_dir,
                    seed=seed,
                    pro_integration_limit=0.3,
                    eval_clf=args.eval_clf,
                    eval_segm=args.eval_segm,
                    time_stats_file=time_stats_file  # 传入时间统计文件
                )

                if args.save_examples:
                    create_sample_plots(
                        results_dir,
                        anomaly_maps_dir=results_dir + f"/anomaly_maps/seed={seed}",
                        seed=seed,
                        dataset=args.dataset,
                        data_root=args.data_root
                    )

                # 关闭后续 seed 的示例创建
                save_examples = False

    print(f"\n{'='*70}")
    print("Finished and evaluated all runs!")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*70}")
