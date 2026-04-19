"""
AnomalySAM: 基于 SAM (Segment Anything Model) 的少样本异常检测

Usage:
    python run_anomalysam.py --dataset MVTec --shots 1 2 4 8 16 --num_seeds 3 --data_root data/mvtec_anomaly_detection
    python run_anomalysam.py --dataset VisA --shots 1 2 4 8 16 --num_seeds 3 --data_root data/VisA_pytorch/1cls/
"""

import argparse
import os
from argparse import ArgumentParser
import yaml
from tqdm import trange
import time
import json
import numpy as np

import csv
from collections import defaultdict

from src.utils import get_dataset_info
from src.detection import run_anomaly_detection
from src.post_eval import eval_finished_run
from src.visualize import create_sample_plots
from src.backbones_sam import get_sam_model


def aggregate_results_across_seeds(results_dir, seeds, args):
    """
    聚合所有 seed 的结果到一个汇总 JSON 文件中
    包含每个类别的均值和标准差，以及所有类别的平均值
    """
    all_metrics = []

    for seed in seeds:
        metric_file = f"{results_dir}/metrics_seed={seed}.json"
        if os.path.exists(metric_file):
            with open(metric_file, 'r') as f:
                metrics = json.load(f)
                all_metrics.append(metrics)

    if not all_metrics:
        print("Warning: No metrics files found for aggregation!")
        return

    # 获取所有对象名称（排除统计键）
    objects = [k for k in all_metrics[0].keys() if k not in 
               ['mean_au_pro', 'mean_segmentation_au_roc', 'mean_segmentation_f1',
                'mean_classification_au_roc', 'mean_classification_ap', 'mean_classification_f1']]

    # 聚合每个对象的指标
    aggregated = {}

    for obj in objects:
        aggregated[obj] = {}

        # 像素级指标 (segmentation)
        for metric in ['seg_AUPRO', 'seg_AUROC', 'seg_F1']:
            values = [m[obj].get(metric, None) for m in all_metrics if obj in m and metric in m[obj]]
            values = [v for v in values if v is not None]
            if values:
                aggregated[obj][f'{metric}_mean'] = float(np.mean(values))
                aggregated[obj][f'{metric}_std'] = float(np.std(values))

        # 图像级指标 (classification)
        for metric in ['classification_AUROC', 'classification_AP', 'classification_F1']:
            values = [m[obj].get(metric, None) for m in all_metrics if obj in m and metric in m[obj]]
            values = [v for v in values if v is not None]
            if values:
                aggregated[obj][f'{metric}_mean'] = float(np.mean(values))
                aggregated[obj][f'{metric}_std'] = float(np.std(values))

    # 计算所有类别的平均指标
    aggregated['summary'] = {}

    # 像素级平均
    for metric in ['seg_AUPRO', 'seg_AUROC', 'seg_F1']:
        values = []
        for obj in objects:
            key = f'{metric}_mean'
            if key in aggregated.get(obj, {}):
                values.append(aggregated[obj][key])
        if values:
            aggregated['summary'][f'mean_{metric}'] = float(np.mean(values))
            aggregated['summary'][f'mean_{metric}_std'] = float(np.std(values))

    # 图像级平均
    for metric in ['classification_AUROC', 'classification_AP', 'classification_F1']:
        values = []
        for obj in objects:
            key = f'{metric}_mean'
            if key in aggregated.get(obj, {}):
                values.append(aggregated[obj][key])
        if values:
            aggregated['summary'][f'mean_{metric}'] = float(np.mean(values))
            aggregated['summary'][f'mean_{metric}_std'] = float(np.std(values))

    # 添加实验配置信息
    aggregated['config'] = {
        'dataset': args.dataset,
        'model_type': args.model_type,
        'resolution': args.resolution,
        'preprocess': args.preprocess,
        'shots': args.shots,
        'num_seeds': args.num_seeds,
        'knn_metric': args.knn_metric
    }

    # 保存聚合结果
    output_file = f"{results_dir}/aggregated_metrics.json"
    with open(output_file, 'w') as f:
        json.dump(aggregated, f, indent=2)

    # 打印汇总结果
    print("\n" + "="*60)
    print("AGGREGATED RESULTS SUMMARY")
    print("="*60)
    print(f"Dataset: {args.dataset}, Model: sam_{args.model_type}, Resolution: {args.resolution}")
    print(f"Shots: {args.shots}, Seeds: {args.num_seeds}")
    print("-"*60)

    if 'summary' in aggregated:
        print("\nPixel-level (Segmentation) Metrics (mean ± std across seeds):")
        print(f"  AU-PRO:  {aggregated['summary'].get('mean_seg_AUPRO', 'N/A'):.4f} ± {aggregated['summary'].get('mean_seg_AUPRO_std', 'N/A'):.4f}")
        print(f"  AUROC:   {aggregated['summary'].get('mean_seg_AUROC', 'N/A'):.4f} ± {aggregated['summary'].get('mean_seg_AUROC_std', 'N/A'):.4f}")
        print(f"  F1:      {aggregated['summary'].get('mean_seg_F1', 'N/A'):.4f} ± {aggregated['summary'].get('mean_seg_F1_std', 'N/A'):.4f}")

        print("\nImage-level (Classification) Metrics (mean ± std across seeds):")
        print(f"  AUROC:   {aggregated['summary'].get('mean_classification_AUROC', 'N/A'):.4f} ± {aggregated['summary'].get('mean_classification_AUROC_std', 'N/A'):.4f}")
        print(f"  AP:      {aggregated['summary'].get('mean_classification_AP', 'N/A'):.4f} ± {aggregated['summary'].get('mean_classification_AP_std', 'N/A'):.4f}")
        print(f"  F1:      {aggregated['summary'].get('mean_classification_F1', 'N/A'):.4f} ± {aggregated['summary'].get('mean_classification_F1_std', 'N/A'):.4f}")

    print("="*60)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MVTec", 
                        help="Dataset name: MVTec or VisA")
    parser.add_argument("--model_type", type=str, default="vit_h", 
                        help="SAM model type: vit_h, vit_l, vit_b")
    parser.add_argument("--data_root", type=str, default="./MVTec",
                        help="Path to the root directory of the dataset.")
    parser.add_argument("--preprocess", type=str, default="agnostic",
                        help="Preprocessing method: agnostic, informed, masking_only")
    parser.add_argument("--resolution", type=int, default=448,
                        help="Image resolution (smaller edge size)")
    parser.add_argument("--knn_metric", type=str, default="L2_normalized",
                        help="Distance metric: L2_normalized (cosine) or L2")
    parser.add_argument("--k_neighbors", type=int, default=1,
                        help="Number of nearest neighbors for kNN search")
    parser.add_argument("--faiss_on_cpu", default=False, 
                        action=argparse.BooleanOptionalAction, 
                        help="Use CPU for FAISS kNN search (default: GPU)")
    parser.add_argument("--shots", nargs='+', type=int, default=[1],
                        help="List of shots to evaluate. Full-shot scenario is -1.")
    parser.add_argument("--num_seeds", type=int, default=1,
                        help="Number of random seeds for evaluation")
    parser.add_argument("--mask_ref_images", type=bool, default=False,
                        help="Whether to apply masking on reference images")
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
                        help="Path to SAM checkpoint file (required)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Optional tag for the saving directory")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # 验证 checkpoint 路径
    if args.sam_checkpoint is None:
        print("Error: --sam_checkpoint is required!")
        print("Please download SAM checkpoints from: https://github.com/facebookresearch/segment-anything")
        print("Default checkpoint names:")
        print("  - sam_vit_h_4b8939.pth")
        print("  - sam_vit_l_0b3195.pth")
        print("  - sam_vit_b_01ec64.pth")
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
    
    # 加载 SAM 模型 调用 src/backbones_sam.py 中的 get_sam_model()
    model = get_sam_model(
        model_type=args.model_type,
        device='cuda',
        smaller_edge_size=args.resolution,
        checkpoint_path=args.sam_checkpoint
    )

    # SAM 不支持 0-shot masking（暂时）
    masking_default = {o: False for o in objects}
    print("Note: SAM masking is not as mature as DINOv2. Using masking=False by default.")

    if args.just_seed is not None:
        seeds = [args.just_seed]
    else:
        seeds = range(args.num_seeds)

    for shot in list(args.shots):
        save_examples = args.save_examples

        # 结果保存目录
        model_name = f"sam_{args.model_type}"
        results_dir = f"E:/part/AnomalyDINO/results_MVTec/{model_name}_{args.resolution}/{shot}-shot_preprocess={args.preprocess}"

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
            print(f"=========== Shot = {shot}, Seed = {seed} ===========")

            if os.path.exists(f"{results_dir}/metrics_seed={seed}.json"):
                print(f"Results for shot {shot}, seed {seed} already exist. Skipping.")
                continue
            else:
                measurements_file = results_dir + f"/measurements_seed={seed}.csv"
                time_stats_file = results_dir + f"/time_statistics_seed={seed}.csv"

                # 用于统计时间的数据结构
                time_stats = defaultdict(lambda: defaultdict(list))  # {object: {anomaly_type: [times]}}
                all_inference_times = []  # 所有样本的推理时间

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
                            img_tensor, grid_size = model.prepare_image(
                                f"{args.data_root}/{object_name}/train/good/{first_image}")
                            features = model.extract_features(img_tensor)

                        # 运行异常检测
                        anomaly_scores, time_memorybank, time_inference = run_anomaly_detection(
                            model,
                            object_name,
                            data_root=args.data_root,
                            n_ref_samples=shot,
                            object_anomalies=object_anomalies,
                            plots_dir=plots_dir,
                            save_examples=save_examples,
                            masking=masking_default[object_name],
                            mask_ref_images=args.mask_ref_images,
                            rotation=rotation_default[object_name],
                            knn_metric=args.knn_metric,
                            knn_neighbors=args.k_neighbors,
                            faiss_on_cpu=args.faiss_on_cpu,
                            seed=seed,
                            save_patch_dists=args.eval_clf,
                            save_tiffs=args.eval_segm
                        )

                        # 写入异常分数和推理时间到 CSV
                        for counter, sample in enumerate(anomaly_scores.keys()):
                            anomaly_score = anomaly_scores[sample]
                            inference_time = time_inference[sample]

                            # 从 sample 路径中提取 anomaly_type
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

                # 统计并保存时间信息到 CSV
                with open(time_stats_file, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["Object", "Anomaly_Type", "Num_Samples", 
                                   "Avg_Time_Per_Sample(s)", "Total_Time(s)"])

                    all_object_avg_times = []  # 存储每个物体的平均时间

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

                        # 计算该物体所有类型的平均时间
                        object_avg_time = object_total_time / object_total_samples if object_total_samples > 0 else 0
                        all_object_avg_times.append(object_avg_time)

                        # 写入该物体的汇总行
                        writer.writerow([object_name, "ALL_TYPES", object_total_samples, 
                                       f"{object_avg_time:.6f}", f"{object_total_time:.6f}"])
                        writer.writerow([])  # 空行分隔

                # 计算所有类别的总平均时间
                if all_inference_times:
                    total_avg_time = sum(all_inference_times) / len(all_inference_times)
                    total_samples = len(all_inference_times)
                    total_time = sum(all_inference_times)

                    # 写入总平均行
                    with open(time_stats_file, 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(["MEAN_ALL_OBJECTS", "ALL_TYPES", total_samples, 
                                       f"{total_avg_time:.6f}", f"{total_time:.6f}"])

                    print(f"Finished AD for {len(objects)} objects (seed {seed}), "
                          f"mean inference time: {total_avg_time:.5f} s/sample = "
                          f"{total_samples/(total_time+1e-10):.2f} samples/s")

                # 检查是否存在 'good' 测试文件夹
                for object_name in objects:
                    good_folder = f"{args.data_root}/{object_name}/test/good/"
                    if not os.path.exists(good_folder):
                        print(f"Warning: 'good' folder not found for {object_name}! "
                              f"No evaluation will be performed for seed {seed}.")
                        print("Finished AD without evaluation, inference results saved to", results_dir)
                        break
                else:
                    # 评估并创建示例异常图
                    print(f"=========== Evaluate seed = {seed} ===========")
                    eval_finished_run(
                        args.dataset,
                        args.data_root,
                        anomaly_maps_dir=results_dir + f"/anomaly_maps/seed={seed}",
                        output_dir=results_dir,
                        seed=seed,
                        pro_integration_limit=0.3,
                        eval_clf=args.eval_clf,
                        eval_segm=args.eval_segm
                    )

                    create_sample_plots(
                        results_dir,
                        anomaly_maps_dir=results_dir + f"/anomaly_maps/seed={seed}",
                        seed=seed,
                        dataset=args.dataset,
                        data_root=args.data_root
                    )

                    # 关闭后续 seed 的示例创建
                    save_examples = False

    print("Finished and evaluated all runs!")
    print(f"\nResults saved to: {results_dir}")
    print(f"Key files:")
    print(f"  - metrics_seed=<seed>.json: 图像级和像素级评估指标")
    print(f"  - time_statistics_seed=<seed>.csv: 各类推理时间统计")
    print(f"  - measurements_seed=<seed>.csv: 每个样本的详细测量数据")

    # ========== 新增：聚合所有 seed 的结果到汇总 JSON ==========
    print("\n=========== Aggregating results across all seeds ===========")
    aggregate_results_across_seeds(results_dir, seeds, args)

    print(f"\nFinal aggregated results saved to: {results_dir}/aggregated_metrics.json")
