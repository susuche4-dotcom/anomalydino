"""
SAM + DINOv2 软掩码融合异常检测 (Soft Mask Fusion)
==================================================

核心创新：
1. 真正的 SAM 融合：修复了原版代码中 SAM 未参与掩码计算的 Bug。
2. 软掩码策略 (Soft Mask)：解决 SAM 硬截断导致边缘缺陷漏检的问题。
3. 记忆库提纯：利用 SAM 剔除参考图像中的背景特征，提升记忆库纯度。

运行命令:
D:\anaconda\envs\A_Dino\python.exe E:\part\AnomalyDINO\run_sam_dinov2_soft_mask.py ^
    --dataset MVTec ^
    --dinov2_model dinov2_vits14 ^
    --sam_model_type vit_b ^
    --shots 1 ^
    --num_seeds 1 ^
    --sam_checkpoint E:/part/AnomalyDINO/sam_vit_b_01ec64.pth ^
    --data_root E:/part/AnomalyDINO/MVTec ^
    --preprocess agnostic ^
    --resolution 448
"""

import argparse
import os
from argparse import ArgumentParser
import yaml
from tqdm import trange, tqdm
import time
import json
import numpy as np
import cv2
import faiss
import torch
import tifffile as tiff
import csv
import matplotlib.pyplot as plt
from collections import defaultdict

from src.utils import get_dataset_info, augment_image, dists2map, plot_ref_images
from src.post_eval import eval_finished_run, mean_top1p
from src.visualize import create_sample_plots
from src.backbones import get_model
from src.backbones_sam_prompt import get_sam_model


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MVTec")
    parser.add_argument("--dinov2_model", type=str, default="dinov2_vits14")
    parser.add_argument("--sam_model_type", type=str, default="vit_b")
    parser.add_argument("--data_root", type=str, default="./MVTec")
    parser.add_argument("--preprocess", type=str, default="agnostic")
    parser.add_argument("--resolution", type=int, default=448)
    parser.add_argument("--knn_metric", type=str, default="L2_normalized")
    parser.add_argument("--k_neighbors", type=int, default=1)
    parser.add_argument("--faiss_on_cpu", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--shots", nargs='+', type=int, default=[1])
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--just_seed", type=int, default=None)
    parser.add_argument('--save_examples', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--eval_clf", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--eval_segm", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--device", default='cuda:0')
    parser.add_argument("--warmup_iters", type=int, default=25)
    parser.add_argument("--sam_checkpoint", type=str, default=None)
    parser.add_argument("--tag", type=str, default=None)
    
    # 软掩码参数
    parser.add_argument("--sam_confidence_threshold", type=float, default=0.3,
                        help="SAM 置信度阈值 (低于此值回退到 PCA)")
    parser.add_argument("--ref_mask_threshold", type=float, default=0.6,
                        help="参考图像掩码阈值 (仅存入覆盖率 > 60% 的纯净特征)")
    parser.add_argument("--test_mask_threshold", type=float, default=0.2,
                        help="测试图像掩码阈值 (覆盖率 > 20% 参与检测)")

    args = parser.parse_args()
    return args


def compute_patch_coverage(mask, grid_size):
    """
    将高分辨率 Mask 转换为 DINO 网格级别的覆盖率 (Soft Mask)
    
    参数:
        mask: (H, W) 二值掩码 (0/1)
        grid_size: (H_grid, W_grid)
    
    返回:
        coverage: (H_grid * W_grid,) 浮点数数组，表示每个 patch 被物体覆盖的比例
    """
    # INTER_AREA 在缩小时会计算区域内的平均值，正好对应覆盖率！
    coverage = cv2.resize(mask.astype(np.float32), (grid_size[1], grid_size[0]), interpolation=cv2.INTER_AREA)
    return coverage.flatten()


def run_sam_dino_detection(
        dinov2_model,
        sam_model,
        object_name,
        data_root,
        n_ref_samples,
        object_anomalies,
        plots_dir,
        save_examples=False,
        masking=True, # 强制开启
        rotation=False,
        knn_metric='L2_normalized',
        knn_neighbors=1,
        faiss_on_cpu=False,
        seed=0,
        save_patch_dists=True,
        save_tiffs=False,
        sam_confidence_threshold=0.3,
        ref_mask_threshold=0.6,
        test_mask_threshold=0.2):
    
    print(f"\n[SAM+DINO Soft Mask] Processing {object_name}...")

    type_anomalies = object_anomalies[object_name]
    good_folder = f"{data_root}/{object_name}/test/good/"
    if os.path.exists(good_folder):
        type_anomalies.append('good')
    type_anomalies = list(set(type_anomalies))

    # ========================================================================
    # 阶段 1: 构建记忆库 (利用 SAM 提纯)
    # ========================================================================
    print(f"[Memory Bank] Building with SAM filtering (Threshold: {ref_mask_threshold})...")
    
    features_ref_list = []
    images_ref = []
    masks_ref = []

    img_ref_folder = f"{data_root}/{object_name}/train/good/"
    if n_ref_samples == -1:
        img_ref_samples = sorted(os.listdir(img_ref_folder))
    else:
        img_ref_samples = sorted(os.listdir(img_ref_folder))[seed*n_ref_samples:(seed+1)*n_ref_samples]

    with torch.inference_mode():
        start_time = time.time()
        for img_ref_n in tqdm(img_ref_samples, desc="Building Bank", leave=False):
            img_ref_path = f"{img_ref_folder}{img_ref_n}"
            image_ref = cv2.cvtColor(cv2.imread(img_ref_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            
            if rotation:
                imgs_aug = augment_image(image_ref)
            else:
                imgs_aug = [image_ref]

            for img_aug in imgs_aug:
                # 1. SAM 分割
                sam_mask, method = sam_model.get_foreground_mask(
                    img_aug, object_name, 
                    prompt_type='bbox', 
                    confidence_threshold=sam_confidence_threshold, 
                    use_pca_fallback=True
                )

                # 2. 获取 DINO 特征
                image_tensor, grid_size = dinov2_model.prepare_image(img_aug)
                features = dinov2_model.extract_features(image_tensor)

                # 3. 计算 Patch 覆盖率
                if method == 'pca' or sam_mask is None:
                    # Fallback 到 PCA 掩码
                    pca_mask = dinov2_model.compute_background_mask(features, grid_size, threshold=10, masking_type=True)
                    coverage = pca_mask.astype(np.float32)
                else:
                    coverage = compute_patch_coverage(sam_mask, grid_size)

                # 4. 提纯：只保留物体中心区域 (高覆盖率) 的特征
                # 这能防止边缘背景噪声混入记忆库
                valid_indices = np.where(coverage >= ref_mask_threshold)[0]
                
                if len(valid_indices) > 0:
                    features_ref_list.append(features[valid_indices])
                    
                    if save_examples:
                        images_ref.append(image_ref)
                        masks_ref.append(coverage.reshape(grid_size))
        
        if not features_ref_list:
            print("Warning: No reference features extracted!")
            return {}, 0, {}

        features_ref = np.concatenate(features_ref_list, axis=0).astype('float32')
        print(f"[Memory Bank] Stored {features_ref.shape[0]} purified patches.")

        # 构建 FAISS 索引
        if knn_metric == "L2_normalized":
            faiss.normalize_L2(features_ref)
        
        use_cpu = faiss_on_cpu or not hasattr(faiss, 'StandardGpuResources')
        if use_cpu:
            index = faiss.IndexFlatL2(features_ref.shape[1])
        else:
            res = faiss.StandardGpuResources()
            index = faiss.GpuIndexFlatL2(res, features_ref.shape[1])
        
        index.add(features_ref)
        time_memorybank = time.time() - start_time

    # ========================================================================
    # 阶段 2: 测试 (软掩码加权检测)
    # ========================================================================
    inference_times = {}
    anomaly_scores = {}
    
    # 诊断计时
    timing_stats = {'sam': [], 'dino': [], 'mask': [], 'knn': [], 'total': []}

    for type_anomaly in tqdm(type_anomalies, desc="Testing"):
        data_dir = f"{data_root}/{object_name}/test/{type_anomaly}"
        if not os.path.exists(data_dir):
            continue

        if save_patch_dists or save_tiffs:
            os.makedirs(f"{plots_dir}/anomaly_maps/seed={seed}/{object_name}/test/{type_anomaly}", exist_ok=True)

        for idx, img_test_nr in tqdm(enumerate(sorted(os.listdir(data_dir))), 
                                     desc=f"'{type_anomaly}'", leave=False, total=len(os.listdir(data_dir))):
            
            start_total = time.time()
            image_test_path = f"{data_dir}/{img_test_nr}"
            image_test = cv2.cvtColor(cv2.imread(image_test_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

            # 1. SAM 分割
            t0 = time.time()
            sam_mask, method = sam_model.get_foreground_mask(
                image_test, object_name, 
                prompt_type='bbox', 
                confidence_threshold=sam_confidence_threshold, 
                use_pca_fallback=True
            )
            timing_stats['sam'].append(time.time() - t0)

            # 2. DINO 特征提取
            t1 = time.time()
            image_tensor, grid_size = dinov2_model.prepare_image(image_test)
            features = dinov2_model.extract_features(image_tensor)
            timing_stats['dino'].append(time.time() - t1)

            # 3. 计算覆盖率
            t2 = time.time()
            if method == 'pca' or sam_mask is None:
                pca_mask = dinov2_model.compute_background_mask(features, grid_size, threshold=10, masking_type=True)
                coverage = pca_mask.astype(np.float32)
            else:
                coverage = compute_patch_coverage(sam_mask, grid_size)
            timing_stats['mask'].append(time.time() - t2)

            # 4. 筛选检测区域 (Threshold: 0.2)
            valid_indices = np.where(coverage >= test_mask_threshold)[0]
            
            # 初始化最终距离图
            final_distances = np.zeros(features.shape[0], dtype=np.float32)

            if len(valid_indices) > 0:
                # 5. kNN 搜索
                t3 = time.time()
                query_features = features[valid_indices]
                if knn_metric == "L2_normalized":
                    faiss.normalize_L2(query_features)
                
                distances, _ = index.search(query_features, k=knn_neighbors)
                if knn_neighbors > 1:
                    distances = distances.mean(axis=1)
                distances = distances / 2  # Cosine distance
                timing_stats['knn'].append(time.time() - t3)

                # 6. 软加权融合 (关键创新!)
                # 距离图 × 覆盖率：边缘缺陷保留但权重降低，背景噪声彻底压制
                patch_weights = coverage[valid_indices]
                weighted_dists = distances.flatten() * patch_weights
                
                final_distances[valid_indices] = weighted_dists
            else:
                timing_stats['knn'].append(0)

            # 记录时间
            torch.cuda.synchronize()
            total_time = time.time() - start_total
            timing_stats['total'].append(total_time)
            
            inference_times[f"{type_anomaly}/{img_test_nr}"] = total_time
            anomaly_scores[f"{type_anomaly}/{img_test_nr}"] = mean_top1p(final_distances.flatten())

            # 保存结果
            img_test_nr_no_ext = img_test_nr.split(".")[0]
            if save_tiffs:
                anomaly_map = dists2map(final_distances.reshape(grid_size), image_test.shape)
                tiff.imwrite(f"{plots_dir}/anomaly_maps/seed={seed}/{object_name}/test/{type_anomaly}/{img_test_nr_no_ext}.tiff", anomaly_map)
            if save_patch_dists:
                np.save(f"{plots_dir}/anomaly_maps/seed={seed}/{object_name}/test/{type_anomaly}/{img_test_nr_no_ext}.npy", final_distances.reshape(grid_size))
    
    # 打印诊断报告
    print(f"\n{'='*20} [SAM+DINO 诊断报告: {object_name}] {'='*20}")
    print(f"1. SAM 耗时: {np.mean(timing_stats['sam'])*1000:.2f} ms")
    print(f"2. DINO 提取: {np.mean(timing_stats['dino'])*1000:.2f} ms")
    print(f"3. Mask/覆盖率: {np.mean(timing_stats['mask'])*1000:.2f} ms")
    print(f"4. kNN 搜索: {np.mean(timing_stats['knn'])*1000:.2f} ms")
    print(f"5. 总耗时: {np.mean(timing_stats['total'])*1000:.2f} ms")
    print(f"{'='*60}\n")

    return anomaly_scores, time_memorybank, inference_times


if __name__ == "__main__":
    args = parse_args()

    if args.sam_checkpoint is None or not os.path.exists(args.sam_checkpoint):
        print("Error: SAM checkpoint missing!")
        exit(1)

    print(f"Dataset: {args.dataset}, Model: {args.dinov2_model} + SAM({args.sam_model_type})")
    print(f"Soft Mask Fusion Enabled: Ref>{args.ref_mask_threshold}, Test>{args.test_mask_threshold}")

    objects, object_anomalies, masking_default, rotation_default = get_dataset_info(
        args.dataset, args.preprocess, data_path=args.data_root)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device[-1])

    # 加载模型
    print("Loading DINOv2...")
    dinov2_model = get_model(args.dinov2_model, 'cuda', smaller_edge_size=args.resolution)
    
    print("Loading SAM...")
    sam_model = get_sam_model(
        model_type=args.sam_model_type,
        device='cuda',
        smaller_edge_size=args.resolution,
        checkpoint_path=args.sam_checkpoint
    )

    if args.just_seed is not None:
        seeds = [args.just_seed]
    else:
        seeds = range(args.num_seeds)

    for shot in list(args.shots):
        save_examples = args.save_examples
        results_dir = f"E:/part/AnomalyDINO/results_MVTec/sam_dino_soft_mask_{args.dinov2_model}_{args.resolution}/{shot}-shot_preprocess={args.preprocess}"
        if args.tag:
            results_dir += "_" + args.tag
        plots_dir = results_dir
        os.makedirs(f"{results_dir}", exist_ok=True)

        with open(f"{results_dir}/args.yaml", "w") as f:
            yaml.dump(vars(args), f)

        for seed in seeds:
            print(f"\n{'='*70}")
            print(f"Shot={shot}, Seed={seed}")
            print(f"{'='*70}")

            if os.path.exists(f"{results_dir}/metrics_seed={seed}.json"):
                print("Exists, skipping.")
                continue

            measurements_file = f"{results_dir}/measurements_seed={seed}.csv"
            time_stats_file = f"{results_dir}/time_statistics_seed={seed}.csv"
            time_stats = defaultdict(lambda: defaultdict(list))
            all_inference_times = []

            with open(measurements_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Object", "Anomaly_Type", "Sample", "Anomaly_Score", "MemoryBank_Time", "Inference_Time"])

                for object_name in objects:
                    if save_examples:
                        os.makedirs(f"{plots_dir}/{object_name}/examples", exist_ok=True)

                    # CUDA Warmup
                    for _ in trange(args.warmup_iters, desc="Warmup", leave=False):
                        first_image = os.listdir(f"{args.data_root}/{object_name}/train/good")[0]
                        img_tensor, _ = dinov2_model.prepare_image(f"{args.data_root}/{object_name}/train/good/{first_image}")
                        dinov2_model.extract_features(img_tensor)
                        sam_model.get_foreground_mask(
                            cv2.imread(f"{args.data_root}/{object_name}/train/good/{first_image}"), 
                            object_name, 'bbox', 0.3, True)

                    # 运行检测
                    anomaly_scores, time_memorybank, time_inference = run_sam_dino_detection(
                        dinov2_model,
                        sam_model,
                        object_name,
                        data_root=args.data_root,
                        n_ref_samples=shot,
                        object_anomalies=object_anomalies,
                        plots_dir=plots_dir,
                        save_examples=save_examples,
                        masking=True,
                        rotation=rotation_default[object_name],
                        knn_metric=args.knn_metric,
                        knn_neighbors=args.k_neighbors,
                        faiss_on_cpu=args.faiss_on_cpu,
                        seed=seed,
                        save_patch_dists=args.eval_clf,
                        save_tiffs=args.eval_segm,
                        sam_confidence_threshold=args.sam_confidence_threshold,
                        ref_mask_threshold=args.ref_mask_threshold,
                        test_mask_threshold=args.test_mask_threshold
                    )

                    # 写入 CSV
                    for sample in anomaly_scores.keys():
                        anomaly_type = sample.split('/')[0] if '/' in sample else 'unknown'
                        writer.writerow([object_name, anomaly_type, sample, 
                                       f"{anomaly_scores[sample]:.5f}", f"{time_memorybank:.5f}", 
                                       f"{time_inference[sample]:.5f}"])
                        
                        time_stats[object_name][anomaly_type].append(time_inference[sample])
                        all_inference_times.append(time_inference[sample])

            # 保存时间统计
            with open(time_stats_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Object", "Anomaly_Type", "Num_Samples", "Avg_Time_Per_Sample(s)", "Total_Time(s)"])
                for obj_name in time_stats.keys():
                    for atype in sorted(time_stats[obj_name].keys()):
                        times = time_stats[obj_name][atype]
                        writer.writerow([obj_name, atype, len(times), f"{np.mean(times):.6f}", f"{np.sum(times):.6f}"])
                    # Summary
                    all_t = [t for sublist in time_stats[obj_name].values() for t in sublist]
                    writer.writerow([obj_name, "ALL_TYPES", len(all_t), f"{np.mean(all_t):.6f}", f"{np.sum(all_t):.6f}"])
                    writer.writerow([])
                
                # Total Mean
                writer.writerow(["MEAN_ALL_OBJECTS", "ALL_TYPES", len(all_inference_times), 
                               f"{np.mean(all_inference_times):.6f}", f"{np.sum(all_inference_times):.6f}"])

                print(f"\nMean inference time: {np.mean(all_inference_times):.5f}s = {len(all_inference_times)/np.sum(all_inference_times):.2f} samples/s")

            # 评估
            skip_eval = False
            for obj_name in objects:
                if not os.path.exists(f"{args.data_root}/{obj_name}/test/good/"):
                    skip_eval = True
                    break
            
            if not skip_eval:
                eval_finished_run(args.dataset, args.data_root,
                                  anomaly_maps_dir=f"{results_dir}/anomaly_maps/seed={seed}",
                                  output_dir=results_dir, seed=seed,
                                  pro_integration_limit=0.3,
                                  eval_clf=args.eval_clf, eval_segm=args.eval_segm)
                if save_examples:
                    create_sample_plots(results_dir, 
                                        anomaly_maps_dir=f"{results_dir}/anomaly_maps/seed={seed}",
                                        seed=seed, dataset=args.dataset, data_root=args.data_root)
                save_examples = False

    print("Finished!")
