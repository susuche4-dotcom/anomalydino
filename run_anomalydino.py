import argparse
import os
from argparse import ArgumentParser, Action
import yaml
from tqdm import trange

import csv
from collections import defaultdict

from src.utils import get_dataset_info
from src.detection import run_anomaly_detection
from src.post_eval import eval_finished_run
from src.visualize import create_sample_plots
from src.backbones import get_model


class IntListAction(Action):
    """
    Define a custom action to always return a list. 
    This allows --shots 1 to be treated as a list of one element [1]. 
    """
    def __call__(self, namespace, values):
        if not isinstance(values, list):
            values = [values]
        setattr(namespace, self.dest, values)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MVTec")
    parser.add_argument("--model_name", type=str, default="dinov2_vits14", help="Name of the backbone model. Choose from ['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14', 'vit_b_16'].")
    parser.add_argument("--data_root", type=str, default="./MVTec",
                        help="Path to the root directory of the dataset.")
    parser.add_argument("--preprocess", type=str, default="agnostic",
                        help="Preprocessing method. Choose from ['agnostic', 'informed', 'masking_only'].")
    parser.add_argument("--resolution", type=int, default=448)
    parser.add_argument("--knn_metric", type=str, default="L2_normalized")
    parser.add_argument("--k_neighbors", type=int, default=1)
    parser.add_argument("--faiss_on_cpu", default=False, action=argparse.BooleanOptionalAction, help="Use GPU for FAISS kNN search. (Conda install faiss-gpu recommended, does usually not work with pip install.)")
    parser.add_argument("--shots", nargs='+', type=int, default=[1], #action=IntListAction,
                        help="List of shots to evaluate. Full-shot scenario is -1.")
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--mask_ref_images", type=bool, default=False)
    parser.add_argument("--just_seed", type=int, default=None)
    parser.add_argument('--save_examples', default=True, action=argparse.BooleanOptionalAction, help="Save example plots.")
    parser.add_argument("--eval_clf", default=True, action=argparse.BooleanOptionalAction, help="Evaluate anomaly detection performance.")
    parser.add_argument("--eval_segm", default=True, action=argparse.BooleanOptionalAction, help="Evaluate anomaly segmentation performance.")
    parser.add_argument("--device", default='cuda:0')
    parser.add_argument("--warmup_iters", type=int, default=25, help="Number of warmup iterations, relevant when benchmarking inference time.")

    parser.add_argument("--tag", help="Optional tag for the saving directory.")

    args = parser.parse_args()
    return args


if __name__=="__main__":

    args = parse_args()
    
    print(f"Requested to run {len(args.shots)} (different) shot(s):", args.shots)
    print(f"Requested to repeat the experiments {args.num_seeds} time(s).")

    objects, object_anomalies, masking_default, rotation_default = get_dataset_info(args.dataset, args.preprocess, data_path=args.data_root)

    # set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device[-1])
    model = get_model(args.model_name, 'cuda', smaller_edge_size=args.resolution)

    if not args.model_name.startswith("dinov2"):
        masking_default = {o: False for o in objects}
        print("Caution: Only DINOv2 supports 0-shot masking (for now)!")

    if args.just_seed != None:
        seeds = [args.just_seed]
    else:
        seeds = range(args.num_seeds)
    
    for shot in list(args.shots):
        save_examples = args.save_examples

        # 使用绝对路径保存结果到 E 盘项目目录
        results_dir = f"E:/part/AnomalyDINO/results_MVTec/{args.model_name}_{args.resolution}/{shot}-shot_preprocess={args.preprocess}"
        
        if args.tag != None:
            results_dir += "_" + args.tag
        plots_dir = results_dir
        os.makedirs(f"{results_dir}", exist_ok=True)
        
        # save preprocessing setups (masking and rotation) to file
        with open(f"{results_dir}/preprocess.yaml", "w") as f:
            yaml.dump({"masking": masking_default, "rotation": rotation_default}, f)

        # save arguments to file
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
                    writer.writerow(["Object", "Anomaly_Type", "Sample", "Anomaly_Score", "MemoryBank_Time", "Inference_Time"])

                    for object_name in objects:
                        
                        if save_examples:
                            os.makedirs(f"{plots_dir}/{object_name}", exist_ok=True)
                            os.makedirs(f"{plots_dir}/{object_name}/examples", exist_ok=True)

                        # CUDA warmup
                        train_dir = f"{args.data_root}/{object_name}/train"
                        if os.path.exists(f"{train_dir}/good"):
                            train_dir = f"{train_dir}/good"
                        for _ in trange(args.warmup_iters, desc="CUDA warmup", leave=False):
                            first_image = os.listdir(train_dir)[0]
                            img_tensor, grid_size = model.prepare_image(f"{train_dir}/{first_image}")
                            features = model.extract_features(img_tensor)
                                         
                        anomaly_scores, time_memorybank, time_inference, timing_stats = run_anomaly_detection(
                                                                                model,
                                                                                object_name,
                                                                                data_root = args.data_root,
                                                                                n_ref_samples = shot,
                                                                                object_anomalies = object_anomalies,
                                                                                plots_dir = plots_dir,
                                                                                save_examples = save_examples,
                                                                                knn_metric = args.knn_metric,
                                                                                knn_neighbors = args.k_neighbors,
                                                                                faiss_on_cpu = args.faiss_on_cpu,
                                                                                masking = masking_default[object_name],
                                                                                mask_ref_images = args.mask_ref_images,
                                                                                rotation = rotation_default[object_name],
                                                                                seed = seed,
                                                                                save_patch_dists = args.eval_clf, # save patch distances for detection evaluation
                                                                                save_tiffs = args.eval_segm)      # save anomaly maps as tiffs for segmentation evaluation
                        
                        # 保存详细步骤时间到 CSV
                        detailed_time_file = results_dir + f"/time_statistics_detailed_{object_name}_seed={seed}.csv"
                        with open(detailed_time_file, 'w', newline='') as dt_file:
                            dt_writer = csv.writer(dt_file)
                            dt_writer.writerow([
                                "Object", "Anomaly_Type", "Sample",
                                "Load_Preprocess(s)", "Feature_Extract(s)",
                                "Mask_Compute(s)", "KNN_Search(s)",
                                "Post_Process(s)", "Total_Inference(s)"
                            ])
                            
                            sample_idx = 0
                            for type_anomaly in object_anomalies[object_name]:
                                data_dir = f"{args.data_root}/{object_name}/test/{type_anomaly}"
                                if not os.path.exists(data_dir):
                                    continue
                                
                                for img_test_nr in sorted(os.listdir(data_dir)):
                                    if sample_idx >= len(timing_stats['load_preprocess']):
                                        break
                                    
                                    dt_writer.writerow([
                                        object_name, type_anomaly, img_test_nr,
                                        f"{timing_stats['load_preprocess'][sample_idx]:.6f}",
                                        f"{timing_stats['feature_extract'][sample_idx]:.6f}",
                                        f"{timing_stats['mask_compute'][sample_idx]:.6f}",
                                        f"{timing_stats['knn_search'][sample_idx]:.6f}",
                                        f"{timing_stats['post_process'][sample_idx]:.6f}",
                                        f"{time_inference.get(f'{type_anomaly}/{img_test_nr}', 0):.6f}"
                                    ])
                                    sample_idx += 1
                        
                        # write anomaly scores and inference times to file
                        for counter, sample in enumerate(anomaly_scores.keys()):
                            anomaly_score = anomaly_scores[sample]
                            inference_time = time_inference[sample]
                            
                            # 从 sample 路径中提取 anomaly_type (格式："anomaly_type/image_name")
                            anomaly_type = sample.split('/')[0] if '/' in sample else 'unknown'
                            
                            # 写入 CSV
                            writer.writerow([object_name, anomaly_type, sample, f"{anomaly_score:.5f}", f"{time_memorybank:.5f}", f"{inference_time:.5f}"])
                            
                            # 统计时间
                            time_stats[object_name][anomaly_type].append(inference_time)
                            all_inference_times.append(inference_time)                        

                # 统计并保存时间信息到 CSV
                with open(time_stats_file, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["Object", "Anomaly_Type", "Num_Samples", "Avg_Time_Per_Sample(s)", "Total_Time(s)"])
                    
                    all_object_avg_times = []  # 存储每个物体的平均时间
                    
                    for object_name in time_stats.keys():
                        object_total_samples = 0
                        object_total_time = 0
                        
                        for anomaly_type in sorted(time_stats[object_name].keys()):
                            times = time_stats[object_name][anomaly_type]
                            num_samples = len(times)
                            avg_time = sum(times) / num_samples if num_samples > 0 else 0
                            total_time = sum(times)
                            
                            writer.writerow([object_name, anomaly_type, num_samples, f"{avg_time:.6f}", f"{total_time:.6f}"])
                            
                            object_total_samples += num_samples
                            object_total_time += total_time
                        
                        # 计算该物体所有类型的平均时间
                        object_avg_time = object_total_time / object_total_samples if object_total_samples > 0 else 0
                        all_object_avg_times.append(object_avg_time)
                        
                        # 写入该物体的汇总行
                        writer.writerow([object_name, "ALL_TYPES", object_total_samples, f"{object_avg_time:.6f}", f"{object_total_time:.6f}"])
                        writer.writerow([])  # 空行分隔
                
                # 计算所有类别的总平均时间
                if all_inference_times:
                    total_avg_time = sum(all_inference_times) / len(all_inference_times)
                    total_samples = len(all_inference_times)
                    total_time = sum(all_inference_times)
                    
                    # 写入总平均行
                    with open(time_stats_file, 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(["MEAN_ALL_OBJECTS", "ALL_TYPES", total_samples, f"{total_avg_time:.6f}", f"{total_time:.6f}"])
                    
                    print(f"Finished AD for {len(objects)} objects (seed {seed}), mean inference time: {total_avg_time:.5f} s/sample = {total_samples/(total_time+1e-10):.2f} samples/s")

                # check wheter 'good' folder exists for testing
                for object_name in objects:
                    good_folder = f"{args.data_root}/{object_name}/test/good/"
                    if not os.path.exists(good_folder):
                        print(f"Warning: 'good' folder not found for {object_name}! No evaluation will be performed for seed {seed}.")
                        print("Finished AD without evaluation, inference results saved to", results_dir)
                        break
                else:
                    # evaluate all finished runs and create sample anomaly maps for inspection
                    print(f"=========== Evaluate seed = {seed} ===========")
                    eval_finished_run(args.dataset,
                                    args.data_root,
                                    anomaly_maps_dir = results_dir + f"/anomaly_maps/seed={seed}",
                                    output_dir = results_dir,
                                    seed = seed,
                                    pro_integration_limit = 0.3,
                                    eval_clf = args.eval_clf,
                                    eval_segm = args.eval_segm,
                                    downsample_factor = 8)  # 下采样8倍以减少内存使用
                    
                    create_sample_plots(results_dir, 
                                        anomaly_maps_dir = results_dir + f"/anomaly_maps/seed={seed}", 
                                        seed = seed,
                                        dataset = args.dataset, 
                                        data_root = args.data_root)
                
                    # deactivate creation of examples for the next seeds...
                    save_examples = False 

    print("Finished and evaluated all runs!")