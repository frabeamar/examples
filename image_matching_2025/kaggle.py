from pathlib import Path
import os
from tqdm import tqdm
from time import time, sleep
import gc
import numpy as np
import h5py
import dataclasses
import pandas as pd
from copy import deepcopy

import torch
import torch.nn.functional as F
import kornia as K
import kornia.feature as KF

from transformers import AutoImageProcessor, AutoModel, SuperPointForKeypointDetection
from gluefactory.geometry.epipolar import relative_pose_error

# Utilities: importing data into colmap and competition metric
import pycolmap

# Don't forget to select an accelerator on the sidebar to the right.
device = K.utils.get_cuda_device_if_available(0)


def array_to_str(array):
    return ";".join([f"{x:.09f}" for x in array.flatten()])


def none_to_str(n):
    return ";".join(["nan"] * n)


def val_to_str(v, n):
    if v is None:
        return none_to_str(n)
    return array_to_str(v)


@dataclasses.dataclass
class Models:
    preprocessor: AutoImageProcessor
    model: torch.nn.Module


MODELS = {
    "superpoint": Models(
        AutoImageProcessor.from_pretrained("magic-leap-community/superpoint"),
        SuperPointForKeypointDetection.from_pretrained(
            "magic-leap-community/superpoint"
        ),
    )
}


def load_torch_image(fname, device=torch.device("cpu")):
    img = K.io.load_image(fname, K.io.ImageLoadType.RGB32, device=device)[None, ...]
    return img


# Use efficientnet global descriptor to get matching shortlists.
def get_global_desc(fnames, device=torch.device("cpu")):
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base")
    model = model.eval()
    model = model.to(device)
    global_descs_dinov2 = []
    for i, img_fname_full in tqdm(enumerate(fnames), total=len(fnames)):
        key = os.path.splitext(os.path.basename(img_fname_full))[0]
        timg = load_torch_image(img_fname_full)
        with torch.inference_mode():
            inputs = processor(images=timg, return_tensors="pt", do_rescale=False).to(
                device
            )
            outputs = model(**inputs)
            dino_mac = F.normalize(
                outputs.last_hidden_state[:, 1:].max(dim=1)[0], dim=1, p=2
            )
        global_descs_dinov2.append(dino_mac.detach().cpu())
    global_descs_dinov2 = torch.cat(global_descs_dinov2, dim=0)
    return global_descs_dinov2


def get_img_pairs_exhaustive(img_fnames):
    index_pairs = []
    for i in range(len(img_fnames)):
        for j in range(i + 1, len(img_fnames)):
            index_pairs.append((i, j))
    return index_pairs


def get_image_pairs_shortlist(
    fnames,
    sim_th=0.6,  # should be strict
    min_pairs=20,
    exhaustive_if_less=20,
    device=torch.device("cpu"),
):
    num_imgs = len(fnames)
    if num_imgs <= exhaustive_if_less:
        return get_img_pairs_exhaustive(fnames)
    descs = get_global_desc(fnames, device=device)
    dm = torch.cdist(descs, descs, p=2).detach().cpu().numpy()
    # removing half
    mask = dm <= sim_th
    total = 0
    matching_list = []
    ar = np.arange(num_imgs)
    already_there_set = []
    for st_idx in range(num_imgs - 1):
        mask_idx = mask[st_idx]
        to_match = ar[mask_idx]
        if len(to_match) < min_pairs:
            to_match = np.argsort(dm[st_idx])[:min_pairs]
        for idx in to_match:
            if st_idx == idx:
                continue
            if dm[st_idx, idx] < 1000:
                matching_list.append(tuple(sorted((st_idx, idx.item()))))
                total += 1
    matching_list = sorted(list(set(matching_list)))
    return matching_list


def detect_features(
    img_fnames,
    feature_dir: str,
    device=torch.device("cpu"),
):
    dtype = torch.float32  # ALIKED has issues with float16d
    # ALIKED(max_num_keypoints=num_features, detection_threshold=0.01, resize=resize_to)
    model = MODELS["superpoint"].model
    extractor = model.eval().to(device, dtype)
    if not os.path.isdir(feature_dir):
        os.makedirs(feature_dir)
    with (
        h5py.File(f"{feature_dir}/keypoints.h5", mode="w") as f_kp,
        h5py.File(f"{feature_dir}/descriptors.h5", mode="w") as f_desc,
    ):
        for img_path in tqdm(img_fnames):
            img_fname = img_path.split("/")[-1]
            key = img_fname
            with torch.inference_mode():
                image0 = load_torch_image(img_path, device=device).to(dtype)
                feats0 = extractor(
                    image0
                )  # auto-resize the image, disable with resize=None
                kpts = feats0["keypoints"].reshape(-1, 2).detach().cpu().numpy()
                descs = (
                    feats0["descriptors"].reshape(len(kpts), -1).detach().cpu().numpy()
                )
                f_kp[key] = kpts
                f_desc[key] = descs
    return


def match_with_lightglue(
    img_fnames,
    index_pairs,
    feature_dir=".featureout",
    device=torch.device("cpu"),
    min_matches=15,
    verbose=True,
):
    lg_matcher = (
        KF.LightGlueMatcher(
            "superpoint",
            {
                "width_confidence": -1,
                "depth_confidence": -1,
                "mp": True if "cuda" in str(device) else False,
            },
        )
        .eval()
        .to(device)
    )
    with (
        h5py.File(f"{feature_dir}/keypoints.h5", mode="r") as f_kp,
        h5py.File(f"{feature_dir}/descriptors.h5", mode="r") as f_desc,
        h5py.File(f"{feature_dir}/matches.h5", mode="w") as f_match,
    ):
        for pair_idx in tqdm(index_pairs):
            idx1, idx2 = pair_idx
            fname1, fname2 = img_fnames[idx1], img_fnames[idx2]
            key1, key2 = fname1.split("/")[-1], fname2.split("/")[-1]
            kp1 = torch.from_numpy(f_kp[key1][...]).to(device)
            kp2 = torch.from_numpy(f_kp[key2][...]).to(device)
            desc1 = torch.from_numpy(f_desc[key1][...]).to(device)
            desc2 = torch.from_numpy(f_desc[key2][...]).to(device)
            with torch.inference_mode():
                dists, idxs = lg_matcher(
                    desc1,
                    desc2,
                    KF.laf_from_center_scale_ori(kp1[None]),
                    KF.laf_from_center_scale_ori(kp2[None]),
                )
            if len(idxs) == 0:
                continue
            n_matches = len(idxs)
            if verbose:
                print(f"{key1}-{key2}: {n_matches} matches")
            group = f_match.require_group(key1)
            if n_matches >= min_matches:
                group.create_dataset(
                    key2, data=idxs.detach().cpu().numpy().reshape(-1, 2)
                )
    return


def create_colmap_database(output_path: Path, image_dir: Path):
    output_path.mkdir()
    mvs_path = output_path / "mvs"
    database_path = output_path / "database.db"
    pycolmap.add_matches_to_database(database_path, image_dir, {})
    pycolmap.extract_features(database_path, image_dir)
    pycolmap.match_exhaustive(database_path)
    maps = pycolmap.incremental_mapping(database_path, image_dir, output_path)
    maps[0].write(output_path)
    # dense reconstruction
    pycolmap.undistort_images(mvs_path, output_path, image_dir)
    pycolmap.patch_match_stereo(mvs_path)  # requires compilation with CUDA
    pycolmap.stereo_fusion(mvs_path / "dense.ply", mvs_path)


def import_into_colmap(img_dir: Path, database_path: Path, output_path: Path):
    mvs_path = output_path / "mvs"

    database_str = str(database_path)
    output_str = str(output_path)
    image_dir_str = str(img_dir)
    mvs_path_str = str(mvs_path)

    # By default colmap does not generate a reconstruction if less than 10 images are registered.
    # Lower it to 3.
    mapper_options = pycolmap.IncrementalPipelineOptions()
    mapper_options.min_model_size = 3
    mapper_options.max_num_models = 25
    pycolmap.extract_features(database_str, image_dir_str)
    pycolmap.match_exhaustive(database_str)
    maps = pycolmap.incremental_mapping(
        database_path=database_str,
        image_path=image_dir_str,
        output_path=output_str,
        options=mapper_options,
    )
    maps[0].write(output_str)
    pycolmap.Reconstruction(output_str)
    # dense reconstruction
    pycolmap.undistort_images(mvs_path_str, output_str, image_dir_str)
    # pycolmap.patch_match_stereo(mvs_path)  # requires compilation with CUDA
    # Purpose: Computes dense depth maps and normals for each image using multi-view stereo (MVS).
    # pycolmap.stereo_fusion(mvs_path_str / "dense.ply", mvs_path)
    # Purpose: Converts the dense depth maps into a 3D point cloud.

    return maps


# Collect info from the dataset


@dataclasses.dataclass
class Prediction:
    image_id: (
        str | None
    )  # A unique identifier for the row -- unused otherwise. Used only on the hidden test set.
    dataset: str
    filename: str
    cluster_index: int | None = None
    rotation: np.ndarray | None = None
    translation: np.ndarray | None = None


# Set is_train=True to run the notebook on the training data.
# Set is_train=False if submitting an entry to the competition (test data is hidden, and different from what you see on the "test" folder).
def main(is_train: bool):
    DATA_DIR = Path.home() / "data/image_matching_2025"
    workdir = Path(__file__).parent

    if is_train:
        sample_submission_csv = os.path.join(DATA_DIR, "train_labels.csv")
    else:
        sample_submission_csv = os.path.join(DATA_DIR, "sample_submission.csv")
        if not Path(sample_submission_csv).exists():
            pd.DataFrame().to_csv(sample_submission_csv)

    samples = {}
    competition_data = pd.read_csv(sample_submission_csv)
    for _, row in competition_data.iterrows():
        # Note: For the test data, the "scene" column has no meaning, and the rotation_matrix and translation_vector columns are random.
        if row.dataset not in samples:
            samples[row.dataset] = []
        samples[row.dataset].append(
            Prediction(
                image_id=None if is_train else row.image_id,
                dataset=row.dataset,
                filename=row.image,
            )
        )
    print("Datasets and number of images from train_labels:")
    for dataset in samples:
        print(f'Dataset "{dataset}" -> num_images={len(samples[dataset])}')
    gc.collect()

    max_images = None  # For debugging only. Set to None to disable.
    datasets_to_process = None  # Not the best convention, but None means all datasets.

    if is_train:
        # max_images = 5

        # Note: When running on the training dataset, the notebook will hit the time limit and die. Use this filter to run on a few specific datasets.
        datasets_to_process = [
            # New data.
            "amy_gardens",
            "ETs",
            "fbk_vineyard",
            "stairs",
            # Data from IMC 2023 and 2024.
            # 'imc2024_dioscuri_baalshamin',
            # 'imc2023_theather_imc2024_church',
            # 'imc2023_heritage',
            # 'imc2023_haiper',
            # 'imc2024_lizard_pond',
            # Crowdsourced PhotoTourism data.
            # 'pt_stpeters_stpauls',
            # 'pt_brandenburg_british_buckingham',
            # 'pt_piazzasanmarco_grandplace',
            # 'pt_sacrecoeur_trevi_tajmahal',
        ]

    timings = {
        "shortlisting": [],
        "feature_detection": [],
        "feature_matching": [],
        "RANSAC": [],
        "Reconstruction": [],
    }
    mapping_result_strs = []

    print(f"Extracting on device {device}")
    for dataset, predictions in samples.items():
        if datasets_to_process and dataset not in datasets_to_process:
            print(f'Skipping "{dataset}"')
            continue

        images_dir = os.path.join(DATA_DIR, "train" if is_train else "test", dataset)
        images = [os.path.join(images_dir, p.filename) for p in predictions]
        if max_images is not None:
            images = images[:max_images]

        print(f'\nProcessing dataset "{dataset}": {len(images)} images')

        filename_to_index = {p.filename: idx for idx, p in enumerate(predictions)}

        feature_dir = os.path.join(workdir, "featureout", dataset)
        os.makedirs(feature_dir, exist_ok=True)

        # Wrap algos in try-except blocks so we can populate a submission even if one scene crashes.
        try:
            t = time()
            index_pairs = get_image_pairs_shortlist(
                images,
                sim_th=0.3,  # should be strict
                min_pairs=20,  # we select at least min_pairs PER IMAGE with biggest similarity
                exhaustive_if_less=20,
                device=device,
            )
            timings["shortlisting"].append(time() - t)
            print(
                f"Shortlisting. Number of pairs to match: {len(index_pairs)}. Done in {time() - t:.4f} sec"
            )
            gc.collect()

            t = time()

            detect_features(images, feature_dir=feature_dir, device=device)
            gc.collect()
            timings["feature_detection"].append(time() - t)
            print(f"Features detected in {time() - t:.4f} sec")

            t = time()
            match_with_lightglue(
                images,
                index_pairs,
                feature_dir=feature_dir,
                device=device,
                verbose=False,
            )
            timings["feature_matching"].append(time() - t)
            print(f"Features matched in {time() - t:.4f} sec")

            database_path = os.path.join(feature_dir, "colmap.db")
            if os.path.isfile(database_path):
                os.remove(database_path)
            gc.collect()
            sleep(1)
            t = time()
            output_path = Path(f"{feature_dir}/colmap_rec_aliked")
            maps = import_into_colmap(
                images_dir, database_path=Path(database_path), output_path=output_path
            )

            timings["RANSAC"].append(time() - t)
            print(f"Ran RANSAC in {time() - t:.4f} sec")

            os.makedirs(output_path, exist_ok=True)
            t = time()

            sleep(1)
            registered = 0
            for map_index, cur_map in maps.items():
                for index, image in cur_map.images.items():
                    prediction_index = filename_to_index[image.name]
                    predictions[prediction_index].cluster_index = map_index
                    predictions[prediction_index].rotation = deepcopy(
                        image.cam_from_world().rotation.matrix()
                    )
                    predictions[prediction_index].translation = deepcopy(
                        image.cam_from_world().translation
                    )
                    registered += 1
            mapping_result_str = f'Dataset "{dataset}" -> Registered {registered} / {len(images)} images with {len(maps)} clusters'
            mapping_result_strs.append(mapping_result_str)
            print(mapping_result_str)
            gc.collect()
        except Exception as e:
            print(e)
            # raise e
            mapping_result_str = f'Dataset "{dataset}" -> Failed!'
            mapping_result_strs.append(mapping_result_str)
            print(mapping_result_str)

    print("\nResults")
    for s in mapping_result_strs:
        print(s)

    print("\nTimings")
    for k, v in timings.items():
        print(f"{k} -> total={sum(v):.02f} sec.")
    # Create a submission file.

    submission_file = Path("submission_file.csv")
    samples_file = pd.concat([pd.DataFrame(s) for s in samples.values()])

    samples_file.cluster_index = samples_file.cluster_index.replace({None: "outliers"})
    samples_file.rotation = samples_file.rotation.apply(lambda x: val_to_str(x, 9))
    samples_file.translation = samples_file.translation.apply(
        lambda x: val_to_str(x, 3)
    )
    if is_train:
        # f.write('dataset,scene,image,rotation_matrix,translation_vector\n')
        samples_file.rename(
            columns={
                "rotation": "rotation_matrix",
                "translation": "translation_vector",
                "filename": "image",
            },
            inplace=True,
        )
    else:
        #  f.write('image_id,dataset,scene,image,rotation_matrix,translation_vector\n')
        samples_file.rename(
            columns={
                "rotation": "rotation_matrix",
                "translation": "translation_vector",
                "filename": "image",
            },
            inplace=True,
        )
    samples_file.to_csv(submission_file, index=False)


def eval():
    submission_file = pd.read_csv(Path("submission_file.csv"))
    train_labels = pd.read_csv(DATA_DIR / "train_labels.csv")
    for col in ["rotation_matrix", "translation_vector"]:
        submission_file[col] = submission_file[col].apply(
            lambda x: np.fromstring(x, sep=";")
        )
        train_labels[col] = train_labels[col].apply(lambda x: np.fromstring(x, sep=";"))

    evals = []
    for idx in submission_file.index:
        row_sub = submission_file[idx]
        row_gt = train_labels[idx].iloc[0]
        Rgt = np.concatenate(
            [
                np.concatenate(
                    [
                        row_gt.rotation_matrix.reshape(3, 3),
                        row_gt.translation_vector.reshape(3, 1),
                    ],
                    axis=1,
                ),
                np.ones((1, 4)),
            ]
        )

        t_err, r_err = relative_pose_error(
            Rgt,
            row_sub.rotation_matrix.reshape(3, 3),
            row_sub.translation_vector.reshape(3),
        )
        evals.append({"rot_err":r_err, "translate_err": t_err, "dataset": row_sub.dataset})

    print(evals.groupby("dataset").mean())

# Compute results if running on the training set.
# Don't do this when submitting a notebook for scoring. All you have to do is save your submission to /kaggle/working/submission.csv.

if __name__ == "__main__":
    t = time()
    DATA_DIR = Path.home() / "data/image_matching_2025"
    main(is_train=True)
    eval()
    print(f"Computed metric in: {time() - t:.02f} sec.")
