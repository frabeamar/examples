#!/bin/bash
#if folder does not exists, clone the sapiens repo
if [ ! -d sapiens ]; then
  git clone git@github.com:facebookresearch/sapiens.git
fi
cd sapiens
# download the models from huggingface
SAPIENS_CHECKPOINT_ROOT=checkpoints

MODE='torchscript' ## original. no optimizations (slow). full precision inference.
# MODE='bfloat16' ## A100 gpus. faster inference at bfloat16


#----------------------------set your input and ls  directories----------------------------------------------
INPUT='/home/fmarsicano/data/output_h264.mp4'
#if input_type is a video, and folder does not exists yet, convert it to a folder of images with ffmpeg
if [[ $INPUT == *.mp4 ]]; then
  if [ ! -d "${INPUT%.*}" ]; then
    ffmpeg -i "$INPUT" "${INPUT%.*}/frame_%04d.png"
  fi
  INPUT="${INPUT%.*}" # Update INPUT to the folder containing images
fi

OUTPUT='/home/fmarsicano/data/estimated_output_h264'

# --------------------------MODEL CARD---------------
# MODEL_NAME='sapiens_0.3b'; CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/sapiens_0.3b_coco_best_coco_AP_796_torchscript.pt2 
MODEL_NAME='sapiens_1b'; CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/sapiens_1b_coco_best_coco_AP_821_torchscript.pth
# MODEL_NAME='sapiens_0.6b'; CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/pose/checkpoints/sapiens_0.6b/sapiens_0.6b_coco_best_coco_AP_812_$MODE.pt2
# MODEL_NAME='sapiens_1b'; CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/pose/checkpoints/sapiens_1b/sapiens_1b_coco_best_coco_AP_821_$MODE.pt2
# MODEL_NAME='sapiens_2b'; CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/pose/checkpoints/sapiens_2b/sapiens_2b_coco_best_coco_AP_822_$MODE.pt2

OUTPUT=$OUTPUT/$MODEL_NAME
mkdir -p $OUTPUT


DETECTION_CONFIG_FILE='./pose/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person_no_nms.py'
DETECTION_CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/detector/checkpoints/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth

#---------------------------VISUALIZATION PARAMS--------------------------------------------------
LINE_THICKNESS=3 ## line thickness of the skeleton
RADIUS=6 ## keypoint radius
KPT_THRES=0.3 ## confidence threshold

##-------------------------------------inference-------------------------------------
RUN_FILE='lite/demo/vis_pose.py'

## number of inference jobs per gpu, total number of gpus and gpu ids
# JOBS_PER_GPU=1; TOTAL_GPUS=8; VALID_GPU_IDS=(0 1 2 3 4 5 6 7)
JOBS_PER_GPU=1; TOTAL_GPUS=1; VALID_GPU_IDS=(0)

BATCH_SIZE=2

# Find all images and sort them, then write to a temporary text file
IMAGE_LIST="${INPUT}/image_list.txt"
find "${INPUT}" -type f \( -iname \*.jpg -o -iname \*.png \) | sort > "${IMAGE_LIST}"

# Check if image list was created successfully
if [ ! -s "${IMAGE_LIST}" ]; then
  echo "No images found. Check your input directory and permissions."
  exit 1
fi

# Count images and calculate the number of images per text file
NUM_IMAGES=$(wc -l < "${IMAGE_LIST}")
if ((TOTAL_GPUS > NUM_IMAGES / BATCH_SIZE)); then
  TOTAL_JOBS=$(( (NUM_IMAGES + BATCH_SIZE - 1) / BATCH_SIZE))
  IMAGES_PER_FILE=$((BATCH_SIZE))
  EXTRA_IMAGES=$((NUM_IMAGES - ((TOTAL_JOBS - 1) * BATCH_SIZE)  ))
else
  TOTAL_JOBS=$((JOBS_PER_GPU * TOTAL_GPUS))
  IMAGES_PER_FILE=$((NUM_IMAGES / TOTAL_JOBS))
  EXTRA_IMAGES=$((NUM_IMAGES % TOTAL_JOBS))
fi

export TF_CPP_MIN_LOG_LEVEL=2
echo "Distributing ${NUM_IMAGES} image paths into ${TOTAL_JOBS} jobs."

# Divide image paths into text files for each job
for ((i=0; i<TOTAL_JOBS; i++)); do
  TEXT_FILE="${INPUT}/image_paths_$((i+1)).txt"
  if [ $i -eq $((TOTAL_JOBS - 1)) ]; then
    # For the last text file, write all remaining image paths
    tail -n +$((IMAGES_PER_FILE * i + 1)) "${IMAGE_LIST}" > "${TEXT_FILE}"
  else
    # Write the exact number of image paths per text file
    head -n $((IMAGES_PER_FILE * (i + 1))) "${IMAGE_LIST}" | tail -n ${IMAGES_PER_FILE} > "${TEXT_FILE}"
  fi
done
# run within docker; mount the entire local directory to the container
DOCKER_RUN="docker run --gpus all --rm -it  \
  --mount type=bind,src=.,dst=/workspace \
  --mount type=bind,src=$INPUT,dst=$INPUT \
  --mount type=bind,src=$OUTPUT,dst=$OUTPUT \
  --shm-size=15g \
  sapiens"

# Run the process on the GPUs, allowing multiple jobs per GPU
    # --det-config ${DETECTION_CONFIG_FILE} \
    # --det-checkpoint ${DETECTION_CHECKPOINT} \
for ((i=0; i<TOTAL_JOBS; i++)); do
  GPU_ID=$((i % TOTAL_GPUS))
  CUDA_VISIBLE_DEVICES=${VALID_GPU_IDS[GPU_ID]} $DOCKER_RUN python ${RUN_FILE} \
    ${CHECKPOINT} \
    --num_keypoints 17 \
    --batch-size ${BATCH_SIZE} \
    --input "${INPUT}/image_paths_$((i+1)).txt" \
    --output-root="${OUTPUT}" \
    --radius ${RADIUS} \
    --kpt-thr ${KPT_THRES} ## add & to process in background
  # Allow a short delay between starting each job to reduce system load spikes
  sleep 1
done

# Wait for all background processes to finish
wait

# Remove the image list and temporary text files
rm "${IMAGE_LIST}"
for ((i=0; i<TOTAL_JOBS; i++)); do
  rm "${INPUT}/image_paths_$((i+1)).txt"
done

# Go back to the original script's directory
cd -

echo "Processing complete."
echo "Results saved to $OUTPUT"
