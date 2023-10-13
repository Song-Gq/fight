set -x

# CONFIG=$1
# CKPT=$2
# VIDEO=$3
# OUTDIR=${4:-"./examples/res"}

# python scripts/demo_inference.py \
#     --cfg ${CONFIG} \
#     --checkpoint ${CKPT} \
#     --video ${VIDEO} \
#     --outdir ${OUTDIR} \
#     --detector yolo  --save_img --save_video

# dir="/home/tj203/sgq/datasets/ucf-crime/Fighting/*"
# for f in $dir
# do 
#     if [ -f $f ]
#     then
#         python scripts/demo_inference.py \
#             --cfg "./configs/single_hand/resnet/256x192_res50_lr1e-3_2x-dcn-regression.yaml" \
#             --checkpoint "./pretrained_models/singlehand_fast50_dcn_regression_256x192.pth" \
#             --detector yolox-x \
#             --video $f \
#             --save_video --outdir "/home/tj203/sgq/datasets/output/alphapose/ucf-crime/Fighting" \
#             --pose_track --save_video \
#             --qsize 2
#     fi
# done

sub_dir="test"
dir="/home/song/dataset/$sub_dir/*"
out_dir="/home/song/dataset/output/alphapose_68/$sub_dir"
for f in $dir
do 
    if [ -f $f ]
    then
        python scripts/demo_inference.py \
            --cfg "./configs/halpe_68_noface/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml" \
            --checkpoint "./pretrained_models/noface_fast50_dcn_combined_256x192.pth" \
            --detector yolox-x \
            --video $f \
            --save_video --outdir $out_dir \
            --pose_track --save_video \
            --qsize 4
        mv $out_dir/alphapose-results.json $out_dir/$f.json
    fi
done

# python scripts/demo_inference.py \
#     --cfg "./configs/halpe_68_noface/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml" \
#     --checkpoint "pretrained_models/noface_fast50_dcn_combined_256x192.pth" \
#     --detector yolox-x \
#     --video "/home/song/dataset/private/vid/mixed.mp4" \
#     --pose_track --save_video \
#     --qsize 1


