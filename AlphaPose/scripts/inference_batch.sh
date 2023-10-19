set -x

sub_dir="fight-sur/fight"
dir="/home/tj203/sgq/datasets/$sub_dir/*"
out_dir="/home/tj203/sgq/datasets/output/alphapose_26_thre/$sub_dir"
for f in $dir
do 
    if [ -f $f ]
    then
        python scripts/demo_inference.py \
            --cfg "./configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml" \
            --checkpoint "./pretrained_models/halpe26_fast_res50_256x192.pth" \
            --detector yolox-x \
            --video $f --outdir $out_dir \
            --pose_track --eval --flip \
            --detbatch 6 --posebatch 4096 \
            --vis_fast --save_video --showbox
        filename=$(basename $f .mp4)
        mv $out_dir/alphapose-results.json $out_dir/$filename.json
    fi
done

# python scripts/demo_inference.py \
#     --cfg "./configs/halpe_68_noface/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml" \
#     --checkpoint "./pretrained_models/noface_fast50_dcn_combined_256x192.pth" \
#     --detector yolox-x \
#     --video $f \
#     --save_video --outdir $out_dir \
#     --pose_track \
#     --vis_fast \
#     --qsize 10  --posebatch 4096 --eval
