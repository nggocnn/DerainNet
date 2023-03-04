for i in $(seq 1 1 1)
do
    echo "dealing with ${i}:";
    python ./validation.py \
    --load_name "./models/model.pth" \
    --save_name "./results/results_test" \
    --baseroot "./datasets/testing" ;
done