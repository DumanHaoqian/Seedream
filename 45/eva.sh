echo "Evaluating Seedream 45..."
python evaluate_generation.py \
  --mode alignment \
  --metadata_csv /Users/haoqian3/Research/AnimationGEN/TestSet/test_data_7B_v17.csv\
  --generation_dir /Users/haoqian3/Research/AnimationGEN/Seedream/45/Samples/output \
  --gt_dir /Users/haoqian3/Research/AnimationGEN/TestSet/GT 
echo "Evaluation completed!"