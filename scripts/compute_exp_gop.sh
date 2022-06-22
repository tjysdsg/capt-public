# python scripts/compute_exp_gop.py \
#   --data-dir data/pre_post_experiments/pre \
#   --text2phone data/pre_post_experiments/text2phone.txt \
#   --out-path data/pre_post_experiments/pre/data.csv || exit 1
#
# python scripts/compute_exp_gop.py \
#   --data-dir data/pre_post_experiments/post \
#   --text2phone data/pre_post_experiments/text2phone.txt \
#   --out-path data/pre_post_experiments/post/data.csv || exit 1

python scripts/gop_data_to_excel.py \
  --pre-path data/pre_post_experiments/pre/data.csv \
  --post-path data/pre_post_experiments/post/data.csv \
  --text2phone data/pre_post_experiments/text2phone.txt \
  --out-path data/pre_post_experiments/data.xlsx || exit 1
