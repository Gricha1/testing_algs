cd ..

python SafeDreamer/train.py --configs bsrp_lang_lag --method osrp_lag --task craftext_drink --jax.logical_gpus 0 \
                            --envs.amount 4 --cost_limit 5.0
                            #--batch_size 4 --data_loaders 1 --envs.amount 1