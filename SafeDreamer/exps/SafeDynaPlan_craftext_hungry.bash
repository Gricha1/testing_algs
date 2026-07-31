cd ..

python SafeDreamer/train.py --configs osrp_lang_lag --method osrp_lang_lag --jax.logical_gpus 0 \
                            --cost_limit 2.0 --action_type discrete  \
                            --task batchcraftext_hungry --envs.amount 1 --env.batchcraftext.num_envs 64 \
                            --envs.parallel none --envs.restart False
                            #--batch_size 4 --data_loaders 1 --envs.amount 1