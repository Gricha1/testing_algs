cd ..

export COMET_API_KEY="3OfuYHwcRgIwG7DzgzJ190igY"
python SafeDreamer/train.py --configs bsrp_lang_lag --method osrp_lag --jax.logical_gpus 0 \
                            --cost_limit 2.0 --action_type discrete  \
                            --task batchcraftext_drinkeasy --envs.amount 1 --env.batchcraftext.num_envs 8 \
                            --cost_weight 0.0001 --pessimistic False \
                            --envs.parallel none --envs.restart False \
                            #--task craftext_hungry --envs.amount 2
                            #--task batchcraftext_hungry --envs.amount 2 --env.batchcraftext.num_envs 2 --envs.parallel none --envs.restart False
                            #--envs.amount 2
                            #--envs.amount 1 --envs.parallel none --envs.restart False
                            #--envs.amount 3
                            #--envs.amount 1 --envs.parallel none --envs.restart False
                            #--envs.amount 3
                            #--envs.amount 1 --envs.parallel none --envs.restart False
                            #--batch_size 4 --data_loaders 1 --envs.amount 1