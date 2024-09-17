cd logs/hrac

# download tensorboard from ml5:

## HRAC SAFE
folder_ml5_name=SafeGym_1__model_185
scp -P 9191 -r ggorbov@gater.frccsc.ru:/home/ggorbov/hrac_safety/testing_algs/logs/hrac/$folder_ml5_name ml5_$folder_ml5_name


## MBPPOL

#folder_ml5_name=test_point_136
#folder_ml5_name=Safexp-PointGoal2-v0_1__model_0
#scp -P 9191 -r ggorbov@gater.frccsc.ru:/home/ggorbov/article_mbppol/testing_algs/src/data/$folder_ml5_name .
