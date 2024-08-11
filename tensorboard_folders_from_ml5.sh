cd logs/hrac

# download weights to ml5:
#scp -P 9191 -r 321 ggorbov@gater.frccsc.ru:/home/ggorbov/hrac_safety/testing_algs/models

folder_ml5_name=SafeAntMaze_2_safety_mb=0.8_safety_mf=0.2_safety=3000_grad=600_model_77
# download weights from ml5:
scp -P 9191 -r ggorbov@gater.frccsc.ru:/home/ggorbov/hrac_safety/testing_algs/logs/hrac/$folder_ml5_name .