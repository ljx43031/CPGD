本项目通过scene_configuration.py生成随机分布的建筑物、空中的检测无人机以及地面目标（地面目标在水平面做随机切换运动模型的CT运动）
train_prior_denoiser_v8.py设计了本项目的方法，并且在文件夹model_prior_denoiser_v8_ft给出了本方法最终训练好的模型。
baseline_dl.py主要包括了三个需要训练的dl模型（用于对比分析），包括LSTM，Transformer和csdi.
对比测试代码mc_compare_v8.py以及augment_mc.py，大家可以直接运行测试. 我已经把一版测试结果放在文件夹mc_results里.

This project generates randomly distributed buildings, aerial detection drones, and ground targets (CT motion of ground targets with random switching motion models in the horizontal plane) through scene_figuration-py
The method for this project was designed by train_prior_denoiser-v8.py, and the final trained model of this method is provided in the folder model_prior_denoiser-v8ft.
Baseline. py mainly includes three DL models that need to be trained (for comparative analysis), including LSTM, Transformer, and CSDI
You can directly run the test by comparing the test codes mc_compare-v8.py and auxing_cc.py I have placed a version of the test results in the folder mc_desults
