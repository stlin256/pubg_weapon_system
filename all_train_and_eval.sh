#!/bin/bash

# 记录开始时间
start_time=$(date +%s)

# --- 步骤 1: 询问是否进行特征提取 ---
read -p "Do you need to run feature extraction? (Only needed if data has changed) [y/N]: " response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]
then
    echo "--- Running feature extraction ---"
    python -m src.sound_recognition.feature_extractor
else
    echo "--- Skipping feature extraction ---"
fi

# --- 步骤 2: 批量训练所有模型 ---

# --- 为目标 "weapon" 训练所有模型 ---
echo "--- Training all models for target: weapon ---"
python -m src.sound_recognition.train --model RandomForest --target weapon
python -m src.sound_recognition.train --model KNN --target weapon
python -m src.sound_recognition.train --model SVM --target weapon
python -m src.sound_recognition.train --model XGBoost --target weapon --use_cuda
python -m src.sound_recognition.train --model LightGBM --target weapon --use_cuda
python -m src.sound_recognition.train --model SVM_GridSearch --target weapon
python -m src.sound_recognition.train_ast --target weapon --epochs 10
python -m src.sound_recognition.train_passt --target weapon --epochs 10

# --- 为目标 "distance" 训练所有模型 ---
echo "--- Training all models for target: distance ---"
python -m src.sound_recognition.train --model RandomForest --target distance
python -m src.sound_recognition.train --model KNN --target distance
python -m src.sound_recognition.train --model SVM --target distance
python -m src.sound_recognition.train --model XGBoost --target distance --use_cuda
python -m src.sound_recognition.train --model LightGBM --target distance --use_cuda
python -m src.sound_recognition.train --model SVM_GridSearch --target distance
python -m src.sound_recognition.train_ast --target distance --epochs 10
python -m src.sound_recognition.train_passt --target distance --epochs 10

# --- 为目标 "direction" 训练所有模型 ---
echo "--- Training all models for target: direction ---"
python -m src.sound_recognition.train --model RandomForest --target direction
python -m src.sound_recognition.train --model KNN --target direction
python -m src.sound_recognition.train --model SVM --target direction
python -m src.sound_recognition.train --model XGBoost --target direction --use_cuda
python -m src.sound_recognition.train --model LightGBM --target direction --use_cuda
python -m src.sound_recognition.train --model SVM_GridSearch --target direction
python -m src.sound_recognition.train_ast --target direction --epochs 10
python -m src.sound_recognition.train_passt --target direction --epochs 10

# --- 步骤 3: 最终综合评估 ---
# 在所有模型都训练完毕后运行
echo "--- Running final evaluation on all trained models ---"
python -m src.sound_recognition.evaluate

# 记录结束时间并计算总时长
end_time=$(date +%s)
runtime=$((end_time - start_time))
minutes=$((runtime / 60))
seconds=$((runtime % 60))

echo "--- All tasks completed! ---"
echo "Total runtime: ${minutes} minutes and ${seconds} seconds."