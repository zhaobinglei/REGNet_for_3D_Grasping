#  REGNet_for_3D_Grasping
RGENet is a REgion-based Grasp Network for End-to-end Grasp Detection in Point Clouds. It aims at generating the optimal grasp of novel objects from partial noisy observations.
<div align=center>
<img src=markdown/1_1.png width=600>
</div>

---
## Install (Using a virtual environment is recommended)
1. Clone the repository
```
cd ${your_project_path}
git clone https://github.com/zhaobinglei/REGNet_for_3D_Grasping.git
```

2. Install the requirements in ```requirements.txt```
3. Install the utils of pointnet++
```
cd REGNet_for_3D_Grasping/multi_model/utils/pn2_utils
python setup.py install
cd functions
python setup.py install
```
4. Modify the path of models and datasets in ```train.py``` and ```test.py```
```
'--load-score-path'  : The path of pretrained ScoreNet model 
'--load-region-path': The path of pretrained GraspRegionNet and RefineNet model 
'--data-path'       : The dataset path
'--model-path'      : to saved model path
'--log-path'        : to saved log path
'--folder-name'     : the folder path of the point cloud file to be tested
'--file-name'       : the name the point cloud file to be tested. If use the default path, it will test all files in the folder-name.
```
5. Modify the parameters of gripper and enviroment configuration in ```train.py``` and ```test.py```
```
width, height, depth: the parameters of the two-finger grippers
table_height        : the height of the table
```
---
## Training
The architecture of REGNet is shown in follows.
<div align=center>
<img src=markdown/4_2.png width=800>
</div>
 
1. Train (Recommand training the three stages based on the pretrained models.
If using the pretrained models, please modify the path of models and datasets in ```train.py```.)
```
cd REGNet_for_3D_Grasping/
python train.py --gpu 0 --gpu-num 3 --gpus 0,1,2 --batch-size 15 --mode train --tag regnet
```

2. Pretrain (You can train the SN and GRN separately or at the same time.)
```
cd REGNet_for_3D_Grasping/
python train.py --gpu 0 --gpu-num 3 --gpus 0,1,2 --batch-size 15 --mode pretrain_score --tag regnet_score
python train.py --gpu 0 --gpu-num 3 --gpus 0,1,2 --batch-size 15 --mode pretrain_region --tag regnet_region
```

3. Launch tensorboard for monitoring
```
cd REGNet_for_3D_Grasping/assets/log
tensorboard --logdir . --port 8080
```

---
## Test and Visualization
1. Test the file in ```test_file/real_data``` or ```test_file/virtual_data```. The corresponding predict results are saved in ```test_file/real_data_predict``` or ```test_file/virtual_data_predict```
```
cd REGNet_for_3D_Grasping/
python test.py --gpu 7 --gpu-num 1
```

2. visulation 
```
cd REGNet_for_3D_Grasping/vis
python vis_grasp.py 
```
---
## Results
There are some results predicted by REGNet.
The first situation is in the validation dataset, which has the same distribution as the training dataset (table_height=0.75). Some objects do not have the corresponding grasps, since the gripper's width is smaller than the object's size. The red grasp has the highest predicted score. 
<div align=center>
<img src=markdown/2.png width=800>
</div>
The second situation is also generated in simulation enviroment. But it uses the objects' models that don't in the training dataset (table_height=0.5).
<div align=center>
<img src=markdown/1.png width=800>
</div>
The next two pictures are observed by Kinect in the real world (table_height=0.7). The objects also don't in the training dataset.
<div align=center>
<img src=markdown/3.png width=800>
</div>
<div align=center>
<img src=markdown/4.png width=800>
</div>