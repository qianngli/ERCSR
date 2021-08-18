ERCSR
======
**This is an implementation of  Exploring the Relationship between 2D/3D Convolution for Hyperspectral Image Super-Resolution.**

Dataset
------
**Three public datasets, i.e., [CAVE](https://www1.cs.columbia.edu/CAVE/databases/multispectral/ "CAVE"), [Harvard](http://vision.seas.harvard.edu/hyperspec/explore.html "Harvard"), [Pavia Centre](http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes#Pavia_Centre_scene "Pavia Centre"), are employed to verify the effectiveness of the  proposed ERCSR. Since there are too few images in these datasets for deep learning algorithm, we augment the training data. With respect to the specific details, please see the implementation details section.**

**Moreover, we also provide the code about data pre-processing in folder [data pre-processing](https://github.com/qianngli/MCNet/tree/master/data_pre-processing "data pre-processing"). The floder contains three parts, including training set augment, test set pre-processing, and band mean for all training set.**

Requirement
---------
**python 2.7, Pytorch 0.3.1, cuda 9.0**

Training
--------
**The ADAM optimizer with beta_1 = 0.9, beta _2 = 0.999 is employed to train our network.  The learning rate is initialized as 10^-4 for all layers, which decreases by a half at every 35 epochs.**

**You can train or test directly from the command line as such:**

###### # python train.py --cuda --datasetName CAVE  --upscale_factor 4
###### # python test.py --cuda --model_name checkpoint/model_4_epoch_200.pth

Result
--------
**To qualitatively measure the proposed ERCSR, three evaluation methods are employed to verify the effectiveness of the algorithm, including  Peak Signal-to-Noise Ratio (PSNR), Structural SIMilarity (SSIM), and Spectral Angle Mapper (SAM).**


| Scale  |  CAVE |  Harvard |  Pavia Centre |
| :------------: | :------------: | :------------: | :------------: | 
|  x2 |  45.332 / 0.9740 / 2.218 | 46.372 / 0.9832 / 1.875  | 35.422 / 0.9498 / 3.435 | 
|  x3 |  41.345 / 0.9527 / 2.789  |  42.783 / 0.9633 / 2.180 | 31.230 / 0.8690 / 4.650  |   
|  x4 | 41.345 / 0.9322 / 3.243 |  40.211 / 0.9374 / 2.384 | 28.912 / 0.7786 / 5.534  | 

Citation 
--------
**Please consider cite this paper if you find it helpful.**

@article{li2020exp,

	title={Exploring the Relationship between 2D/3D Convolution for Hyperspectral Image Super-Resolution},
	author={Q. Li, Q. Wang, and X. Li},
	journal={ IEEE Transactions on Geoscience and Remote Sensing},
	year={2020},
	doi={10.1109/TGRS.2020.3047363}
	}
  
--------

If you has any questions, please send e-mail to liqmges@gmail.com.

