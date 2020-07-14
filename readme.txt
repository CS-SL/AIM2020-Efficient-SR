Dataset
  1.We use DIV2K dataset for training. 
    1-1.Download DIV2K and unzip on dataset directory.
  2.To accelerate training, we first convert training images to numpy format as follow.
     cd dataset && python png2npy.py
  3.Put the test images in dataset/testset/ directory.


Training Model
   CUDA_VISIBLE_DEVICES=1 python train.py --snapshot_dir weights/spsr/ --log_dir ./log/spsr --lr 2.5e-4 --decay 100 --batch_size 32 --upscaling_factor 4


Test Pretrained Model
  python eval.py --data_test ./dataset/testset/ --resume_dir weights/spsr/model_493_epoch.pth --result_save_dir results/DIV2K-test/ --upscaling_factor 4

  You can find the result images from 'results/DIV2K-test/' folder.

