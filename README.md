## Instructions for training DeepLab v3+ on customized (AutoVision) dataset

### 1. Before start

The folder is originally taken from [DeepLab's official GitHub repository](https://github.com/tensorflow/models/tree/master/research/deeplab). Some files are modified in order for more customized training and to be adapted for the AutoVision project. Thus, this folder is already different from the original one.

Please pay more attention to the instructions where `${SOME_DESCRIPTION}` appears because **you are required to replace these with information of your own customized dataset**. Here we also provide the pre-trained models of the backbone **MobileNet v2** and **Xception 65** for fine-tuning.

With the setting of images with maximum 513 pixes in long edge, and totally 15 defined classes, the pre-trained models can achieve mIoU of 56.5% and 69.2% respectively for the backbone **MobileNet v2** and **Xception 65**.

The newly defined class labels are as follows:

| ID | Class Name | RGB Color |
| :-: | :--------: | :--------: |
| 0 | Barrier | 90, 120, 150 |
| 1 | Building | 70, 70, 70 |
| 2 | Car | 0, 0, 142 |
| 3 | Terrain | 152, 251, 152 |
| 4 | Heavy Vehicle | 0, 60, 100 |
| 5 | Motorcycle | 119, 11, 32 |
| 6 | Paved Road | 128, 64, 128 |
| 7 | Pedestrian Area | 170, 170, 170 |
| 8 | Person | 220, 20, 60 |
| 9 | Pole Object | 250, 170, 30 |
| 10 | Sky | 70, 130, 180 |
| 11 | Unpaved Road | 220, 180, 50 |
| 12 | Vegetation | 107, 142, 35 |
| 13 | Water | 0, 170, 30 |
| 14 | Other Objects | 255, 255, 255 |
| 255 | Undefined | 0, 0, 0 |

Now you can download the package file `tensorflow_models.zip` from this [link](https://drive.google.com/open?id=1cj-NrZaC83ZVtopNqFtIStCSbh8uOHVy).

### 2. Create customized dataset

#### Step 1: Place images and labels correctly

In the location of `tensorflow_models.zip`, run the following commands in the terminal.
```
git clone https://github.com/lizuoyue/AutoVision.git
unzip tensorflow_models.zip
cp AutoVision/model.py tensorflow_models/research/deeplab
cp AutoVision/export*.py tensorflow_models/research/deeplab
cp AutoVision/local_test*.sh tensorflow_models/research/deeplab
cd tensorflow_models/research/deeplab/datasets
cp -r default ${DATASET_NAME}
```
After doing so, you should see a folder structure like this:
```
+ ${DATASET_NAME}
    + leftImg8bit
        + train
            + train_default	# This is the folder to place training images (TODO)
        + val
            + val_default	# This is the folder to place validation images (TODO)
    + gtFine
        + train
            + train_default	# This is the folder to place training GT labels (TODO)
        + val
            + val_default	# This is the folder to place validation GT labels (TODO)
    + init_models
        + mobilenet_v2		# This is the folder to place pre-trained MobileNet v2 model (DONE)
        + xception_65		# This is the folder to place pre-trained Xception 65 model (DONE)
```

**Images**: All the training images are supposed to be placed in the folder `${DATASET_NAME}/leftImg8bit/train/train_default/`. Alternatively, you can also create several folders in `${DATASET_NAME}/leftImg8bit/train/` and place images separately in each folder you created. The procedure is same for validation images in `${DATASET_NAME}/leftImg8bit/val/`. For file extension, only `.jpg` and `.png` are supported.
	
**Labels**: All the training labels are supposed to be placed in the folder `${DATASET_NAME}/gtFine/train/train_default/`. If you create several folders in `${DATASET_NAME}/leftImg8bit/train/`, here the folder structure should be exactly the same. The procedure is same for validation labels in `${DATASET_NAME}/gtFine/val/`. Please also note that **the label filename should be the same as image filename except the extension and the postfix**. For file extension, only `.png` is supported.

**Pre-trained Models**: Both the pre-trained models with different backbones (MobileNet v2 and Xception 65) are already correctly placed in the folder `${DATASET_NAME}/init_models/`.

#### Step 2: Modify data generator

In the folder `tensorflow_models/research/deeplab/datasets/`, modify `data_generator.py`. Do not forget to modify `${DATASET_NAME}` with the name of your customized dataset.

In around line 98 of file `data_generator.py`, add:

```
_${DATASET_NAME}_INFORMATION = DatasetDescriptor(
	splits_to_sizes={
		'train': ${NUMBER_OF_TRAINING_IMAGES_IN_TOTAL},
		'val': ${NUMBER_OF_VALIDATION_IMAGES_IN_TOTAL},
	},
	num_classes=${NUMBER_OF_CLASSES_OF_THE_DATASET},
	ignore_label=255,
)
```

Please do not modify `ignore_label` here, let it be `255`. Consistently, the ignored label in the label files should be `255` as well. Note that `num_classes` should not count `ignore_label`. If you want to use the newly defined labels as shown in the table above, please set `num_classes=15`, otherwise set the corresponding one. Next, **add** the created information to the dictionary `_DATASETS_INFORMATION` (just below):

```
_DATASETS_INFORMATION = {
    ...,
    '${DATASET_NAME}': _${DATASET_NAME}_INFORMATION,
    ...,
}
```

Save `data_generator.py` and close.

#### Step 3: Modify data building files

Again, in the folder `tensorflow_models/research/deeplab/datasets/`, run the following commands in the terminal.

```
cp build_cityscapes_data.py build_${DATASET_NAME}_data.py
```

Modify `build_${DATASET_NAME}_data.py`:

(1) In line 73, replace `'./cityscapes'` with `'./${DATASET_NAME}'`;

(2) In line 92, replace `'_leftImg8bit'` with `''` if you want use all the training images in the dataset folder (no specific postfix), otherwise use your own postfix;

(3) In line 93, replace `'_gtFine_labelTrainIds'` with `''` if you want use all the training labels in the dataset folder (no specific postfix), otherwise use your own postfix;

(4) In line 98, replace `'png'` with `'${YOUR_IMAGE_EXTENSION}'`;

(5) In line 142, replace `'png'` with `'${YOUR_IMAGE_EXTENSION}'`, and replace `channels=3` with `channels=${NUMBER_OF_CHANNELS_OF_IMAGE}`. For AutoVision, since grayscale image is used, here we set `channels=1`.

#### Step 4: Generate TensorFlow data files

Finally, in the folder `tensorflow_models/research/deeplab/datasets/`, run the following commands in the terminal.
```
mkdir -p ./${DATASET_NAME}/tfrecord
python build_${DATASET_NAME}_data.py \
  --cityscapes_root="./${DATASET_NAME}" \
  --output_dir="./${DATASET_NAME}/tfrecord"
```

If it runs successfully, you will see several files in the folder `${DATASET_NAME}/tfrecord`.

### 3. Train, validate and export model

Go to the folder `tensorflow_models/research/deeplab/`. Modify `local_test_mobilenet_v2.sh` and `local_test_xception_65.sh`. In line 43, replace `DATASET_FOLDER=""` with `DATASET_FOLDER="${DATASET_NAME}"`, and run the following commands in the terminal.

```
sh local_test_mobilenet_v2.sh
# or
sh local_test_xception_65.sh
```
Do not forget to press `ENTER` when it requires you to do so. All the experiment results will be placed in the folder `tensorflow_models/research/deeplab/datasets/${DATASET_NAME}/exp`. You can use `tensorboard` to see the loss decreasing or the evaluation results.

#### 3.1 Arguments that should be noticed
(1) `--{train,val,vis}_crop_size="A,B"`. `A` and `B` should satisfy the form of `n * output_stride + 1`;

(2) `--num_clones=1`. You can set a higher number for training using multiple GPUs.

(3) `--base_learning_rate=0.00001`. This is relatively small learning rate. You can set a higher number for training (like `0.007`) or fine-tune (like `0.0001`).

#### 3.2 Arguments that are added additionally

(4) `--max_ckpt=5`. Maximum number of checkpoints to keep. 

(5) `--img_channels=1`. `1` for grayscale image, `3` for RGB image.

(6) `--init_exclude="[]"`. The argument should be a **string**. Both the pre-trained model is used for 1-channel input and 15-channel output. If you use RGB image, please exclude `"['xception_65/entry_flow/conv1_1/']"` for Xception 65 or `"['MobilenetV2/Conv']"` for MobileNet v2. If you use different number of classes, please exclude `"['logits']"` for either model. You can also exclude multiple variables by providing the **PREFIX** of the variables in a string of Python list of strings, like `"['MobilenetV2/Conv','logits']"`. For the names of all the variables, please refer to [Xception 65](xception_65.txt) and [MobileNet v2](mobilenet_v2.txt) or use `tensorboard` to have a look.

(7) `--train_only="[]"`. The argument should be a **string**. Same as above, you can also provide the **PREFIX** of the variables you want to train **ONLY** in a string of Python list of strings. For example, if you have different number of classes and still want to use the pre-trained model, just simply set `--init_exclude="['logits']"` and `--train_only="['logits']"`.

(8) `--aug_labels="[11]"` and `--aug_weights="[3]"`. `11` refers to the class unpaved road and its weight is `3` times higher than other classes. You can set more labels with higher weights by providing a string of Python list of numbers to each argument.

For other arguments, you can refer to the inside of the files `train.py`, `eval.py`, `vis.py` and `export_model.py`.
