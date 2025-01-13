# DEFIGNN: Detector-Free Image Captioning with Superpixels and Graph Neural Networks 

## Setup

### Getting the Code

Clone the repository and its submodules
```bash
git clone --recurse-submodules -j8 git@github.com:Delphboy/baseline.git
cd baseline

# Create directory for eval
mkdir vis/ 
```


### Setting up the Python Environment

```bash
python3 -m venv .venv

source .venv/bin/activate

python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python3 -m pip install gensim
python3 -m pip install matplotlib
python3 -m pip install pandas
python3 -m pip install scikit-image
python3 -m pip install six
python3 -m pip install typing
python3 -m pip install requests
python3 -m pip install h5py

```

### Additional Data

For a full guide, please see this [blog post](https://henrysenior.com/words/2024-04-03-coco-supplementary-dataset-download-guide).

#### Directory Creation

You're going to need to set up an empty directory where you can set everything up ie) `mkdir data/`

#### Karpathy Splits
As the COCO test server is a thing of the past, [Karpathy proposed an alternative split for the COCO dataset](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Karpathy_Deep_Visual-Semantic_Alignments_2015_CVPR_paper.pdf) which is $113, 287$ / $5000$ / $5000$ train/val/test. I keep a copy of the [JSON files on my GitHub](https://github.com/Delphboy/karpathy-splits)

```bash
cd data/

wget -c https://github.com/Delphboy/karpathy-splits/raw/main/dataset_coco.json?download= -O dataset_coco.json
```

You will then need to run the following [prepro_labels.py](scripts/prepro_labels.py) script.

```bash
python3 prepro_labels.py --input_json data/dataset_coco.json --output_json data/cocotalk.json --output_h5 data/cocotalk
```

#### Bottom Up Top Down Object Detection

Most of the time you will have to download the BUTD data
```bash
mkdir data/bu_data
cd data/bu_data
wget https://storage.googleapis.com/up-down-attention/trainval.zip
unzip trainval.zip
```

Then run the `make_bu_data.py` script to extract the `npy`/`npz` files:

```bash
python3 script/make_bu_data.py --output_dir data/butd
```

Once this has run, you should have three folders in `data/`
1. `butd_att`: Which contains an $N\times2048$-dimension numpy array of ResNet101 features for the $N$ objects detected in the image
2. `butd_fc`: Containing a $2048$-dimension ResNet101 feature for the whole image
3. `butd_box`: That contains the bounding box information for the detected objects



### Enabling SPICE

The SPICE evaluation metric requires that you have Java 1.8 and some additional libraries stored in `coco-caption/pycocoevalcap/spice/lib`. You may need to follow [this guide](https://henrysenior.com/words/2024-04-03-adding-spice-to-meshed-memory) to get some of the library files if they don't clone as expected.

```bash
cd coco-caption/
bash get_stanford_models.sh
```


## Running The Code

### Locally

To run the code locally, please consider making use of the `run.sh` script.

### HPC

There are a selection of `.qsub` files that should demonstrate how to train and test the models.


