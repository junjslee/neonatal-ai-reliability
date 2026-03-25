FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-devel

# Avoid interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    apt-utils libgl1-mesa-glx openslide-tools libgtk2.0-dev \
    libxext6 libxrender-dev python3-tk libglib2.0-0 libsm6 \
    libxrender1 libfontconfig1 dcm2niix vim sudo \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install your packages
RUN pip install --no-cache-dir \
    albumentations==1.4.3 gpustat grad-cam imutils ipywidgets kornia==0.7.0 \
    libauc==1.3.0 librosa==0.10.1 livelossplot==0.5.5 lifelines medpy \
    monai==1.3.0 natsort==8.4.0 neurokit2==0.2.7 numpy==1.26.0 \
    opencv-python==4.8.0.76 openpyxl openslide-python pandas==2.2.3 \
    peft plotly pillow pydicom "pydicom[gdcm]" pylibjpeg pylibjpeg-libjpeg \
    pylibjpeg-openjpeg pyyaml pytorch-gradcam pytorch-lightning python-pptx \
    pycox scikit-image scikit-learn==1.3.2 scikit-survival scikit-posthocs \
    scipy==1.11.2 shap seaborn segmentation-models-pytorch==0.4.0 \
    tiatoolbox torchmetrics tqdm transformers totalsegmentator