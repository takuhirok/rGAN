#!/bin/bash

if [ $# -ne 6 -a $# -ne 7 ]; then
    echo "Usage: $0 MODEL GAN DATASET DATA_NOISE_TYPE DATA_NOISE_RATE OUT [OPTION]" 1>&2
    exit 1
fi

MODEL=$1
GAN=$2
DATASET=$3
DATA_NOISE_TYPE=$4
DATA_NOISE_RATE=$5
OUT=$6
if [ $# -eq 7 ]; then
    OPTION=$7
else
    OPTION=""
fi

MODEL_NOISE_TYPE=${DATA_NOISE_TYPE}
if [ ${MODEL} = "acgan" ]; then
    TRAINER=racgan
    MODEL_NOISE_RATE=0
elif [ ${MODEL} = "racgan" ]; then
    TRAINER=racgan
    MODEL_NOISE_RATE=${DATA_NOISE_RATE}
elif [ ${MODEL} = "cgan" ]; then
    TRAINER=rcgan
    MODEL_NOISE_RATE=0
elif [ ${MODEL} = "rcgan" ]; then
    TRAINER=rcgan
    MODEL_NOISE_RATE=${DATA_NOISE_RATE}
else
    echo "expected [MODEL] is acgan, racgan, cgan, or rcgan (got ${MODEL})"
    exit 1
fi

if [ ${GAN} = "sngan" ]; then
    G_CHANNELS=256
    G_SPECTRAL_NORM=0
    D_CHANNELS=128
    D_DROPOUT=0
    D_SPECTRAL_NORM=1
    D_POOLING=sum
    GAN_LOSS=hinge
    G_LR=2e-4
    D_LR=2e-4
    BETA1=0.0
    BETA2=0.9
    NUM_CRITIC=5
    LAMBDA_GP=0.0
    LAMBDA_CT=0.0
    FACTOR_M=0
elif [ ${GAN} = "wgangp" ]; then
    G_CHANNELS=128
    G_SPECTRAL_NORM=0
    D_CHANNELS=128
    D_DROPOUT=0
    D_SPECTRAL_NORM=0
    D_POOLING=mean
    GAN_LOSS=wgan
    G_LR=2e-4
    D_LR=2e-4
    BETA1=0.0
    BETA2=0.9    
    NUM_CRITIC=5
    LAMBDA_GP=10.0
    LAMBDA_CT=0.0    
    FACTOR_M=0
elif [ ${GAN} = "ctgan" ]; then
    G_CHANNELS=128
    G_SPECTRAL_NORM=0
    D_CHANNELS=128
    D_DROPOUT=1
    D_SPECTRAL_NORM=0
    D_POOLING=mean
    GAN_LOSS=wgan
    G_LR=2e-4
    D_LR=2e-4
    BETA1=0.0
    BETA2=0.9    
    NUM_CRITIC=5
    LAMBDA_GP=10.0
    LAMBDA_CT=2.0    
    FACTOR_M=0
else
    echo "expected [GAN] is sngan, wgangp, or ctgan (got ${GAN})"
    exit 1
fi

python train.py --dataset ${DATASET} --data_noise_type ${DATA_NOISE_TYPE} --data_noise_rate ${DATA_NOISE_RATE} --g_channels ${G_CHANNELS} --g_spectral_norm ${G_SPECTRAL_NORM} --d_channels ${D_CHANNELS} --d_dropout ${D_DROPOUT} --d_spectral_norm ${D_SPECTRAL_NORM} --d_pooling ${D_POOLING} --trainer ${TRAINER} --gan_loss ${GAN_LOSS} --g_lr ${G_LR} --d_lr ${D_LR} --beta1 ${BETA1} --beta2 ${BETA2} --num_critic ${NUM_CRITIC} --lambda_gp ${LAMBDA_GP} --lambda_ct ${LAMBDA_CT} --factor_m ${FACTOR_M} --model_noise_type ${MODEL_NOISE_TYPE} --model_noise_rate ${MODEL_NOISE_RATE} --out ${OUT} ${OPTION}
