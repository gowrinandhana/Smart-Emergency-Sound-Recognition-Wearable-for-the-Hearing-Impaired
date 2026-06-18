# Dataset Information

## Overview

This project uses a combination of the ESC-50 dataset and custom-collected audio samples for environmental sound classification.

## Sound Classes

The model was trained on 13 sound categories:

1. car_horn
2. crying_baby
3. dog
4. door_knock
5. doorbell
6. fire_smoke_alarm
7. glass_breaking
8. gunshot
9. rain
10. screaming
11. siren
12. thunderstorm
13. train

## Data Sources

### ESC-50 Classes

The following classes were obtained from the ESC-50 environmental sound dataset:

- car_horn
- crying_baby
- dog
- door_knock
- doorbell
- glass_breaking
- rain
- siren
- train

Dataset Source:
https://github.com/karolpiczak/ESC-50

### Custom Classes

The following classes were collected from publicly available audio sources:

- fire_smoke_alarm
- gunshot
- screaming
- thunderstorm

## Preprocessing Steps

The audio dataset underwent the following preprocessing operations:

- Audio augmentation
- Audio padding and trimming
- Resampling to 16 kHz
- Feature extraction using YAMNet embeddings
- Dataset preparation for SVM classification

## Project Purpose

The dataset was prepared for training an AI-powered sound recognition system that detects important environmental sounds and alerts hearing-impaired users through vibration and visual notifications.
