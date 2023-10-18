# Using CNNs to Identify Contrails in Satellite Images and Reduce Global Warming

**Overview:**  
This repository contains our solution to the "Identify Contrails to Reduce Global Warming" Kaggle competition hosted by Google Research. Our solution involves using neural networks for binary semantic segmentation on satellite images to identify contrails.

See our medium article: https://pub.aimind.so/ai-driven-skies-paving-the-way-for-greener-air-travel-85b335acf153

**Authors:** 
- Maya Halevy
- Sasha Baich
- Eliott Caen

## Motivation:
Contrails, the line-shaped clouds formed by airplanes, account for about 1% of human-caused global warming. At night, these contrails act as insulators. With increasing aviation activities, the environmental impact of these contrails is on the rise. Our model helps to quickly and reliably automate the process of identifying contrails, paving the way for pilots to adjust their flight paths and minimize contrail formation.

## Dataset:
- Provided by Kaggle with around 22,000 samples.
- Contains satellite images captured in 9 infrared ABI bands.
- Due to similarity between adjacent bands, we used bands 8, 12, and 16.
- Class imbalance is notable, with only an average of 1.2% of the image pixels being contrails.

## Model:
- **U-Net Architecture:** A convolutional neural network ideal for image segmentation tasks.
- **Variations:** 
  - VGG16 (pre-trained on ImageNet)
  - MobileNetV2
  - Custom Residual U-Net with Attention

## Metrics:
- **Dice Coefficient:** Measures overlap between predicted mask and ground truth. Ranges from 0 (no overlap) to 1 (perfect overlap).

## Post-Processing Techniques:
- Averaging Predictions: Rotate or flip the image multiple times, predict, then average out the predictions. Improved the dice coefficient by up to 7% in some cases.

## Results:
The custom-built U-Net with residual connections and attention mechanisms surpassed other models.

## Challenges:
- Extensive training time for untrained models.
- Limited computational resources.

## Future Scope:
- Exploring more model architectures.
- Utilizing the temporal aspect of the dataset.
- Fine-tuning model hyperparameters.

## Conclusion:
Our model's results are apt for informing pilots on contrail formation avoidance. Even a 50% reduction in contrail formation can be significant. We aim to make air travel eco-friendlier and contribute to mitigating climate change.

