# Furniture Identification

Hello there! Thank you for reading me.\
This repository covers a basic implementation to detect three types of furniture samples - Sofa, Chair and Bed.\
\
The goal of the project was to ensure the ML model completes an end-to-end pipeline. \
Final conclusion : Almost reached there. (Almost!)

## Dataset

A well-balanced limited dataset is employed for this project. \
\
Each of the three classes comprises of 100 samples each, with varying dimensions. For the sake of simplicity, all samples are against the white background.\
Hence, dataset size = 300 (3 x 100)

## Wishlist of the challenge

- [x] Build a classification model (Deep learning preferred).
- [x] Build an API to access the model. The API should accept an image as input and return  the predicted label (category) as output  (Preferred frameworks based on Python).
- [ ] Create a Docker image of your code by following docker best practices.
- [x] Implement CI/CD pipeline on Github Actions.
- [x] Add a clear README file with instructions.

## Backbone Architecture

[VGG16](https://arxiv.org/abs/1409.1556) architecture model was employed as our backbone architecture. Pre-trained weights of ImageNet was enabled as well in order to use low-level representations over our small dataset.  

## Augmentations
Training samples are `Resize` to (224 x 224 x 3) resolution, followed by `RandomHorizontalFlip` and `RandomRotation` (0-20 degrees).\
\
Validation samples are subjected to `Resize` only.  

## Model Training
Using Cross Entropy loss, the model was trained for 50 epochs. \
After every 10 epochs, the weights are locally stored.

```Python
python3 train.py --epochs 50
```

## Model Evaluation
Loss Plot\
![Loss](https://github.com/kappa14/Furniture_Identification/blob/build_model/outputs/Loss.png)
\
Accuracy Plot\
![Accuracy](https://github.com/kappa14/Furniture_Identification/blob/build_model/outputs/Accuracy.png)

## API to access model
Owning to the success of developing microservices and fast web APIs, `FastAPI` is employed to locally wrap the ML inference model.\
\
Once the `fast_imageAPI.py` file is created, just run the following command :
```Python
python -m uvicorn fast_imageAPI:app --reload
```

Execution Sample ([http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)):\
![API Sample](https://github.com/kappa14/Furniture_Identification/blob/build_model/outputs/API_sample.PNG)
