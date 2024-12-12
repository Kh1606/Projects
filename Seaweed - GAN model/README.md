# Seaweed Image Generation Using GANs
![alt text](https://github.com/Kh1606/Projects/blob/main/Seaweed%20-%20GAN%20model/gan.gif)

Tools I used in this Project: ![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-%23F9A825.svg?style=for-the-badge&logo=googlecolab&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Google Drive](https://img.shields.io/badge/Google%20Drive-4285F4?style=for-the-badge&logo=googledrive&logoColor=white)
## Project Overview
This project focuses on generating images of seaweed using Generative Adversarial Networks (GANs). Similar to typical GAN architectures, it consists of two main components:

- **Generator**: Creates synthetic seaweed images from random noise.
- **Discriminator**: Distinguishes between real seaweed images and those created by the generator.
- 
## Key Parts of the Code

### Imports and Setup
The project begins by importing necessary libraries, primarily using TensorFlow to build and train the neural networks. Key components include:

- **Layers**: `Dense`, `Conv2D`, `UpSampling2D`
- **Optimizer**: `Adam`

These are used to construct the generator and discriminator models.

### Data Loading (`load_images()`)
The function `load_images()` reads images from a given directory (`/content/gdrive/MyDrive/new/`) and resizes them to a size of 1024x1024. 

- Pixel values are normalized to the range `[0, 1]` to facilitate easier training of the neural networks.

### Model Building

#### Generator (`build_generator()`)
- The generator takes a latent vector as input and uses `Dense`, `Reshape`, `Conv2D`, and `UpSampling2D` layers to progressively upscale the vector into an image of size 1024x1024.
- It starts from a small representation and increases the resolution step by step using `UpSampling2D`.
- The final layer uses a `tanh` activation function to generate RGB images with values in the range `[-1, 1]`.

#### Discriminator (`build_discriminator()`)
- The discriminator takes an image as input and attempts to classify it as real or fake.
- It consists of a series of `Conv2D` layers with strides of 2, followed by `LeakyReLU` activations and dropout layers to reduce overfitting.
- The output layer uses a `sigmoid` activation function to produce a probability score indicating whether the input image is real or fake.

### Combined Model for GAN Training
The **combined model** is built by connecting the generator and discriminator:

- It feeds the output of the generator (i.e., a synthetic image) into the discriminator.
- The combined model is trained to adjust the generator such that it can fool the discriminator into classifying generated images as real.

### Training Process (`train_gan()`)
The `train_gan()` function manages the training of both the discriminator and generator.

#### Training Loop:
- In each iteration:
  - The **discriminator** is trained on both real and generated images to minimize classification error for real images and maximize it for generated images.
  - The **generator** is trained via the combined model to produce images that the discriminator classifies as real.
  
#### Training Metrics:
- During each epoch, loss values for both the generator and discriminator are printed to track training progress.
- Images are periodically saved using the `save_generated_images()` function to visualize the generator's learning over time.

### Saving Generated Images (`save_generated_images()`)
This function generates and saves images produced by the generator during training.

- Images are saved in PNG format, allowing the evaluation of the generator's performance at different training epochs.

---

## Usage

1. **Install Dependencies**: Make sure you have TensorFlow and other required libraries installed.
2. **Data Preparation**: Place seaweed images in the `/content/gdrive/MyDrive/new/` directory for training.
3. **Train the Model**: Run the `train_gan()` function to start training.
4. **Monitor Training**: The generated images are saved periodically to monitor progress.

---

## Results
The project aims to generate realistic seaweed images. Check the saved images periodically to observe how the quality improves as training progresses.

---

## Acknowledgments
- TensorFlow for providing the framework for building GANs.
- The deep learning community for sharing knowledge on GANs and model-building techniques.

--
