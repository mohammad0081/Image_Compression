# Image_Compression
 Image Compression using K-Means clustering algorithm

* **What is Image Compression ?**
  Image compression in machine learning is a technique where machine learning models, particularly deep learning models, are used to reduce the size of images without significant loss of information .
  The goal of image compression is to eliminate image redundancy and store or transfer data in a more efficient manner

* **How can we use K-Means algorithm on image compression tasks?**
  K-means clustering can be used for image compression by reducing the number of colors in an image. Here’s how it works:
  
  1- Convert the image to RGB color space: An image is made up of several intensity values known as Pixels. In a colored image, each pixel is of 3 bytes (RGB - Red, Blue, Green) containing intensity values

  2- Flatten the image into a 2D array: Each row represents a pixel and each column represents a color channel (red, green, or blue)

  3- Apply K-means clustering: Group similar pixels together based on their RGB values and assign each group a mean RGB value1. The ‘k’ in K-means represents the desired number of colors in the compressed image

  4- Replace each pixel with the mean RGB value of its assigned cluster: This results in an image with fewer colors, but a similar overall appearance to the original

  5- Convert the compressed image back to its original color space, if necessary

  
