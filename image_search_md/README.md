# ImageMetaSearch
  
## Overview:
ImageMetaSearch is an image search tool that allows users to search for images based on their content. The tool uses state-of-the-art Deep Learning models, EfficientNet-B4 and ResNet-50, to predict labels/tags for images in a given directory. This tool allows users to quickly find images based on the objects and content within them.

To train the models, we used the COCO 2017 dataset, which contains a large number of images with corresponding JSON files containing object coordinates and labels. We created a custom dataset for our use case by selecting only the images and JSON files with corresponding image names and labels, and then trained the models for multi-label classification. We selected the most accurate model and integrated it with a sqlite database for storing images name/paths and their corresponding tags.

Our tool also includes a script that monitors the directory for changes and automatically updates the database with newly added images and their predicted tags. This ensures that the database is always up-to-date and that users can quickly find the images they are looking for.

Overall, ImageMetaSearch is a powerful and efficient tool for searching images based on their content, making it an ideal choice for individuals who need to manage large numbers of images.
	
Dataset Used: https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset
