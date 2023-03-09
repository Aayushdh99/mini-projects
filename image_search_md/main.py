import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import json
import os
import time
from PIL import Image
import numpy as np

def transform_fn(image):
    # Get the size of the image
    image_height, image_width = image.size

    # Calculate the padding required to make the image square
    padding = (0, max(0, (image_height - image_width) // 2), 0, max(0, (image_width - image_height) // 2))

    # Apply padding, resize and convert to tensor
    transform = transforms.Compose([
    transforms.Pad(padding, fill=0, padding_mode='constant'),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
    ])

    return transform(image)

def im_convert(tensor):
    image = tensor.cpu().clone().detach().numpy()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.485, 0.456, 0.406)) + np.array((0.229, 0.224, 0.225))
    # image = image.clip(0, 1)
    return image

def label_output(pred):
    # Opening labels JSON file
    with open('coco_labels.json') as json_file:
        data = json.load(json_file)
  
    tag_list = []
    for id, j in enumerate(pred[0]):
        if int(j) == 1:
            tag_list.append(data[str(id+1)])
    return tag_list

def calculate_labels(image):
    # Define hyperparameters
    num_classes = 90

    # Define device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define Resnet50 model
    model = models.resnet50(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    # model = model.to(device)

    # Define loss function and optimizer
    # criterion = nn.BCEWithLogitsLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.load_state_dict(torch.load('model_4.pth', map_location=torch.device('cpu')))
    model.eval()

    # Transform Image
    img = transform_fn(image)

    with torch.no_grad():
        output = model(img.unsqueeze(0))
        pred = np.array(output > 0.4, dtype=float)

    return label_output(pred)

def process_data(filename):
    try:
        image = Image.open('folder_path/'+filename).convert('RGB')
        return calculate_labels(image)
    except Exception as e:
        return e

def watch_directory(path, callback):
    # get initial file list
    file_list = set(os.listdir(path))

    while True:
        # wait for 1 second
        time.sleep(1)

        # get updated file list
        updated_file_list = set(os.listdir(path))

        # find new files
        new_files = updated_file_list - file_list

        # call callback function with new files
        if new_files:
            for new_file in new_files:
                print(callback(new_file))
                print('found')

        # update file list
        file_list = updated_file_list


watch_directory('folder_path', process_data)