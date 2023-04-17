import torch

import numpy as np
import gradio as gr
from PIL import Image
from unet import UNet
from torchvision import transforms
from torchvision.transforms.functional import to_tensor, to_pil_image
import matplotlib.pyplot as plt

device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
# Load the trained model
model_path = 'cityscapes_dataUNet.pth'
num_classes = 10
model = UNet(num_classes=num_classes)
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()




# Define the prediction function that takes an input image and returns the segmented image
def predict_segmentation(image):
    print(device)
    # Convert the input image to a PyTorch tensor and normalize it
    image = Image.fromarray(image, 'RGB')
    # image = transforms.functional.resize(image, (256, 256))
    image = to_tensor(image).unsqueeze(0)
    image = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(image)
    image=image.to(device)
    
    print("input shape",image.shape) # input shape torch.Size([1, 3, 256, 256])
    print("input dtype",image.dtype) # input dtype torch.float32

    # Make a prediction using the model
    with torch.no_grad():
        
        print(image.shape, image.dtype) # torch.Size([1, 3, 256, 256]) torch.float32

        output= model(image)
        # print(output.shape,output.dtype) # torch.Size([1, 10, 256, 256]) torch.float32

        predicted_class = torch.argmax(output, dim=1).squeeze(0)                                                                                                                                                                                                                                                                                                                                                                                                                        
        predicted_class = predicted_class.cpu().detach().numpy().astype(np.uint8)
        print(predicted_class.dtype , predicted_class.shape) #  int64 (256, 256)
       

# Visualize the predicted segmentation mask
    plt.imshow(predicted_class)
    plt.show()                                                                                                                                                                                                                                                                                                                                                                   
    # Apply the inverse transform to convert the normalized image back to RGB
    # predicted_class = inverse_transform(torch.from_numpy(predicted_class))

    print("predicted class ",predicted_class)
    
    predicted_class = to_pil_image(predicted_class)

    # Return the predicted segmentation                                                                                                                                                                                                         
    return predicted_class

# Define the Gradio interface
input_image = gr.inputs.Image()
output_image = gr.outputs.Image(type='numpy')

gr.Interface(fn=predict_segmentation, inputs=input_image, outputs=output_image, 
             title='UNet Image Segmentation', 
             description='Segment an image using a UNet model').launch()