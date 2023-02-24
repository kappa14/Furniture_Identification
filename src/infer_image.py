import torch
import cv2
import torchvision.transforms as transforms
import torchvision
import numpy as np

device = ('cuda' if torch.cuda.is_available() else 'cpu')
labels = ['Sofa', 'Chair', 'Bed']

model = torchvision.models.vgg16().to(device)
checkpoint = torch.load('model.pth')

# define preprocess transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


def predict_image(filename):
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # read and preprocess the image
    image = cv2.imdecode(np.frombuffer(filename.read(), np.uint8), cv2.IMREAD_COLOR)

    # convert to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image)

    # add batch dimension
    image = torch.unsqueeze(image, 0)
    with torch.no_grad():
        outputs = model(image.to(device))
    output_label = torch.topk(outputs, 1)
    pred_class = labels[int(output_label.indices)]

    return pred_class

