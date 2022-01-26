import numpy as np
import torch
import torchvision.transforms as T

import core
from PIL import Image

img = Image.open("test.jpg")
image_array = np.array(img)
image_array = np.transpose(image_array, (2,0,1))
x, y, z = image_array.shape

print(image_array.shape)

img_torch = torch.tensor(image_array)
print(img_torch.shape)
x = torch.randn(3, 456, 321)
print(x.shape)
img_output = core.imresize(img_torch, scale=10.0)

print(core.imresize(x, scale=2).shape)


transform = T.ToPILImage()

img_out = transform(img_output)
img_out.save("output.jpg")
