import torch
import torch.nn as nn
import coremltools as ct

# --- 1. THE BLUEPRINT (Pasted directly here) ---
class openedOrClosedCNN(nn.Module):
    def __init__(self):
        super(openedOrClosedCNN, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.fc(x)
        return x

# --- 2. THE CONVERSION LOGIC ---

# Initialize and load weights
model = openedOrClosedCNN()
model.load_state_dict(torch.load('racket_arm_cnn.pth', map_location=torch.device('cpu')))
model.eval()

# Trace
example_input = torch.rand(1, 3, 224, 224) 
traced_model = torch.jit.trace(model, example_input)

# Convert
image_input = ct.ImageType(
    name="input_1", 
    shape=example_input.shape, 
    scale=1/255.0, 
    color_layout=ct.colorlayout.RGB
)

mlmodel = ct.convert(
    traced_model,
    inputs=[image_input],
    minimum_deployment_target=ct.target.iOS15
)

mlmodel.save("ArmTracker.mlpackage")
print("✅ Done! ArmTracker.mlpackage created.")