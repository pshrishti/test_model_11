import torchxrayvision as xrv
import skimage, torch, torchvision
import matplotlib.pyplot as plt
import numpy as np

import torch
import numpy as np
import matplotlib.pyplot as plt
import skimage.filters

def get_img(img_path):
    # Prepare the image:
    img = skimage.io.imread(img_path)
    img = xrv.datasets.normalize(img, 255) # convert 8-bit image to [-1024, 1024] range
    # Check that images are 2D arrays
    if len(img.shape) > 2:
        img = img[:, :, 0]
    if len(img.shape) < 2:
        print("error, dimension lower than 2 for image")

    img = img[None, :, :]
    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)])

    img = transform(img)

    return img


def get_output(img_val):
    final_val = []
    img = torch.from_numpy(img_val)

    # Load model and process image
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    outputs = model(img[None,...]) # or model.features(img[None,...])
    output = (dict(zip(model.pathologies,outputs[0].detach().numpy())))
    label = list(output.keys())
    prob_value = list(output.values())
    return label,prob_value,output


  
# ## heatmap
def plot_heatmap(img_val,output_dictonary):
    # filtered_dict = {key: value for key, value in output_dictonary.items() if value > 0.50}
    label = list(output_dictonary.keys())
    prob_value = list(output_dictonary.values())

    # val = list(output.keys())
    img = torch.from_numpy(img_val).unsqueeze(0)

    # Load model and process image
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    img = img.requires_grad_()

    # Assuming we have a list of target values and their corresponding labels
    index_values = []  # Example target values
    target_labels = []  # Example target labels
    for index, value in enumerate(label):
        index_values.append(index)
        target_labels.append(value)
    grads = []
    for target in index_values:
        outputs = model(img)
        grad = torch.autograd.grad(outputs[:, target], img)[0]
        grads.append(grad)


    # Calculate the number of rows needed to display images (4 images per row)
    num_rows = int(np.ceil(len(index_values) / 4))

    # Create subplots with the desired structure
    fig, axs = plt.subplots(num_rows, 4, figsize=(8, 4*num_rows))

    for i, target in enumerate(index_values):
        # Calculate the row and column indices for the current image
        row_idx = i // 4
        col_idx = i % 4

        # Perform the same image processing and plotting as before
        blurred = skimage.filters.gaussian(grads[i][0][0].detach().cpu().numpy() ** 2, sigma=(5, 5), truncate=3.5)

        ax = axs[row_idx, col_idx]
        ax.imshow(img[0][0].detach().cpu().numpy(), cmap="gray", aspect='auto')
        ax.imshow(blurred, alpha=0.5)
        ax.axis('off')

        # Add target class, target name, and target value as text annotations
        target_class = target
        target_name = target_labels[i]
        target_value = outputs[0, target_class].item()

        text = f"Target Name: {target_name}\nTarget Value: {target_value:.2f}"
        ax.text(0.5, -0.1, text, transform=ax.transAxes,
                fontsize=6, fontweight='bold', ha='center', va='center', color='white',
                bbox=dict(facecolor='black', edgecolor='black', pad=4))

    # Hide any empty subplots
    for i in range(len(index_values), num_rows * 4):
        row_idx = i // 4
        col_idx = i % 4
        axs[row_idx, col_idx].axis('off')

    plt.tight_layout()
    plt.show()


def convert_to_text(output_dictonary):
    with open("img1.txt", 'w') as f: 
        for key, value in output_dictonary.items(): 
            f.write('%s:%s\n' % (key, value))
            
    f=open('img1.txt','r')
    data = f.read()

    return 'Results\n' + data
# ## Bar graph 
# # Extract the class labels and corresponding probabilities
# class_labels = list(output.keys())
# probabilities = list(output.values())

# # # Create a histogram
# # plt.figure(figsize=(12, 6))
# # plt.bar(class_labels, probabilities, color='skyblue')
# # plt.xlabel('Class Labels')
# # plt.ylabel('Probabilities')
# # plt.title('Predictions')
# # plt.xticks(rotation=90)
# # plt.tight_layout()

# # Display the histogram
# # plt.show()
# Create a histogram
def create_histogram(output_dictonary):
    # Create a histogram
    label = list(output_dictonary.keys())
    prob_value = list(output_dictonary.values())

    plt.figure(figsize=(12, 6))
    bars = plt.bar(label, prob_value, color='skyblue')
    plt.xlabel('Class Labels')
    plt.ylabel('Probabilities')
    plt.title('Predictions')
    plt.xticks(rotation=90)
    plt.tight_layout()

    # Add probability values as text annotations above each bar
    for bar, prob in zip(bars, prob_value):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{prob:.2f}', ha='center', va='bottom', fontsize=10)

    # Display the histogram
    plt.show()

# img = get_img("img/1.jpeg")

# # # label = get_output(img)[0]
# # # prob_value = get_output(img)[1]
# output = get_output(img)[2]
# # # print("Labels  are:",label)
# # # print("\n")
# # # print("Values  are:",prob_value)
# # print('Output is:',output)

# plot_heatmap(img,output)# # Print results
# # convert_to_text(output)
# # # create_histogram(output)