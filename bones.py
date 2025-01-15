import kagglehub

# Download latest version
path = kagglehub.dataset_download("nikhilroxtomar/ct-heart-segmentation")

print("Path to dataset files:", path)