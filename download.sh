#!/bin/bash

# Base URL for downloading model files
BASE_URL="https://huggingface.co/microsoft/OmniParser/resolve/main"

# Define folder structure and create folders
mkdir -p weights/icon_detect
mkdir -p weights/icon_caption_florence
mkdir -p weights/icon_caption_blip2

# Declare an associative array of required files with paths
declare -A model_files=(
  ["weights/icon_detect/model.safetensors"]="$BASE_URL/icon_detect/model.safetensors"
  ["weights/icon_detect/model.yaml"]="$BASE_URL/icon_detect/model.yaml"
  ["weights/icon_caption_florence/model.safetensors"]="$BASE_URL/icon_caption_florence/model.safetensors"
  ["weights/icon_caption_florence/config.json"]="$BASE_URL/icon_caption_florence/config.json"
  ["weights/icon_caption_blip2/model.safetensors"]="$BASE_URL/icon_caption_blip2/model.safetensors"
  ["weights/icon_caption_blip2/config.json"]="$BASE_URL/icon_caption_blip2/config.json"
)

# Download each file into its specified directory
for file_path in "${!model_files[@]}"; do
  wget -O "$file_path" "${model_files[$file_path]}"
done

echo "All required model and configuration files downloaded and organised."

# Run the conversion script if necessary files are present
if [ -f "weights/icon_detect/model.safetensors" ] && [ -f "weights/icon_detect/model.yaml" ]; then
  python weights/convert_safetensor_to_pt.py
  echo "Conversion to best.pt completed."
else
  echo "Error: Required files for conversion not found."
fi