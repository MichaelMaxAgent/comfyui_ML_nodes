import os
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from datetime import datetime


class SaveImageNoMetadata:
    """
    Save images without workflow metadata

    This node allows you to save images with custom paths and filenames,
    and removes all metadata (including ComfyUI workflow data) from the output.
    """

    def __init__(self):
        self.output_dir = None
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        """
        Define input parameters for the node
        """
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Input images to save"}),
                "output_path": ("STRING", {
                    "default": "output",
                    "multiline": False,
                    "tooltip": "Output directory path (relative to ComfyUI root or absolute path)"
                }),
                "naming_mode": (["prefix_number", "custom"], {
                    "default": "prefix_number",
                    "tooltip": "Naming mode: prefix+number or custom name"
                }),
            },
            "optional": {
                "filename_prefix": ("STRING", {
                    "default": "image",
                    "tooltip": "Prefix for the output filename (used in prefix_number mode)"
                }),
                "custom_filename": ("STRING", {
                    "default": "",
                    "tooltip": "Custom filename without extension (used in custom mode)"
                }),
                "start_number": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 999999,
                    "step": 1,
                    "tooltip": "Starting number for sequential filenames (prefix_number mode)"
                }),
                "add_timestamp": (["enable", "disable"], {
                    "default": "disable",
                    "tooltip": "Add timestamp to filename"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "image/save"
    DESCRIPTION = "Save images without metadata (workflow data removed)"

    def save_images(self, images, output_path="output", naming_mode="prefix_number",
                   filename_prefix="image", custom_filename="", start_number=1, add_timestamp="disable"):
        """
        Save images without any metadata

        Args:
            images: Input images tensor
            output_path: Output directory path
            naming_mode: Naming mode (prefix_number or custom)
            filename_prefix: Prefix for output filenames (prefix_number mode)
            custom_filename: Custom filename (custom mode)
            start_number: Starting number for sequential naming
            add_timestamp: Whether to add timestamp to filename

        Returns:
            Tuple containing the path where images were saved
        """
        # Create output directory if it doesn't exist
        if not os.path.isabs(output_path):
            # If relative path, create it in current working directory
            output_path = os.path.abspath(output_path)

        os.makedirs(output_path, exist_ok=True)

        saved_paths = []

        # Process each image in the batch
        for idx, image in enumerate(images):
            # Convert from tensor to numpy array
            # ComfyUI images are in format [H, W, C] with values 0-1
            i = 255. * image.cpu().numpy()
            img_array = np.clip(i, 0, 255).astype(np.uint8)

            # Convert to PIL Image
            img = Image.fromarray(img_array)

            # Generate filename based on mode
            timestamp_str = ""
            if add_timestamp == "enable":
                timestamp_str = f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            if naming_mode == "custom" and custom_filename.strip():
                # Custom mode: use custom filename
                if len(images) > 1:
                    # Multiple images: add index
                    filename = f"{custom_filename}_{idx + 1}{timestamp_str}.png"
                else:
                    # Single image: use custom name directly
                    filename = f"{custom_filename}{timestamp_str}.png"
            else:
                # Prefix+number mode (default)
                counter = start_number + idx
                filename = f"{filename_prefix}_{counter:05d}{timestamp_str}.png"

            filepath = os.path.join(output_path, filename)

            # Save without any metadata
            # By not passing pnginfo parameter, we ensure no metadata is saved
            img.save(filepath, "PNG", compress_level=4)

            saved_paths.append(filepath)
            print(f"Saved: {filepath}")

        # Return the directory path where images were saved
        result = f"Saved {len(saved_paths)} image(s) to: {output_path}"
        return (result,)


class SaveImageCleanMetadata:
    """
    Save images with clean metadata (only basic info, no workflow)

    Similar to SaveImageNoMetadata but allows adding custom metadata
    while still removing workflow data.
    """

    def __init__(self):
        self.output_dir = None
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Input images to save"}),
                "output_path": ("STRING", {
                    "default": "output",
                    "multiline": False,
                    "tooltip": "Output directory path"
                }),
                "naming_mode": (["prefix_number", "custom"], {
                    "default": "prefix_number",
                    "tooltip": "Naming mode: prefix+number or custom name"
                }),
            },
            "optional": {
                "filename_prefix": ("STRING", {
                    "default": "image",
                    "tooltip": "Prefix for the output filename (used in prefix_number mode)"
                }),
                "custom_filename": ("STRING", {
                    "default": "",
                    "tooltip": "Custom filename without extension (used in custom mode)"
                }),
                "start_number": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 999999,
                    "step": 1,
                }),
                "add_timestamp": (["enable", "disable"], {
                    "default": "disable",
                }),
                "custom_metadata": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Custom metadata to add (key=value format, one per line)"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "image/save"
    DESCRIPTION = "Save images with clean custom metadata, workflow data removed"

    def save_images(self, images, output_path="output", naming_mode="prefix_number",
                   filename_prefix="image", custom_filename="", start_number=1,
                   add_timestamp="disable", custom_metadata=""):
        """
        Save images with clean custom metadata
        """
        # Create output directory
        if not os.path.isabs(output_path):
            output_path = os.path.abspath(output_path)

        os.makedirs(output_path, exist_ok=True)

        # Parse custom metadata
        metadata_dict = {}
        if custom_metadata.strip():
            for line in custom_metadata.strip().split('\n'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    metadata_dict[key.strip()] = value.strip()

        saved_paths = []

        # Process each image
        for idx, image in enumerate(images):
            # Convert to PIL Image
            i = 255. * image.cpu().numpy()
            img_array = np.clip(i, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_array)

            # Generate filename based on mode
            timestamp_str = ""
            if add_timestamp == "enable":
                timestamp_str = f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            if naming_mode == "custom" and custom_filename.strip():
                # Custom mode: use custom filename
                if len(images) > 1:
                    # Multiple images: add index
                    filename = f"{custom_filename}_{idx + 1}{timestamp_str}.png"
                else:
                    # Single image: use custom name directly
                    filename = f"{custom_filename}{timestamp_str}.png"
            else:
                # Prefix+number mode (default)
                counter = start_number + idx
                filename = f"{filename_prefix}_{counter:05d}{timestamp_str}.png"

            filepath = os.path.join(output_path, filename)

            # Create clean PNG metadata
            pnginfo = PngInfo()
            for key, value in metadata_dict.items():
                pnginfo.add_text(key, value)

            # Save with only custom metadata (no workflow data)
            img.save(filepath, "PNG", pnginfo=pnginfo, compress_level=4)

            saved_paths.append(filepath)
            print(f"Saved with custom metadata: {filepath}")

        result = f"Saved {len(saved_paths)} image(s) to: {output_path}"
        return (result,)


# Export nodes
NODE_CLASS_MAPPINGS = {
    "SaveImageNoMetadata": SaveImageNoMetadata,
    "SaveImageCleanMetadata": SaveImageCleanMetadata,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveImageNoMetadata": "ML Save Image (No Metadata)",
    "SaveImageCleanMetadata": "ML Save Image (Clean Metadata)",
}
