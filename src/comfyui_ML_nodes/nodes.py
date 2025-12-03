import os
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from datetime import datetime
import subprocess
import json
import tempfile
import torch
import cv2


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


class SaveVideoNoMetadata:
    """
    Save video without workflow metadata

    This node converts image sequences to video files without any metadata.
    Supports common video formats: mp4, webm, avi, mov, gif
    """

    def __init__(self):
        self.output_dir = None
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Input image sequence to save as video"}),
                "output_path": ("STRING", {
                    "default": "output",
                    "multiline": False,
                    "tooltip": "Output directory path"
                }),
                "naming_mode": (["prefix_number", "custom"], {
                    "default": "prefix_number",
                    "tooltip": "Naming mode: prefix+number or custom name"
                }),
                "fps": ("INT", {
                    "default": 30,
                    "min": 1,
                    "max": 120,
                    "step": 1,
                    "tooltip": "Frames per second"
                }),
                "format": (["mp4", "webm", "avi", "mov", "gif"], {
                    "default": "mp4",
                    "tooltip": "Output video format"
                }),
            },
            "optional": {
                "filename_prefix": ("STRING", {
                    "default": "video",
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
                "quality": (["high", "medium", "low"], {
                    "default": "high",
                    "tooltip": "Video quality preset"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save_video"
    OUTPUT_NODE = True
    CATEGORY = "video/save"
    DESCRIPTION = "Save video without metadata (workflow data removed)"

    def save_video(self, images, output_path="output", naming_mode="prefix_number",
                   fps=30, format="mp4", filename_prefix="video", custom_filename="",
                   start_number=1, add_timestamp="disable", quality="high"):
        """
        Save image sequence as video without metadata
        """
        # Create output directory
        if not os.path.isabs(output_path):
            output_path = os.path.abspath(output_path)

        os.makedirs(output_path, exist_ok=True)

        # Generate filename
        timestamp_str = ""
        if add_timestamp == "enable":
            timestamp_str = f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if naming_mode == "custom" and custom_filename.strip():
            filename = f"{custom_filename}{timestamp_str}.{format}"
        else:
            counter = start_number
            filename = f"{filename_prefix}_{counter:05d}{timestamp_str}.{format}"

        filepath = os.path.join(output_path, filename)

        # Convert images to numpy arrays
        frames = []
        for image in images:
            i = 255. * image.cpu().numpy()
            img_array = np.clip(i, 0, 255).astype(np.uint8)
            frames.append(img_array)

        # Create temporary directory for frames
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save frames as temporary images
            for idx, frame in enumerate(frames):
                frame_path = os.path.join(temp_dir, f"frame_{idx:06d}.png")
                Image.fromarray(frame).save(frame_path)

            # Quality settings
            quality_map = {
                "high": {"crf": 18, "preset": "slow"},
                "medium": {"crf": 23, "preset": "medium"},
                "low": {"crf": 28, "preset": "fast"}
            }
            q = quality_map[quality]

            # Build ffmpeg command based on format
            if format == "gif":
                # GIF conversion
                cmd = [
                    "ffmpeg", "-y",
                    "-framerate", str(fps),
                    "-i", os.path.join(temp_dir, "frame_%06d.png"),
                    "-vf", "fps=15,scale=512:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
                    "-loop", "0",
                    filepath
                ]
            elif format == "webm":
                # WebM conversion
                cmd = [
                    "ffmpeg", "-y",
                    "-framerate", str(fps),
                    "-i", os.path.join(temp_dir, "frame_%06d.png"),
                    "-c:v", "libvpx-vp9",
                    "-crf", str(q["crf"]),
                    "-b:v", "0",
                    "-map_metadata", "-1",  # Remove all metadata
                    filepath
                ]
            else:
                # MP4, AVI, MOV conversion
                codec = "libx264" if format in ["mp4", "mov"] else "mpeg4"
                cmd = [
                    "ffmpeg", "-y",
                    "-framerate", str(fps),
                    "-i", os.path.join(temp_dir, "frame_%06d.png"),
                    "-c:v", codec,
                    "-crf", str(q["crf"]),
                    "-preset", q["preset"],
                    "-pix_fmt", "yuv420p",
                    "-map_metadata", "-1",  # Remove all metadata
                    filepath
                ]

            # Execute ffmpeg
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                print(f"Saved video: {filepath}")
            except subprocess.CalledProcessError as e:
                error_msg = f"FFmpeg error: {e.stderr}"
                print(error_msg)
                return (f"Error: {error_msg}",)
            except FileNotFoundError:
                error_msg = "FFmpeg not found. Please install ffmpeg."
                print(error_msg)
                return (f"Error: {error_msg}",)

        result = f"Saved video to: {filepath}"
        return (result,)


class MLFrameRateResampler:
    """
    Resample image sequences from 25fps to 16fps using ffmpeg

    This node takes an image sequence (typically 25 frames) and resamples it to 16 frames
    while maintaining the original resolution. It uses ffmpeg for high-quality temporal resampling.
    """

    def __init__(self):
        self.type = "processing"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Input image sequence (typically 25 frames at 25fps)"}),
                "input_fps": ("INT", {
                    "default": 25,
                    "min": 1,
                    "max": 120,
                    "step": 1,
                    "tooltip": "Input frame rate (fps)"
                }),
                "output_fps": ("INT", {
                    "default": 16,
                    "min": 1,
                    "max": 120,
                    "step": 1,
                    "tooltip": "Output frame rate (fps)"
                }),
                "interpolation_method": (["blend", "minterpolate", "framestep"], {
                    "default": "blend",
                    "tooltip": "Interpolation method for frame resampling"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("resampled_images",)
    FUNCTION = "resample_frames"
    CATEGORY = "image/processing"
    DESCRIPTION = "Resample image sequence from one frame rate to another (e.g., 25fps to 16fps)"

    def resample_frames(self, images, input_fps=25, output_fps=16, interpolation_method="blend"):
        """
        Resample image sequence to different frame rate

        Args:
            images: Input images tensor [N, H, W, C]
            input_fps: Input frame rate
            output_fps: Output frame rate
            interpolation_method: Method for frame interpolation

        Returns:
            Tuple containing resampled images tensor
        """
        # Check if ffmpeg is available
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Error: FFmpeg not found. Please install ffmpeg.")
            return (images,)  # Return original images if ffmpeg not available

        # Get image dimensions
        num_frames = len(images)
        if num_frames == 0:
            return (images,)

        # Calculate expected output frame count
        expected_output_frames = int(num_frames * output_fps / input_fps)

        # Convert images to numpy arrays
        frames = []
        for image in images:
            i = 255. * image.cpu().numpy()
            img_array = np.clip(i, 0, 255).astype(np.uint8)
            frames.append(img_array)

        # Create temporary directories for input and output
        with tempfile.TemporaryDirectory() as input_dir, \
             tempfile.TemporaryDirectory() as output_dir:

            # Save input frames as temporary images
            for idx, frame in enumerate(frames):
                frame_path = os.path.join(input_dir, f"frame_{idx:06d}.png")
                Image.fromarray(frame).save(frame_path)

            # Build ffmpeg command based on interpolation method
            input_pattern = os.path.join(input_dir, "frame_%06d.png")
            output_pattern = os.path.join(output_dir, "frame_%06d.png")

            if interpolation_method == "minterpolate":
                # Use minterpolate filter for motion-compensated interpolation
                cmd = [
                    "ffmpeg", "-y",
                    "-framerate", str(input_fps),
                    "-i", input_pattern,
                    "-vf", f"minterpolate='fps={output_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1'",
                    "-pix_fmt", "rgb24",
                    output_pattern
                ]
            elif interpolation_method == "framestep":
                # Simple frame selection (no blending)
                fps_filter = f"fps={output_fps}"
                cmd = [
                    "ffmpeg", "-y",
                    "-framerate", str(input_fps),
                    "-i", input_pattern,
                    "-vf", fps_filter,
                    "-vsync", "0",
                    "-pix_fmt", "rgb24",
                    output_pattern
                ]
            else:  # blend (default)
                # Use fps filter with blend for smooth resampling
                fps_filter = f"fps={output_fps}"
                cmd = [
                    "ffmpeg", "-y",
                    "-framerate", str(input_fps),
                    "-i", input_pattern,
                    "-vf", fps_filter,
                    "-pix_fmt", "rgb24",
                    output_pattern
                ]

            # Execute ffmpeg
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                print(f"Resampled {num_frames} frames at {input_fps}fps to {output_fps}fps using {interpolation_method} method")
            except subprocess.CalledProcessError as e:
                error_msg = f"FFmpeg error during resampling: {e.stderr}"
                print(error_msg)
                return (images,)  # Return original images on error

            # Load resampled frames
            output_frames = []
            output_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')])

            for frame_file in output_files:
                frame_path = os.path.join(output_dir, frame_file)
                img = Image.open(frame_path)
                img_array = np.array(img).astype(np.float32) / 255.0
                output_frames.append(img_array)

            if len(output_frames) == 0:
                print("Error: No output frames generated")
                return (images,)

            # Convert back to torch tensor
            output_tensor = torch.from_numpy(np.stack(output_frames))

            print(f"Successfully resampled from {num_frames} frames to {len(output_frames)} frames")
            return (output_tensor,)


class MLFrameRateResampler_GPU:
    """
    GPU-accelerated frame rate resampling using CUDA/NVENC

    This node uses GPU acceleration for faster frame rate conversion.
    Requires NVIDIA GPU with CUDA support and ffmpeg compiled with NVENC.
    Falls back to CPU processing if GPU is unavailable.
    """

    def __init__(self):
        self.type = "processing"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Input image sequence (typically 25 frames at 25fps)"}),
                "input_fps": ("INT", {
                    "default": 25,
                    "min": 1,
                    "max": 120,
                    "step": 1,
                    "tooltip": "Input frame rate (fps)"
                }),
                "output_fps": ("INT", {
                    "default": 16,
                    "min": 1,
                    "max": 120,
                    "step": 1,
                    "tooltip": "Output frame rate (fps)"
                }),
                "interpolation_method": (["blend", "minterpolate", "framestep"], {
                    "default": "blend",
                    "tooltip": "Interpolation method for frame resampling"
                }),
                "gpu_device": (["auto", "cuda:0", "cuda:1", "cpu"], {
                    "default": "auto",
                    "tooltip": "GPU device selection (auto will try CUDA first, then fall back to CPU)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("resampled_images",)
    FUNCTION = "resample_frames_gpu"
    CATEGORY = "image/processing"
    DESCRIPTION = "GPU-accelerated frame rate resampling (25fps to 16fps) using CUDA/NVENC"

    def resample_frames_gpu(self, images, input_fps=25, output_fps=16,
                           interpolation_method="blend", gpu_device="auto"):
        """
        GPU-accelerated frame rate resampling

        Args:
            images: Input images tensor [N, H, W, C]
            input_fps: Input frame rate
            output_fps: Output frame rate
            interpolation_method: Method for frame interpolation
            gpu_device: GPU device to use

        Returns:
            Tuple containing resampled images tensor
        """
        # Check if ffmpeg is available
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Error: FFmpeg not found. Please install ffmpeg.")
            return (images,)

        # Determine hardware acceleration method
        use_gpu = False
        hwaccel_method = None

        if gpu_device != "cpu":
            # Check for CUDA support
            try:
                result = subprocess.run(["ffmpeg", "-hwaccels"],
                                      capture_output=True, text=True, check=True)
                available_hwaccels = result.stdout.lower()

                if "cuda" in available_hwaccels and gpu_device in ["auto", "cuda:0", "cuda:1"]:
                    use_gpu = True
                    hwaccel_method = "cuda"
                    print(f"Using CUDA GPU acceleration on device: {gpu_device}")
                else:
                    print("CUDA not available, falling back to CPU")
            except subprocess.CalledProcessError:
                print("Could not detect hardware acceleration, using CPU")

        # Get image dimensions
        num_frames = len(images)
        if num_frames == 0:
            return (images,)

        # Convert images to numpy arrays
        frames = []
        for image in images:
            i = 255. * image.cpu().numpy()
            img_array = np.clip(i, 0, 255).astype(np.uint8)
            frames.append(img_array)

        # Create temporary directories for input and output
        with tempfile.TemporaryDirectory() as input_dir, \
             tempfile.TemporaryDirectory() as output_dir:

            # Save input frames as temporary images
            for idx, frame in enumerate(frames):
                frame_path = os.path.join(input_dir, f"frame_{idx:06d}.png")
                Image.fromarray(frame).save(frame_path)

            # Build ffmpeg command
            input_pattern = os.path.join(input_dir, "frame_%06d.png")
            output_pattern = os.path.join(output_dir, "frame_%06d.png")

            # Base command
            cmd = ["ffmpeg", "-y"]

            # Add GPU acceleration if available
            if use_gpu and hwaccel_method == "cuda":
                # Extract GPU index for CUDA
                gpu_id = "0"
                if ":" in gpu_device:
                    gpu_id = gpu_device.split(":")[1]

                # CUDA hardware acceleration
                cmd.extend([
                    "-hwaccel", "cuda",
                    "-hwaccel_device", gpu_id,
                    "-hwaccel_output_format", "cuda"
                ])

            # Input settings
            cmd.extend([
                "-framerate", str(input_fps),
                "-i", input_pattern
            ])

            # Build filter chain based on interpolation method and GPU usage
            if use_gpu and hwaccel_method == "cuda":
                # GPU-accelerated filters
                if interpolation_method == "minterpolate":
                    # For minterpolate, we need to download from GPU, process, then upload
                    filter_chain = f"hwdownload,format=nv12,minterpolate='fps={output_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1'"
                elif interpolation_method == "framestep":
                    filter_chain = f"hwdownload,format=nv12,fps={output_fps}"
                else:  # blend
                    # Use scale_cuda for GPU processing
                    filter_chain = f"hwdownload,format=nv12,fps={output_fps}"

                cmd.extend(["-vf", filter_chain])
            else:
                # CPU filters (same as original)
                if interpolation_method == "minterpolate":
                    cmd.extend([
                        "-vf", f"minterpolate='fps={output_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1'"
                    ])
                elif interpolation_method == "framestep":
                    cmd.extend(["-vf", f"fps={output_fps}", "-vsync", "0"])
                else:  # blend
                    cmd.extend(["-vf", f"fps={output_fps}"])

            # Output settings
            cmd.extend([
                "-pix_fmt", "rgb24",
                output_pattern
            ])

            # Execute ffmpeg
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                device_info = f"GPU ({hwaccel_method})" if use_gpu else "CPU"
                print(f"Resampled {num_frames} frames at {input_fps}fps to {output_fps}fps "
                      f"using {interpolation_method} method on {device_info}")
            except subprocess.CalledProcessError as e:
                error_msg = f"FFmpeg error during GPU resampling: {e.stderr}"
                print(error_msg)
                print("Falling back to CPU processing...")

                # Retry with CPU if GPU failed
                if use_gpu:
                    return self._fallback_cpu_resample(images, input_fps, output_fps,
                                                       interpolation_method, input_dir, output_dir)
                return (images,)

            # Load resampled frames
            output_frames = []
            output_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')])

            for frame_file in output_files:
                frame_path = os.path.join(output_dir, frame_file)
                img = Image.open(frame_path)
                img_array = np.array(img).astype(np.float32) / 255.0
                output_frames.append(img_array)

            if len(output_frames) == 0:
                print("Error: No output frames generated")
                return (images,)

            # Convert back to torch tensor
            output_tensor = torch.from_numpy(np.stack(output_frames))

            print(f"Successfully resampled from {num_frames} frames to {len(output_frames)} frames")
            return (output_tensor,)

    def _fallback_cpu_resample(self, images, input_fps, output_fps,
                               interpolation_method, input_dir, output_dir):
        """Fallback to CPU processing if GPU fails"""
        input_pattern = os.path.join(input_dir, "frame_%06d.png")
        output_pattern = os.path.join(output_dir, "frame_%06d.png")

        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(input_fps),
            "-i", input_pattern
        ]

        if interpolation_method == "minterpolate":
            cmd.extend(["-vf", f"minterpolate='fps={output_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1'"])
        elif interpolation_method == "framestep":
            cmd.extend(["-vf", f"fps={output_fps}", "-vsync", "0"])
        else:
            cmd.extend(["-vf", f"fps={output_fps}"])

        cmd.extend(["-pix_fmt", "rgb24", output_pattern])

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("CPU fallback successful")
        except subprocess.CalledProcessError as e:
            print(f"CPU fallback also failed: {e.stderr}")
            return (images,)

        # Load frames (same as GPU path)
        output_frames = []
        output_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')])

        for frame_file in output_files:
            frame_path = os.path.join(output_dir, frame_file)
            img = Image.open(frame_path)
            img_array = np.array(img).astype(np.float32) / 255.0
            output_frames.append(img_array)

        if len(output_frames) > 0:
            return (torch.from_numpy(np.stack(output_frames)),)
        return (images,)


class MLVideoRateConverter:
    """
    Resample image sequences with custom input FPS

    This node is similar to MLFrameRateResampler but allows you to specify the original
    frame rate as an input parameter. Useful when the source frame rate is variable or
    comes from other nodes.
    """

    def __init__(self):
        self.type = "processing"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Input image sequence"}),
                "input_fps": ("INT,FLOAT", {
                    "default": 25,
                    "forceInput": True,
                    "tooltip": "Original frame rate (connect from other node)"
                }),
                "output_fps": ("INT", {
                    "default": 16,
                    "min": 1,
                    "max": 120,
                    "step": 1,
                    "tooltip": "Target output frame rate"
                }),
                "interpolation_method": (["blend", "minterpolate", "framestep"], {
                    "default": "blend",
                    "tooltip": "Interpolation method for frame resampling"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("resampled_images",)
    FUNCTION = "resample_with_input_fps"
    CATEGORY = "image/processing"
    DESCRIPTION = "Resample image sequence with custom input FPS (e.g., 25fps input → 16fps output)"

    def resample_with_input_fps(self, images, input_fps=25.0, output_fps=16, interpolation_method="blend"):
        """
        Resample image sequence with specified input FPS

        Args:
            images: Input images tensor [N, H, W, C]
            input_fps: Input frame rate (can be float)
            output_fps: Output frame rate
            interpolation_method: Method for frame interpolation

        Returns:
            Tuple containing resampled images tensor
        """
        # Check if ffmpeg is available
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Error: FFmpeg not found. Please install ffmpeg.")
            return (images,)

        # Get image dimensions
        num_frames = len(images)
        if num_frames == 0:
            return (images,)

        # Convert images to numpy arrays
        frames = []
        for image in images:
            i = 255. * image.cpu().numpy()
            img_array = np.clip(i, 0, 255).astype(np.uint8)
            frames.append(img_array)

        # Create temporary directories for input and output
        with tempfile.TemporaryDirectory() as input_dir, \
             tempfile.TemporaryDirectory() as output_dir:

            # Save input frames as temporary images
            for idx, frame in enumerate(frames):
                frame_path = os.path.join(input_dir, f"frame_{idx:06d}.png")
                Image.fromarray(frame).save(frame_path)

            # Build ffmpeg command based on interpolation method
            input_pattern = os.path.join(input_dir, "frame_%06d.png")
            output_pattern = os.path.join(output_dir, "frame_%06d.png")

            if interpolation_method == "minterpolate":
                # Use minterpolate filter for motion-compensated interpolation
                cmd = [
                    "ffmpeg", "-y",
                    "-framerate", str(input_fps),
                    "-i", input_pattern,
                    "-vf", f"minterpolate='fps={output_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1'",
                    "-pix_fmt", "rgb24",
                    output_pattern
                ]
            elif interpolation_method == "framestep":
                # Simple frame selection (no blending)
                fps_filter = f"fps={output_fps}"
                cmd = [
                    "ffmpeg", "-y",
                    "-framerate", str(input_fps),
                    "-i", input_pattern,
                    "-vf", fps_filter,
                    "-vsync", "0",
                    "-pix_fmt", "rgb24",
                    output_pattern
                ]
            else:  # blend (default)
                # Use fps filter with blend for smooth resampling
                fps_filter = f"fps={output_fps}"
                cmd = [
                    "ffmpeg", "-y",
                    "-framerate", str(input_fps),
                    "-i", input_pattern,
                    "-vf", fps_filter,
                    "-pix_fmt", "rgb24",
                    output_pattern
                ]

            # Execute ffmpeg
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                print(f"Resampled {num_frames} frames at {input_fps}fps to {output_fps}fps using {interpolation_method} method")
            except subprocess.CalledProcessError as e:
                error_msg = f"FFmpeg error during resampling: {e.stderr}"
                print(error_msg)
                return (images,)

            # Load resampled frames
            output_frames = []
            output_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')])

            for frame_file in output_files:
                frame_path = os.path.join(output_dir, frame_file)
                img = Image.open(frame_path)
                img_array = np.array(img).astype(np.float32) / 255.0
                output_frames.append(img_array)

            if len(output_frames) == 0:
                print("Error: No output frames generated")
                return (images,)

            # Convert back to torch tensor
            output_tensor = torch.from_numpy(np.stack(output_frames))

            print(f"Successfully resampled from {num_frames} frames to {len(output_frames)} frames")
            return (output_tensor,)


class RemovePoseHead:
    """
    Remove head/face region from OpenPose skeleton images (Full Model support)

    Supports all OpenPose formats:
    - OpenPose BODY_25 (25 body keypoints)
    - OpenPose Full Model (BODY_25 + FACE_70 + HAND_21×2)

    Removes from skeleton:
    - BODY_25: Points 0 (nose), 15-18 (eyes, ears)
    - FACE_70: All 70 facial keypoints (dense point cluster)

    Preserves:
    - BODY_25: Point 1 (neck) and below (2-14, 19-24)
    - HAND_21×2: All hand keypoints

    The node intelligently detects the dense FACE_70 region in Full Model
    and precisely removes everything above the neck keypoint (point 1).

    Supports both automatic detection and manual control modes.
    """

    def __init__(self):
        self.type = "processing"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Input OpenPose skeleton images"}),
                "detection_mode": (["auto_detect_neck", "manual_percentage", "manual_pixels"], {
                    "default": "auto_detect_neck",
                    "tooltip": "Method to determine head removal area"
                }),
            },
            "optional": {
                "neck_offset": ("INT", {
                    "default": 20,
                    "min": -100,
                    "max": 200,
                    "step": 5,
                    "tooltip": "Offset in pixels from detected neck position (positive = remove more, negative = keep more). Only used in auto_detect_neck mode."
                }),
                "manual_percentage": ("FLOAT", {
                    "default": 18.0,
                    "min": 0.0,
                    "max": 50.0,
                    "step": 0.5,
                    "tooltip": "Percentage of image height to remove from top (manual_percentage mode)"
                }),
                "manual_pixels": ("INT", {
                    "default": 150,
                    "min": 0,
                    "max": 1000,
                    "step": 1,
                    "tooltip": "Number of pixels to remove from top (manual_pixels mode)"
                }),
                "detection_threshold": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Brightness threshold for skeleton detection (lower = more sensitive)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images_no_head",)
    FUNCTION = "remove_head"
    CATEGORY = "image/processing"
    DESCRIPTION = "Remove head/face region from OpenPose skeleton (supports Full Model with FACE_70). Auto-detects neck or use manual control."

    def _detect_face_region(self, img_array, threshold=0.1):
        """
        Detect the face region (FACE_70 dense points cluster) in OpenPose Full Model

        OpenPose Full Model contains:
        - BODY_25: 25 body keypoints (colored lines and points)
        - FACE_70: 70 facial keypoints (dense white point cluster)
        - HAND_21×2: Hand keypoints

        We need to identify the FACE_70 region and the body neck keypoint.

        Args:
            img_array: Image array [H, W, C]
            threshold: Brightness threshold for detection

        Returns:
            Y coordinate below which to keep the skeleton (neck position)
        """
        height, width, channels = img_array.shape

        # Convert to grayscale
        if channels >= 3:
            gray = np.max(img_array, axis=2)
        else:
            gray = img_array[:, :, 0]

        # Convert to uint8 for OpenCV
        gray_uint8 = (gray * 255).astype(np.uint8)

        # Threshold to get skeleton pixels
        _, binary = cv2.threshold(gray_uint8, int(threshold * 255), 255, cv2.THRESH_BINARY)

        # Find all contours (to identify dense face region)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Analyze pixel density to find face region
        # Face region has much higher point density than body
        search_height = int(height * 0.4)  # Search in top 40%

        # Calculate point density for each horizontal strip
        strip_height = 5
        densities = []

        for y in range(0, search_height, strip_height):
            strip = binary[y:y+strip_height, :]
            density = np.sum(strip > 0) / (strip.shape[0] * strip.shape[1])
            densities.append((y, density))

        if not densities:
            return int(height * 0.15)

        # Find the region with highest density (face region with 70 points)
        max_density_y = max(densities, key=lambda x: x[1])[0]

        # Face region is the high-density cluster
        # Find where density drops significantly (transition from face to neck/body)
        face_end_y = max_density_y

        # Smooth density array
        density_values = [d[1] for d in densities]
        if len(density_values) > 5:
            smoothed = np.convolve(density_values, np.ones(3)/3, mode='same')

            # Find significant density drop (end of face region)
            max_density = max(smoothed)
            for i in range(len(smoothed)):
                if smoothed[i] > max_density * 0.5:  # High density (face region)
                    face_end_y = densities[i][0]
                elif face_end_y > 0 and smoothed[i] < max_density * 0.2:
                    # Significant drop after face region - this is below face
                    break

        return face_end_y + strip_height * 3  # Add margin to ensure face is removed

    def _detect_keypoints(self, img_array, threshold=0.1):
        """
        Detect OpenPose keypoint positions in the skeleton image

        Args:
            img_array: Image array [H, W, C]
            threshold: Brightness threshold for detection

        Returns:
            List of (x, y) keypoint positions
        """
        height, width, channels = img_array.shape

        # Convert to grayscale
        if channels >= 3:
            gray = np.max(img_array, axis=2)
        else:
            gray = img_array[:, :, 0]

        # Convert to uint8 for OpenCV
        gray_uint8 = (gray * 255).astype(np.uint8)

        # Threshold to get skeleton pixels
        _, binary = cv2.threshold(gray_uint8, int(threshold * 255), 255, cv2.THRESH_BINARY)

        # Use blob detection to find keypoints (circles)
        params = cv2.SimpleBlobDetector_Params()

        # Filter by area - adjust for both body points and small face points
        params.filterByArea = True
        params.minArea = 5  # Smaller to catch face points
        params.maxArea = 500

        # Relax filters to catch more points
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(binary)

        # Extract positions
        positions = [(int(kp.pt[0]), int(kp.pt[1])) for kp in keypoints]

        return positions

    def _detect_neck_position(self, img_array, threshold=0.1):
        """
        Detect the neck position by finding OpenPose keypoints

        OpenPose Full Model format:
        - FACE_70: 70 facial keypoints (dense white point cluster) - to be removed
        - BODY_25 points 0,15-18: nose, eyes, ears - to be removed
        - BODY_25 point 1: Neck - this is the cutoff point (keep this)
        - BODY_25 points 2,5: Shoulders - below the neck (keep)

        Strategy:
        1. For OpenPose Full Model: Detect the dense FACE_70 region
        2. Find where the face cluster ends (neck transition)
        3. Fallback: Use keypoint detection for body parts

        Args:
            img_array: Image array [H, W, C]
            threshold: Brightness threshold for skeleton detection

        Returns:
            Y coordinate of neck position (pixels from top)
        """
        height, width, channels = img_array.shape

        # Method 1: Try to detect dense face region (FACE_70 cluster)
        # This is the most reliable for OpenPose Full Model
        try:
            face_end_y = self._detect_face_region(img_array, threshold)
            if face_end_y > 0 and face_end_y < height * 0.5:
                # Valid face region detected
                return face_end_y
        except Exception as e:
            print(f"Face region detection failed: {e}")

        # Method 2: Try keypoint-based detection
        try:
            keypoints = self._detect_keypoints(img_array, threshold)

            if len(keypoints) >= 3:
                # Sort keypoints by Y coordinate (top to bottom)
                keypoints_sorted = sorted(keypoints, key=lambda p: p[1])

                # Find horizontally centered points in the upper portion
                center_x = width / 2
                upper_points = [p for p in keypoints_sorted if p[1] < height * 0.4]

                if len(upper_points) >= 2:
                    # Find points near the horizontal center (likely face/neck points)
                    center_points = []
                    for p in upper_points:
                        # Check if point is in center region (middle 50% of width)
                        if abs(p[0] - center_x) < width * 0.25:
                            center_points.append(p)

                    if len(center_points) >= 2:
                        # Sort center points by Y
                        center_points.sort(key=lambda p: p[1])

                        # Look for shoulder points (wider spacing, lower position)
                        shoulder_candidates = [p for p in keypoints_sorted
                                             if p[1] > center_points[-1][1]
                                             and p[1] < height * 0.5]

                        if len(shoulder_candidates) >= 2:
                            # Find the two widest points at similar Y level (shoulders)
                            shoulder_y = min(shoulder_candidates, key=lambda p: p[1])[1]
                            shoulders = [p for p in shoulder_candidates
                                       if abs(p[1] - shoulder_y) < 30]

                            if len(shoulders) >= 2:
                                # Neck is just above the shoulders
                                # Find the last center point above shoulders
                                neck_candidates = [p for p in center_points
                                                 if p[1] < shoulder_y]

                                if neck_candidates:
                                    neck_y = neck_candidates[-1][1]
                                    return neck_y

                        # Fallback: use the last (lowest) center point as neck
                        neck_y = center_points[-1][1]
                        return neck_y

        except Exception as e:
            print(f"Keypoint detection failed: {e}")

        # Method 3: Fallback to density-based detection
        print("Using density-based fallback detection")
        return self._detect_neck_position_density(img_array, threshold)

    def _detect_neck_position_density(self, img_array, threshold=0.1):
        """
        Fallback method: density-based neck detection
        """
        height, width, channels = img_array.shape

        # Convert to grayscale by taking max across channels
        if channels >= 3:
            gray = np.max(img_array, axis=2)
        else:
            gray = img_array[:, :, 0]

        # Create binary mask of skeleton
        skeleton_mask = (gray > threshold).astype(np.uint8)

        # Calculate horizontal density for each row
        row_density = np.sum(skeleton_mask, axis=1)

        # Find the first row with skeleton content
        skeleton_rows = np.where(row_density > 0)[0]
        if len(skeleton_rows) == 0:
            return 0

        top_of_head = skeleton_rows[0]

        # Search in the top 35% of the image for neck
        search_end = min(int(height * 0.35), len(row_density))
        search_region = row_density[top_of_head:search_end]

        if len(search_region) < 10:
            return int(height * 0.15)

        # Smooth the density
        try:
            from scipy.ndimage import gaussian_filter1d
            smoothed = gaussian_filter1d(search_region.astype(float), sigma=2)
        except:
            window = 5
            smoothed = np.convolve(search_region, np.ones(window)/window, mode='same')

        mean_density = np.mean(smoothed[smoothed > 0]) if np.any(smoothed > 0) else 0

        # Look for transition: low density (head/neck) -> high density (shoulders)
        for i in range(len(smoothed) - 5):
            if smoothed[i] < mean_density * 0.5 and smoothed[i+5:i+10].mean() > mean_density * 0.8:
                neck_y = top_of_head + i
                return neck_y

        # Fallback: proportional estimation
        estimated_neck = top_of_head + int(height * 0.12)
        return min(estimated_neck, search_end)

    def remove_head(self, images, detection_mode="auto_detect_neck",
                   neck_offset=20, manual_percentage=18.0, manual_pixels=150,
                   detection_threshold=0.1):
        """
        Remove head region from OpenPose skeleton images

        Args:
            images: Input images tensor [N, H, W, C]
            detection_mode: Detection method (auto_detect_neck, manual_percentage, manual_pixels)
            neck_offset: Offset from detected neck position
            manual_percentage: Percentage for manual mode
            manual_pixels: Pixels for manual mode
            detection_threshold: Threshold for skeleton detection

        Returns:
            Tuple containing processed images tensor
        """
        if len(images) == 0:
            return (images,)

        # Check if scipy is available for better smoothing
        scipy_available = True
        try:
            from scipy.ndimage import gaussian_filter1d
        except ImportError:
            scipy_available = False
            if detection_mode == "auto_detect_neck":
                print("Warning: scipy not available, using simpler neck detection algorithm")

        # Process each image in the batch
        processed_frames = []
        remove_stats = []

        for idx, image in enumerate(images):
            # Convert from tensor to numpy array
            img_array = image.cpu().numpy()

            # Get image dimensions
            height, width, channels = img_array.shape

            # Determine the cutoff line based on detection mode
            if detection_mode == "auto_detect_neck":
                # Automatically detect neck position
                neck_y = self._detect_neck_position(img_array, detection_threshold)
                remove_pixels = neck_y + neck_offset
                remove_pixels = max(0, min(remove_pixels, height))
            elif detection_mode == "manual_percentage":
                remove_pixels = int(height * (manual_percentage / 100.0))
            else:  # manual_pixels
                remove_pixels = min(manual_pixels, height)

            # Create a copy of the image
            processed_img = img_array.copy()

            # Set the top region to black (0.0)
            # In OpenPose images, black background means no skeleton
            # This effectively removes all skeleton lines and points in the head region
            processed_img[0:remove_pixels, :, :] = 0.0

            processed_frames.append(processed_img)
            remove_stats.append(remove_pixels)

        # Convert back to torch tensor
        output_tensor = torch.from_numpy(np.stack(processed_frames)).float()

        # Print summary
        avg_removed = int(np.mean(remove_stats))
        if detection_mode == "auto_detect_neck":
            print(f"Removed head region from {len(processed_frames)} frames using auto-detection "
                  f"(avg: {avg_removed} pixels from top, offset: {neck_offset}px)")
        else:
            print(f"Removed head region from {len(processed_frames)} frames "
                  f"({avg_removed} pixels from top using {detection_mode} mode)")

        return (output_tensor,)


# Export nodes
NODE_CLASS_MAPPINGS = {
    "SaveImageNoMetadata": SaveImageNoMetadata,
    "SaveImageCleanMetadata": SaveImageCleanMetadata,
    "SaveVideoNoMetadata": SaveVideoNoMetadata,
    "MLFrameRateResampler": MLFrameRateResampler,
    "MLFrameRateResampler_GPU": MLFrameRateResampler_GPU,
    "MLVideoRateConverter": MLVideoRateConverter,
    "RemovePoseHead": RemovePoseHead,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveImageNoMetadata": "ML Save Image (No Metadata)",
    "SaveImageCleanMetadata": "ML Save Image (Clean Metadata)",
    "SaveVideoNoMetadata": "ML Save Video (No Metadata)",
    "MLFrameRateResampler": "ML Frame Rate Resampler",
    "MLFrameRateResampler_GPU": "ML Frame Rate Resampler (GPU)",
    "MLVideoRateConverter": "ML Video Rate Converter 🎬",
    "RemovePoseHead": "ML Remove Pose Head",
}
