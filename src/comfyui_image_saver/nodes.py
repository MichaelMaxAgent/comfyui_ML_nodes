import os
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from datetime import datetime
import subprocess
import json
import tempfile
import torch


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


# Export nodes
NODE_CLASS_MAPPINGS = {
    "SaveImageNoMetadata": SaveImageNoMetadata,
    "SaveImageCleanMetadata": SaveImageCleanMetadata,
    "SaveVideoNoMetadata": SaveVideoNoMetadata,
    "MLFrameRateResampler": MLFrameRateResampler,
    "MLFrameRateResampler_GPU": MLFrameRateResampler_GPU,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveImageNoMetadata": "ML Save Image (No Metadata)",
    "SaveImageCleanMetadata": "ML Save Image (Clean Metadata)",
    "SaveVideoNoMetadata": "ML Save Video (No Metadata)",
    "MLFrameRateResampler": "ML Frame Rate Resampler",
    "MLFrameRateResampler_GPU": "ML Frame Rate Resampler (GPU)",
}
