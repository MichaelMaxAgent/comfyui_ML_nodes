# ComfyUI Image Saver (No Metadata)

ComfyUI自定义节点，用于保存不含工作流元数据的图片。

## 功能特点

- **清除元数据**：保存的图片不包含ComfyUI工作流数据
- **自定义保存路径**：可指定输出目录（相对路径或绝对路径）
- **完全自定义文件名**：支持两种命名模式
  - 前缀+序号模式：`image_00001.png`, `image_00002.png`
  - 完全自定义模式：可以指定任意文件名
- **时间戳选项**：可选择在文件名中添加时间戳
- **批量处理**：支持批量保存多张图片
- **清洁元数据模式**：可选择添加自定义元数据而不包含工作流

## 安装方法

### 方法1：手动安装

1. 将此文件夹复制到ComfyUI的 `custom_nodes` 目录：
   ```bash
   cd /path/to/ComfyUI/custom_nodes/
   git clone <your-repo-url> comfyui_image_saver
   ```

2. 安装依赖（如果需要）：
   ```bash
   cd comfyui_image_saver
   pip install -e .
   ```

3. 重启ComfyUI

### 方法2：使用ComfyUI Manager

1. 打开ComfyUI Manager
2. 搜索 "Image Saver No Metadata"
3. 点击安装

## 使用方法

### ML Save Image (No Metadata) 节点

完全不保存任何元数据的节点。

**输入参数：**
- `images`: 输入图片（IMAGE类型）
- `output_path`: 输出目录路径（默认："output"）
- `naming_mode`: 命名模式（prefix_number / custom）
- `filename_prefix`: 文件名前缀（prefix_number模式下使用，默认："image"）
- `custom_filename`: 自定义文件名（custom模式下使用，不含扩展名）
- `start_number`: 起始序号（prefix_number模式，默认：1）
- `add_timestamp`: 是否添加时间戳（enable/disable）

**输出：**
- `saved_path`: 保存路径信息字符串

**命名模式示例：**

1. **prefix_number 模式（默认）**
   - 文件名前缀设为 "my_image"
   - 起始序号设为 1
   - 输出文件：`my_image_00001.png`, `my_image_00002.png`, ...

2. **custom 模式**
   - 自定义文件名设为 "final_render"
   - 单张图片输出：`final_render.png`
   - 多张图片输出：`final_render_1.png`, `final_render_2.png`, ...

3. **添加时间戳**
   - 启用时间戳后输出：`my_image_00001_20251010_143022.png`

### ML Save Image (Clean Metadata) 节点

保存带有自定义元数据但不含工作流的图片。

**额外参数：**
- `custom_metadata`: 自定义元数据（多行文本，格式：key=value）

**元数据格式示例：**
```
Author=Your Name
Description=My artwork
Date=2025-10-10
```

**注意：** 此节点也支持两种命名模式（prefix_number / custom）

## 文件路径说明

- **相对路径**：相对于ComfyUI运行目录
  - 例如："output" → ComfyUI根目录下的output文件夹
  - 例如："my_images/batch1" → my_images/batch1

- **绝对路径**：完整的文件系统路径
  - Linux/Mac: "/home/user/images"
  - Windows: "C:/Users/user/images"

## 技术实现

### 元数据清除原理

ComfyUI默认会在PNG图片中保存workflow和prompt信息作为PNG元数据。本节点通过以下方式清除：

1. 使用PIL保存图片时不传递任何`pnginfo`参数
2. 确保不复制原有图片的EXIF和其他元数据
3. 生成的PNG仅包含图像数据本身

### 项目结构

```
comfyui_image_saver/
├── __init__.py                      # 包入口
├── pyproject.toml                   # 项目配置
├── README.md                        # 说明文档
└── src/
    └── comfyui_image_saver/
        ├── __init__.py              # 模块初始化
        └── nodes.py                 # 节点实现
```

## 开发

### 运行测试

```bash
pip install -e .[dev]
pytest tests/
```

### 代码检查

```bash
ruff check .
```

## 常见问题

**Q: 为什么需要清除元数据？**
A: ComfyUI默认在图片中保存完整的工作流信息，这会：
- 增加文件大小
- 暴露您的工作流程
- 在某些平台上传时可能包含敏感信息

**Q: 清除元数据会影响图片质量吗？**
A: 不会。元数据和图片的实际像素数据是分开存储的，删除元数据不影响图片质量。

**Q: 相对路径从哪里开始计算？**
A: 从ComfyUI的运行目录开始（通常是ComfyUI的根目录）。

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！

## 节点名称

在ComfyUI中，这两个节点显示为：
- **ML Save Image (No Metadata)** - 完全无元数据保存
- **ML Save Image (Clean Metadata)** - 带自定义元数据保存

"ML"前缀用于与其他节点区分。

## 更新日志

### v0.1.0 (2025-10-10)
- 初始版本
- 实现基本的无元数据保存功能
- 支持两种命名模式：前缀+序号 / 完全自定义
- 支持自定义路径和文件名
- 添加时间戳选项
- 支持自定义清洁元数据模式
- 节点名称添加ML前缀
