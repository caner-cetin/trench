# Trench

Trench is a local desktop app for finding objects in an image, isolating them on transparency, and upscaling the result.

![Trench object extraction workflow](./static/example/all.gif)

## Features

- Automatically discovers distinct objects and regions in an image.
- Lets you visually select masks from a thumbnail gallery.
- Exports selected objects with a transparent background.
- Upscales exports 4× with Real-ESRGAN.
- Processes images locally in a desktop app—your images are not sent to a hosted service.

## Why I built this

I made Trench for my brother, who wanted to isolate small details from images without having to learn image-editing software.

## Installation and first run

Trench requires Python 3.12 and [uv](https://docs.astral.sh/uv/). Create an environment, then install the desktop and imaging dependencies with bounded version ranges:

```bash
uv venv --python 3.12

uv pip install --python .venv/bin/python \
  "pyside6>=6.8,<7" \
  "opencv-python>=4.10,<5" \
  "requests>=2.32,<3"

uv pip install --python .venv/bin/python \
  "torch>=2.5,<3" \
  "torchvision>=0.20,<1" \
  "basicsr>=1.4,<2" \
  "realesrgan>=0.3,<1" \
  "segment-anything>=1.0,<2"
```

Start the app:

```bash
PYTHONPATH=src .venv/bin/python -m trench.app
```

On first use, Trench downloads its model weights into `models/`: Real-ESRGAN is about 64 MB, while Segment Anything is roughly 358 MB for the CPU-friendly ViT-B checkpoint or 2.4 GB for the higher-quality ViT-H checkpoint. A GPU is not required, but it substantially improves segmentation and upscaling speed; ViT-H is best suited to CUDA-capable hardware. Download the models once over a reliable connection before processing large images.

## Usage

1. Select an image in the desktop app.
2. Process it to generate object masks.
3. Choose an object from the visual thumbnails or with the selection controls.
4. Upscale it 4× and/or export the transparent PNG.

### Example: selecting and upscaling an apartment

Choose the apartment mask from the detected objects:

![Apartment selected from the object masks](./static/example/apt.png)

Then upscale the isolated object and export it with transparency:

![Upscaled apartment extraction](./static/example/extracted.png)

## Architecture

Trench keeps the interface responsive while it processes images locally:

- **PySide6** provides the desktop UI and mask-selection gallery.
- **Background workers** run long-running segmentation and upscaling work off the UI thread.
- **Segment Anything (SAM)** generates candidate object masks.
- **Real-ESRGAN** performs 4× image upscaling.
- **OpenCV** handles image conversion, compositing, masks, and export preparation.

## Limitations

- Model downloads are large, especially the high-quality SAM checkpoint.
- Results depend on the source image and the masks generated for it.
- CPU-only processing can be slow, particularly for segmentation and large images.

## Development

Install the repository’s development tools:

```bash
uv sync --group dev
```

Run the checks:

```bash
uv run ruff format --check .
uv run ruff check .
uv run basedpyright
uv run python -m unittest discover -v
```

## License

Trench is released under the [MIT License](./LICENSE).

## Acknowledgements

Trench is built on [Segment Anything](https://github.com/facebookresearch/segment-anything) and [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN). Thanks to their authors and contributors.
