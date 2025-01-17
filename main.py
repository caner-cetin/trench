import os
import sys
from typing import Any, Final, List, Optional, final, override

import cv2
import numpy as np
import requests
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from cv2.typing import MatLike
from PySide6.QtCore import QSize, Qt, QThread, Signal
from PySide6.QtGui import QImage, QMouseEvent, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from realesrgan import RealESRGANer
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from tqdm import tqdm


class ImageProcessor:
    def __init__(
        self, device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> None:
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        model_path: Final[str] = "./models/RealESRGAN_x4plus.pth"
        if not os.path.exists(model_path):
            print("Downloading RealESRGAN model...")
            self.download_file_with_progress(
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                model_path,
            )
        self.device: str = device
        self.upsampler: RealESRGANer = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=model,
            tile=512,
            tile_pad=32,
            pre_pad=0,
            device=self.device,
            half=self.device == "cuda",
        )

    def upscale_image(self, img: MatLike) -> MatLike:
        # rgba => rgb for upscaling
        has_alpha: bool = img.shape[2] == 4
        if has_alpha:
            alpha = img[:, :, 3]
            img_rgb = img[:, :, :3]
        else:
            img_rgb = img
            alpha = None

        output, _ = self.upsampler.enhance(img_rgb, outscale=4)

        # restore alpha channel if it existed
        if alpha is not None:
            alpha_upscaled = cv2.resize(
                alpha,
                (output.shape[1], output.shape[0]),
                interpolation=cv2.INTER_LANCZOS4,
            )
            output_rgba = np.dstack((output, alpha_upscaled))
            return output_rgba

        return output

    def download_file_with_progress(self, url: str, filename: str) -> bool:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            total_size: int = int(response.headers.get("content-length", 0))
            progress = tqdm(
                total=total_size,
                unit="iB",
                unit_scale=True,
                desc=f"Downloading {filename}",
            )

            with open(filename, "wb") as f:
                chunk: bytes
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    progress.update(len(chunk))

            progress.close()
            return True
        return False


@final
class MaskWorker(QThread):
    finished_signal = Signal(list)

    def __init__(
        self, image: MatLike, mask_generator: SamAutomaticMaskGenerator
    ) -> None:
        super().__init__()
        self.image: MatLike = image
        self.mask_generator: SamAutomaticMaskGenerator = mask_generator

    @override
    def run(self) -> None:
        masks: list[dict[str, Any]] = self.mask_generator.generate(self.image)
        significant_masks: List[dict[str, Any]] = [
            mask for mask in masks if mask["area"] > 1000
        ]
        self.finished_signal.emit(significant_masks)


@final
class UpscaleWorker(QThread):
    finished_signal = Signal(np.ndarray)

    def __init__(self, processor: ImageProcessor, rgba: MatLike) -> None:
        super().__init__()
        self.processor: ImageProcessor = processor
        self.rgba = rgba

    @override
    def run(self) -> None:
        upscaled_rgba = self.processor.upscale_image(self.rgba)
        self.finished_signal.emit(upscaled_rgba)


@final
class ThumbnailWidget(QFrame):
    clicked = Signal(int)

    def __init__(
        self, image: MatLike, index: int, parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self.index: int = index
        self.setFrameStyle(QFrame.Shape.Box)
        self.setLineWidth(2)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.thumbnail_label = QLabel()
        self.thumbnail_label.setFixedSize(100, 100)
        self.thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.set_image(image)

        index_label = QLabel(f"Object {index + 1}")
        index_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.thumbnail_label)
        layout.addWidget(index_label)

        self.setStyleSheet(
            """
            ThumbnailWidget {
                border-radius: 5px;
                margin: 2px;
            }
        """
        )

    def set_image(self, image: MatLike) -> None:
        if image.shape[2] == 4:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width, channel = image_rgb.shape
        bytes_per_line = channel * width
        q_img = QImage(
            image_rgb.data.tobytes(),
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGBA8888
            if channel == 4
            else QImage.Format.Format_RGB888,
        )

        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(
            QSize(90, 90),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.thumbnail_label.setPixmap(scaled_pixmap)

    @override
    def mousePressEvent(self, event: QMouseEvent) -> None:
        self.clicked.emit(self.index)
        self.setStyleSheet(
            """
            ThumbnailWidget {
                border: 2px solid #2196F3;
                border-radius: 5px;
                margin: 2px;
            }
        """
        )

    def set_selected(self, selected: bool) -> None:
        if selected:
            self.setStyleSheet(
                """
                ThumbnailWidget {
                    border: 2px solid #2196F3;
                    border-radius: 5px;
                    margin: 2px;
                }
            """
            )
        else:
            self.setStyleSheet("")


@final
class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Trench")
        self.setGeometry(100, 100, 800, 600)

        # Initialize UI
        self.init_ui()

        # Initialize image processor
        self.processor: ImageProcessor = ImageProcessor()
        self.masks: List[dict[str, float] | MatLike] = []
        """
        A list over records for masks. Each record is a dict containing the following keys:
        segmentation (dict(str, any) or np.ndarray): The mask. If
            output_mode='binary_mask', is an array of shape HW. Otherwise, is a dictionary containing the RLE.
        bbox (list(float)): The box around the mask, in XYWH format.
        area (int): The area in pixels of the mask.
        predicted_iou (float): The model's own prediction of the mask's
            quality. This is filtered by the pred_iou_thresh parameter.
        point_coords (list(list(float))): The point coordinates input
            to the model to generate this mask.
        stability_score (float): A measure of the mask's quality. This
            is filtered on using the stability_score_thresh parameter.
        crop_box (list(float)): The crop of the image used to generate
            the mask, given in XYWH format.
        """
        self.current_mask_index: int = 0
        self.image: Optional[MatLike] = None
        self.output_dir: Final[str] = "extracted_objects"
        checkpoint_path: Final[str] = "./models/sam_vit_h_4b8939.pth"
        if not os.path.exists(checkpoint_path):
            self.processor.download_file_with_progress(
                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                checkpoint_path,
            )

        sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
        self.mask_generator = SamAutomaticMaskGenerator(sam)
        self.thumbnail_widgets: List[ThumbnailWidget] = []

        # Initialize workers as class attributes
        self.worker: Optional[MaskWorker] = None
        self.upscale_worker: Optional[UpscaleWorker] = None

    def init_ui(self) -> None:
        """Initialize the user interface"""
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        gallery_widget = QWidget()
        gallery_layout = QVBoxLayout()
        gallery_widget.setLayout(gallery_layout)

        gallery_label = QLabel("Gallery")
        gallery_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        gallery_label.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")
        gallery_layout.addWidget(gallery_label)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.scroll_area.setMinimumWidth(150)
        self.scroll_area.setMaximumWidth(200)

        self.gallery_container = QWidget()
        self.gallery_container_layout = QVBoxLayout()
        self.gallery_container.setLayout(self.gallery_container_layout)
        self.scroll_area.setWidget(self.gallery_container)

        gallery_layout.addWidget(self.scroll_area)
        main_layout.addWidget(gallery_widget)

        content_widget = QWidget()
        content_layout = QVBoxLayout()
        content_widget.setLayout(content_layout)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content_layout.addWidget(self.image_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        content_layout.addWidget(self.progress_bar)

        button_layout = QHBoxLayout()

        self.select_button = QPushButton("Select Image")
        self.select_button.clicked.connect(self.select_image)
        button_layout.addWidget(self.select_button)

        self.process_button = QPushButton("Process Image")
        self.process_button.clicked.connect(self.process_image)
        self.process_button.setEnabled(False)
        button_layout.addWidget(self.process_button)

        button_layout.addStretch()

        self.prev_button = QPushButton("←")
        self.prev_button.clicked.connect(self.show_previous_mask)
        self.prev_button.setEnabled(False)
        button_layout.addWidget(self.prev_button)

        self.upscale_button = QPushButton("Upscale Object")
        self.upscale_button.clicked.connect(self.upscale_object)
        self.upscale_button.setEnabled(False)
        button_layout.addWidget(self.upscale_button)

        self.next_button = QPushButton("→")
        self.next_button.clicked.connect(self.show_next_mask)
        self.next_button.setEnabled(False)
        button_layout.addWidget(self.next_button)

        content_layout.addLayout(button_layout)
        main_layout.addWidget(content_widget)

    def select_image(self) -> None:
        """Open a file dialog to select an image"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.image = cv2.imread(file_path)
            self.display_image(self.image)
            self.process_button.setEnabled(True)
            self.clear_gallery()

    def clear_gallery(self) -> None:
        """Clear all thumbnails from the gallery"""
        for widget in self.thumbnail_widgets:
            self.gallery_container_layout.removeWidget(widget)
            widget.deleteLater()
        self.thumbnail_widgets.clear()

    def create_mask_preview(self, mask: MatLike) -> MatLike:
        """Create a preview image for a mask"""
        try:
            if not self.image:
                QMessageBox.warning(
                    self,
                    "Error",
                    "No image selected!",
                    QMessageBox.StandardButton.Ok,
                    QMessageBox.StandardButton.NoButton,
                )
                return np.zeros_like(mask)
        except ValueError:
            pass
        highlighted = self.image.copy()  # pyright: ignore[reportOptionalMemberAccess] see try except clause
        overlay = np.zeros_like(self.image)
        overlay[mask == 0] = [128, 128, 128]

        alpha = 0.7
        cv2.addWeighted(highlighted, alpha, overlay, 1 - alpha, 0, highlighted)

        rgba = cv2.cvtColor(highlighted, cv2.COLOR_BGR2RGBA)
        alpha_channel = np.where(mask == 1, 255, 128).astype(np.uint8)
        rgba[:, :, 3] = alpha_channel

        return rgba

    def display_image(self, image: MatLike) -> None:
        """Display an image in the QLabel with proper scaling"""
        if image.shape[2] == 4:  # If the image has an alpha channel (RGBA)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        else:  # If the image is RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width, channel = image_rgb.shape
        bytes_per_line: int = channel * width
        q_img = QImage(
            image_rgb.data.tobytes(),
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGBA8888
            if channel == 4
            else QImage.Format.Format_RGB888,
        )

        label_size = self.image_label.size()
        scaled_pixmap = QPixmap.fromImage(q_img).scaled(
            label_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled_pixmap)

    def process_image(self) -> None:
        if self.image is None:
            QMessageBox.warning(
                self,
                "Error",
                "No image selected!",
                QMessageBox.StandardButton.Ok,
                QMessageBox.StandardButton.NoButton,
            )
            return

        # Start the worker thread
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.worker = MaskWorker(self.image, self.mask_generator)
        self.worker.finished_signal.connect(self.on_masks_generated)
        self.worker.start()

    def on_masks_generated(self, masks: list[dict[str, float] | MatLike]) -> None:
        """Handle the finished signal from the worker thread"""
        self.progress_bar.setVisible(False)
        self.masks = masks
        self.current_mask_index = 0

        self.clear_gallery()

        for i, _ in enumerate(self.masks):
            thumbnail = self.create_mask_thumbnail(i)
            self.thumbnail_widgets.append(thumbnail)
            self.gallery_container_layout.addWidget(thumbnail)
            thumbnail.clicked.connect(self.on_thumbnail_clicked)

        self.gallery_container_layout.addStretch()

        self.show_mask(0)

        self.update_navigation_buttons()

    def create_mask_thumbnail(self, index: int) -> ThumbnailWidget:
        """Create a thumbnail for a mask"""
        mask_data = self.masks[index]
        mask = mask_data["segmentation"].astype(np.uint8)

        thumbnail_img = self.create_mask_preview(mask)

        return ThumbnailWidget(thumbnail_img, index)

    def on_thumbnail_clicked(self, index: int) -> None:
        """Handle thumbnail click events"""
        self.show_mask(index)

        for widget in self.thumbnail_widgets:
            widget.set_selected(widget.index == index)

    def show_mask(self, index: int) -> None:
        if 0 <= index < len(self.masks):
            self.current_mask_index = index
            mask_data = self.masks[index]
            mask = mask_data["segmentation"].astype(np.uint8)

            preview = self.create_mask_preview(mask)
            self.display_image(preview)

            self.upscale_button.setEnabled(True)
            self.update_navigation_buttons()

    def show_next_mask(self) -> None:
        if self.current_mask_index < len(self.masks) - 1:
            self.show_mask(self.current_mask_index + 1)

    def show_previous_mask(self) -> None:
        if self.current_mask_index > 0:
            self.show_mask(self.current_mask_index - 1)

    def update_navigation_buttons(self) -> None:
        self.prev_button.setEnabled(self.current_mask_index > 0)
        self.next_button.setEnabled(self.current_mask_index < len(self.masks) - 1)

    def upscale_object(self) -> None:
        """upscale and save the current mask"""
        try:
            if not self.image:
                QMessageBox.warning(
                    self,
                    "Error",
                    "No image selected!",
                    button0=QMessageBox.StandardButton.Ok,
                    button1=QMessageBox.StandardButton.NoButton,
                )
                return
        except ValueError:
            pass
        mask_data = self.masks[self.current_mask_index]
        mask = mask_data["segmentation"].astype(np.uint8) * 255

        rgba = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGBA)  # pyright: ignore[reportCallIssue] see other try/excepts for if not self.image
        rgba[:, :, 3] = mask

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # indeterminate progress
        self.upscale_worker = UpscaleWorker(self.processor, rgba)
        self.upscale_worker.finished_signal.connect(self.on_upscale_finished)
        self.upscale_worker.start()

    def on_upscale_finished(self, upscaled_rgba: MatLike) -> None:
        """Handle the finished signal from the upscale worker thread"""
        self.progress_bar.setVisible(False)

        # bgra => rgba for display
        display_rgb = cv2.cvtColor(upscaled_rgba, cv2.COLOR_BGRA2RGBA)
        self.display_image(display_rgb)

        # rgba => bgra for save
        save_bgra = cv2.cvtColor(upscaled_rgba, cv2.COLOR_RGBA2BGRA)

        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(
            self.output_dir, f"object_{self.current_mask_index}.png"
        )

        cv2.imwrite(output_path, save_bgra)

        QMessageBox.information(self, "Success", f"Object saved as {output_path}")

    def skip_object(self) -> None:
        """skip the current mask"""
        self.current_mask_index += 1
        self.show_next_mask()

    def clear(self) -> None:
        """clear all masks and reset to default state"""
        self.masks = []
        self.current_mask_index = 0
        self.image = None
        self.image_label.clear()
        self.process_button.setEnabled(False)
        self.upscale_button.setEnabled(False)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
