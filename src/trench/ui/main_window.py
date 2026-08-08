from pathlib import Path
from typing import final

import cv2
import numpy as np
import torch
from cv2.typing import MatLike
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
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
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from trench.services.export import ExportPathError, prepare_export_path
from trench.services.image_processor import (
    ImageProcessor,
    apply_mask_alpha,
    create_mask_preview,
)
from trench.services.models import (
    REALESRGAN_X4PLUS,
    ModelSpec,
    choose_sam_checkpoint,
    format_bytes,
    is_model_valid,
)
from trench.types import MaskRecord
from trench.ui.thumbnail import ThumbnailWidget
from trench.workers import MaskWorker, ModelDownloadWorker, UpscaleWorker


@final
class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Trench")
        self.setGeometry(100, 100, 800, 600)

        self.output_dir = Path.cwd() / "extracted_objects"
        self.operation_in_progress = False

        # Initialize UI
        self.init_ui()

        self.masks: list[MaskRecord] = []
        self.current_mask_index: int = 0
        self.image: MatLike | None = None
        self.thumbnail_widgets: list[ThumbnailWidget] = []

        # Populated once model downloads finish; both are unusable before then.
        self.processor: ImageProcessor | None = None
        self.mask_generator: SamAutomaticMaskGenerator | None = None

        # Initialize workers as class attributes
        self.worker: MaskWorker | None = None
        self.upscale_worker: UpscaleWorker | None = None
        self.model_download_worker: ModelDownloadWorker | None = None

        self.sam_registry_key, self.sam_checkpoint_spec = choose_sam_checkpoint(
            torch.cuda.is_available()
        )
        self._model_queue: list[ModelSpec] = [REALESRGAN_X4PLUS, self.sam_checkpoint_spec]
        self.set_operation_in_progress(True)
        self._prepare_next_model()

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

        status_layout = QHBoxLayout()
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.status_label, 1)

        self.cancel_download_button = QPushButton("Cancel Download")
        self.cancel_download_button.clicked.connect(self.cancel_model_download)
        self.cancel_download_button.setVisible(False)
        status_layout.addWidget(self.cancel_download_button)
        content_layout.addLayout(status_layout)

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

        export_layout = QHBoxLayout()
        self.export_folder_button = QPushButton("Choose Export Folder…")
        self.export_folder_button.clicked.connect(self.choose_export_folder)
        export_layout.addWidget(self.export_folder_button)

        self.export_location_label = QLabel()
        self.export_location_label.setWordWrap(True)
        export_layout.addWidget(self.export_location_label, 1)
        content_layout.addLayout(export_layout)
        self.update_export_location_label()
        main_layout.addWidget(content_widget)

    def choose_export_folder(self) -> None:
        """Let the user choose where exported objects are saved."""
        selected_directory = QFileDialog.getExistingDirectory(
            self,
            "Choose Export Folder",
            str(self.output_dir),
            QFileDialog.Option.ShowDirsOnly,
        )
        if selected_directory:
            self.output_dir = Path(selected_directory)
            self.update_export_location_label()

    def update_export_location_label(self) -> None:
        """Show the currently selected export destination in the UI."""
        location = self.output_dir.expanduser().resolve()
        self.export_location_label.setText(f"Exports to: {location}")
        self.export_location_label.setToolTip(str(location))

    def set_operation_in_progress(self, in_progress: bool) -> None:
        """Lock interactive controls while a worker is using the current image."""
        self.operation_in_progress = in_progress
        self.progress_bar.setVisible(in_progress)

        if in_progress:
            self.select_button.setEnabled(False)
            self.process_button.setEnabled(False)
            self.prev_button.setEnabled(False)
            self.upscale_button.setEnabled(False)
            self.next_button.setEnabled(False)
            self.export_folder_button.setEnabled(False)
            self.scroll_area.setEnabled(False)
            return

        self.select_button.setEnabled(True)
        self.process_button.setEnabled(self.image is not None)
        self.export_folder_button.setEnabled(True)
        self.scroll_area.setEnabled(True)
        self.upscale_button.setEnabled(
            0 <= self.current_mask_index < len(self.masks)
        )
        self.update_navigation_buttons()

    def on_worker_error(self, message: str) -> None:
        """Restore the UI and surface a worker failure on the main thread."""
        self.set_operation_in_progress(False)
        QMessageBox.critical(self, "Operation Failed", message)

    def _prepare_next_model(self) -> None:
        """Download (or skip, if already valid) the next queued model checkpoint."""
        if not self._model_queue:
            self._on_models_ready()
            return

        spec = self._model_queue[0]
        if is_model_valid(spec):
            self._model_queue.pop(0)
            self._prepare_next_model()
            return

        self.status_label.setText(f"Downloading {spec.name} ({format_bytes(spec.size_bytes)})…")
        self.cancel_download_button.setVisible(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        self.model_download_worker = ModelDownloadWorker(spec)
        self.model_download_worker.progress_signal.connect(self.on_model_download_progress)
        self.model_download_worker.error_signal.connect(self.on_model_download_error)
        self.model_download_worker.finished_signal.connect(self.on_model_download_finished)
        self.model_download_worker.start()

    def on_model_download_progress(self, downloaded: int, total: int) -> None:
        spec = self._model_queue[0]
        if total > 0:
            self.progress_bar.setValue(int(downloaded * 100 / total))
        self.status_label.setText(
            f"Downloading {spec.name}: {format_bytes(downloaded)} / {format_bytes(total)}"
        )

    def on_model_download_error(self, message: str) -> None:
        """A download attempt was exhausted; let the user retry or give up."""
        choice = QMessageBox.critical(
            self,
            "Model Download Failed",
            f"{message}\n\nRetry the download, or cancel model setup?",
            QMessageBox.StandardButton.Retry | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Retry,
        )
        if choice == QMessageBox.StandardButton.Retry:
            self._prepare_next_model()
        else:
            self.cancel_model_download()

    def on_model_download_finished(self, success: bool) -> None:
        """Advance the queue on success; errors/cancellation are handled elsewhere."""
        if not success:
            return
        self._model_queue.pop(0)
        self._prepare_next_model()

    def cancel_model_download(self) -> None:
        """Abort model setup. The app stays open but mask/upscale features are disabled."""
        if self.model_download_worker is not None:
            self.model_download_worker.cancel()
        self._model_queue.clear()
        self.cancel_download_button.setVisible(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText(
            "Model setup cancelled. Restart the app to try downloading again — "
            "Process Image will stay disabled until then."
        )
        self.set_operation_in_progress(False)

    def _on_models_ready(self) -> None:
        self.processor = ImageProcessor(model_path=str(REALESRGAN_X4PLUS.path))
        sam = sam_model_registry[self.sam_registry_key](
            checkpoint=str(self.sam_checkpoint_spec.path)
        )
        self.mask_generator = SamAutomaticMaskGenerator(sam)

        self.cancel_download_button.setVisible(False)
        self.status_label.setText(f"Ready. Using {self.sam_checkpoint_spec.name} — {self.sam_checkpoint_spec.notes}")
        self.set_operation_in_progress(False)

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

        if self.mask_generator is None:
            QMessageBox.warning(
                self,
                "Models Not Ready",
                "Segmentation is unavailable because model setup was cancelled. "
                "Restart the app to download the required models.",
            )
            return

        self.set_operation_in_progress(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.worker = MaskWorker(self.image, self.mask_generator)
        self.worker.finished_signal.connect(self.on_masks_generated)
        self.worker.error_signal.connect(self.on_worker_error)
        self.worker.start()

    def on_masks_generated(self, masks: list[MaskRecord]) -> None:
        """Handle the finished signal from the worker thread"""
        try:
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
        finally:
            self.set_operation_in_progress(False)

    def create_mask_thumbnail(self, index: int) -> ThumbnailWidget:
        """Create a thumbnail for a mask"""
        mask = self.masks[index].segmentation.astype(np.uint8)
        thumbnail_img = create_mask_preview(self.image, mask)  # pyright: ignore[reportArgumentType] mask_generator only runs once self.image is set

        return ThumbnailWidget(thumbnail_img, index)

    def on_thumbnail_clicked(self, index: int) -> None:
        """Handle thumbnail click events"""
        self.show_mask(index)

        for widget in self.thumbnail_widgets:
            widget.set_selected(widget.index == index)

    def show_mask(self, index: int) -> None:
        if 0 <= index < len(self.masks):
            self.current_mask_index = index
            mask = self.masks[index].segmentation.astype(np.uint8)

            preview = create_mask_preview(self.image, mask)  # pyright: ignore[reportArgumentType] mask_generator only runs once self.image is set
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
        self.prev_button.setEnabled(
            not self.operation_in_progress and self.current_mask_index > 0
        )
        self.next_button.setEnabled(
            not self.operation_in_progress
            and self.current_mask_index < len(self.masks) - 1
        )

    def upscale_object(self) -> None:
        """upscale and save the current mask"""
        if self.image is None:
            QMessageBox.warning(
                self,
                "Error",
                "No image selected!",
                button0=QMessageBox.StandardButton.Ok,
                button1=QMessageBox.StandardButton.NoButton,
            )
            return

        mask = self.masks[self.current_mask_index].segmentation.astype(np.uint8) * 255
        rgba = apply_mask_alpha(self.image, mask)

        self.set_operation_in_progress(True)
        self.progress_bar.setRange(0, 0)  # indeterminate progress
        self.upscale_worker = UpscaleWorker(self.processor, rgba)
        self.upscale_worker.finished_signal.connect(self.on_upscale_finished)
        self.upscale_worker.error_signal.connect(self.on_worker_error)
        self.upscale_worker.start()

    def on_upscale_finished(self, upscaled_rgba: MatLike) -> None:
        """Handle the finished signal from the upscale worker thread"""
        self.set_operation_in_progress(False)

        # bgra => rgba for display
        display_rgb = cv2.cvtColor(upscaled_rgba, cv2.COLOR_BGRA2RGBA)
        self.display_image(display_rgb)

        # rgba => bgra for save
        save_bgra = cv2.cvtColor(upscaled_rgba, cv2.COLOR_RGBA2BGRA)

        try:
            output_path = prepare_export_path(self.output_dir, self.current_mask_index)
            saved = cv2.imwrite(str(output_path), save_bgra)
        except ExportPathError as error:
            QMessageBox.critical(self, "Export Failed", str(error))
            return

        if not saved:
            QMessageBox.critical(
                self, "Export Failed", f"Could not save object to {output_path}."
            )
            return

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
