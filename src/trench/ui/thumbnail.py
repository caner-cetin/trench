from typing import final, override

import cv2
from cv2.typing import MatLike
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QImage, QMouseEvent, QPixmap
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget


@final
class ThumbnailWidget(QFrame):
    clicked = Signal(int)

    def __init__(
        self, image: MatLike, index: int, parent: QWidget | None = None
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
