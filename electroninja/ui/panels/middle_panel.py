# middle_panel.py

import os
import logging
import re
from PyQt5.QtWidgets import (
    QFrame, QVBoxLayout, QLabel, QSizePolicy,
    QHBoxLayout, QWidget, QGraphicsOpacityEffect
)
from PyQt5.QtCore import Qt, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QFont, QPixmap
from electroninja.config.settings import Config

logger = logging.getLogger('electroninja')

class MiddlePanel(QFrame):
    """Middle panel for circuit visualization"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = Config()
        self.current_image_path = None
        self.current_iteration = 0
        self.transition_duration = 500  # Animation duration in ms
        self.initUI()
        
    def initUI(self):
        """Initialize the UI components"""
        # Main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(15, 15, 15, 15)
        self.main_layout.setSpacing(15)
        
        # Title
        self.circuit_title = QLabel("Current Circuit", self)
        self.circuit_title.setStyleSheet(
            "font-size: 36px; font-weight: bold; color: white; letter-spacing: 1px; background: transparent;"
        )
        self.circuit_title.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.circuit_title)
        
        # Iteration indicator
        indicator_layout = QHBoxLayout()
        indicator_layout.addStretch()
        
        self.iteration_indicator = QLabel(self)
        self.iteration_indicator.setStyleSheet("""
            background-color: #4B2F4C; 
            color: white; 
            border-radius: 12px; 
            padding: 5px 10px;
            font-weight: bold;
        """)
        self.iteration_indicator.setAlignment(Qt.AlignCenter)
        self.iteration_indicator.hide()
        
        indicator_layout.addWidget(self.iteration_indicator)
        indicator_layout.addStretch()
        self.main_layout.addLayout(indicator_layout)
        
        # Image display container
        self.display_container = QWidget()
        display_layout = QVBoxLayout(self.display_container)
        display_layout.setContentsMargins(20, 20, 20, 20)
        
        h_layout = QHBoxLayout()
        h_layout.setAlignment(Qt.AlignCenter)
        
        # Frame for image display
        self.display_frame = QFrame()
        self.display_frame.setStyleSheet("""
            background-color: #2B2B2B;
            border: 1px dashed #5A5A5A;
            border-radius: 5px;
        """)
        self.display_frame.setMinimumSize(400, 400)
        
        frame_layout = QVBoxLayout(self.display_frame)
        frame_layout.setAlignment(Qt.AlignCenter)
        
        # Placeholder text
        self.circuit_display = QLabel("Circuit Screenshot Placeholder")
        self.circuit_display.setAlignment(Qt.AlignCenter)
        self.circuit_display.setStyleSheet("border: none; color: #AAAAAA;")
        self.circuit_display.setFont(QFont("Segoe UI", 16))
        self.circuit_display.setWordWrap(True)
        
        # Image label
        self.image_label = QLabel(self.display_frame)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: none;")
        
        # Setup fade effect
        self.opacity_effect = QGraphicsOpacityEffect(self.image_label)
        self.opacity_effect.setOpacity(1.0)
        self.image_label.setGraphicsEffect(self.opacity_effect)
        
        self.fade_animation = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_animation.setDuration(self.transition_duration)
        self.fade_animation.setEasingCurve(QEasingCurve.InOutCubic)
        
        frame_layout.addWidget(self.image_label)
        frame_layout.addWidget(self.circuit_display)
        h_layout.addWidget(self.display_frame)
        display_layout.addLayout(h_layout)
        
        self.main_layout.addWidget(self.display_container)
        self.main_layout.addStretch()
        
    def set_circuit_image(self, image_path, iteration=None):
        """
        Update the circuit image with transition animation
        
        Args:
            image_path (str): Path to the image file
            iteration (int, optional): Iteration number, extracted from path if None
        """
        logger.info(f"Setting circuit image: {image_path}")
        
        if not os.path.exists(image_path):
            logger.error(f"Image file does not exist: {image_path}")
            return
            
        logger.info(f"Image file exists and is being processed: {image_path}, file size: {os.path.getsize(image_path)} bytes")
        
        # Extract iteration from path if not provided
        if iteration is None:
            iteration = 0
            if "output" in image_path:
                try:
                    match = re.search(r'output(\d+)', image_path)
                    if match:
                        iteration = int(match.group(1))
                except:
                    pass
        
        # Update iteration display
        self.current_iteration = iteration
        self._update_iteration_indicator(iteration)
        
        # Skip if same image
        if self.current_image_path and os.path.normpath(self.current_image_path) == os.path.normpath(image_path):
            logger.info(f"Skipping duplicate image: {image_path}")
            return
            
        self.current_image_path = image_path
        
        # Load the pixmap first to check if it's valid
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            logger.error(f"Failed to load pixmap from image: {image_path}")
            self._set_placeholder_text("Failed to load image")
            return
            
        logger.info(f"Successfully loaded pixmap: {pixmap.width()}x{pixmap.height()}")
        
        # Hide placeholder text when image is loaded
        self.circuit_display.hide()
        
        # Set directly if no animation needed
        if not pixmap.isNull():
            # Scale to display frame size accounting for padding
            scaled_pixmap = pixmap.scaled(
                self.display_frame.width() - 40, 
                self.display_frame.height() - 40, 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            
            # Set directly
            self.image_label.setPixmap(scaled_pixmap)
            logger.info(f"Set image directly: {scaled_pixmap.width()}x{scaled_pixmap.height()}")
    
    def _update_iteration_indicator(self, iteration):
        """Update the iteration indicator display"""
        if iteration == 0:
            self.iteration_indicator.setText("Initial Design")
        else:
            self.iteration_indicator.setText(f"Iteration {iteration}")
        
        self.iteration_indicator.show()
        
    def _set_placeholder_text(self, text):
        """Show placeholder text and hide image"""
        self.circuit_display.setText(text)
        self.circuit_display.show()
        self.image_label.clear()
        
    def resizeEvent(self, event):
        """Handle resize to maintain square display"""
        super().resizeEvent(event)
        
        available_width = self.width() - 100
        available_height = self.height() - 200
        square_size = min(available_width, available_height)
        
        if square_size > 0:
            self.display_frame.setFixedSize(square_size, square_size)
            
            # Resize current image if one is displayed
            if self.current_image_path and os.path.exists(self.current_image_path):
                pixmap = QPixmap(self.current_image_path)
                if not pixmap.isNull():
                    # Scale to display frame size accounting for padding
                    self.image_label.setPixmap(pixmap.scaled(
                        square_size - 40, 
                        square_size - 40, 
                        Qt.KeepAspectRatio, 
                        Qt.SmoothTransformation
                    ))
    
    def clear_image(self):
        """Clear the current image display"""
        self.current_image_path = None
        self.image_label.clear()
        self.circuit_display.setText("Circuit Screenshot Placeholder")
        self.circuit_display.show()
        self.iteration_indicator.hide()