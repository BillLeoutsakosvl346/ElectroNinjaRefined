# right_panel.py

import logging
from PyQt5.QtWidgets import (
    QFrame, QVBoxLayout, QLabel, QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont
from electroninja.ui.components.chat_panel import ChatPanel
from electroninja.ui.components.chat_input import ChatInputWidget

logger = logging.getLogger('electroninja')

class RightPanel(QFrame):
    """Right panel for chat interface"""
    
    messageSent = pyqtSignal(str)  # Emitted when the user sends a new message
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_processing = False  # Track if we're processing a request
        self.last_message = ""
        self.initUI()
        
    def initUI(self):
        """Initialize the UI components"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Panel title
        self.chat_title = QLabel("Chat with ElectroNinja", self)
        self.chat_title.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: white; letter-spacing: 0.5px;"
        )
        self.chat_title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.chat_title)
        
        # Chat messages area
        self.chat_panel = ChatPanel(self)
        self.chat_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.chat_panel, stretch=1)
        
        # Chat input area
        self.chat_input = ChatInputWidget(max_lines=5, parent=self)
        self.chat_input.sendMessage.connect(self.onSendMessage)
        main_layout.addWidget(self.chat_input)
        
    def onSendMessage(self, text):
        """Handle a new message from the user"""
        if not text.strip() or self.is_processing:
            return
            
        # Don't add user message to chat panel here - let main_window handle it
        # to avoid duplication
            
        # Clear input
        self.chat_input.message_input.clear()
        
        # Emit signal to notify parent
        self.messageSent.emit(text)
        
    def set_processing(self, is_processing):
        """
        Set whether circuit processing is in progress
        
        Args:
            is_processing (bool): True if processing is active, False otherwise
        """
        self.is_processing = is_processing
        
        # Update send button state
        self.chat_input.send_button.setEnabled(not is_processing)
        
        # Update styling based on state
        if is_processing:
            self.chat_input.send_button.setStyleSheet("""
                background-color: #555555;
                color: #999999;
                border-radius: 8px;
            """)
        else:
            self.chat_input.send_button.setStyleSheet("")  # Reset to default styling
            
    def receive_message(self, message):
        """
        Display a message from the assistant
        
        Args:
            message (str): Message to display
        """
        # Skip empty messages
        if not message or not message.strip():
            return
        
        logger.info(f"Receiving message in chat panel: {message[:50]}...")
        
        # Check for duplicate message
        if self.last_message == message:
            logger.info("Skipping duplicate message")
            return
        
        # Store for duplicate checking
        self.last_message = message
        
        # Add to chat with small delay for smooth UI
        QTimer.singleShot(50, lambda: self.chat_panel.add_message(message, is_user=False))
    
    def receive_message_with_type(self, message, message_type="normal"):
        """
        Display a message with type-specific styling
        
        Args:
            message (str): Message to display
            message_type (str): Message type ('normal', 'initial', 'refining', 'complete')
        """
        # Skip empty messages
        if not message or not message.strip():
            return
        
        logger.info(f"Receiving {message_type} message: {message[:50]}...")
        
        # Check for duplicate message
        if self.last_message == message:
            logger.info("Skipping duplicate message")
            return
        
        # Store for duplicate checking
        self.last_message = message
        
        # Add to chat with small delay for smooth UI
        QTimer.singleShot(50, lambda: self._add_styled_message(message, message_type))
    
    def _add_styled_message(self, message, message_type="normal"):
        """
        Add a message with type-specific styling
        
        Args:
            message (str): The message content
            message_type (str): Type for styling ('normal', 'initial', 'refining', 'complete')
        """
        bubble = self.chat_panel.add_message(message, is_user=False)
        
        # Apply styling based on message type
        if message_type == "initial":
            # Use default styling
            pass
        elif message_type == "refining":
            # Refinement message - orange hint
            bubble.setStyleSheet("""
                background-color: #664B33;  /* Slightly orange tint */
                border-radius: 6px;
                color: white;
                border: none;
            """)
        elif message_type == "complete":
            # Completion message - green hint
            bubble.setStyleSheet("""
                background-color: #335940;  /* Slightly green tint */
                border-radius: 6px;
                color: white;
                border: none;
            """)
        
    def clear_chat(self):
        """Clear all chat messages"""
        self.chat_panel.clear_chat()
        self.last_message = ""