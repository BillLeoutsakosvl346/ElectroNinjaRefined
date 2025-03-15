import logging
import os
import time
from PyQt5.QtCore import QThread, pyqtSignal

from electroninja.config.settings import Config
from electroninja.llm.providers.openai import OpenAIProvider
from electroninja.backend.workflow_orchestrator import WorkflowOrchestrator
from electroninja.backend.ltspice_manager import LTSpiceManager
from electroninja.backend.vision_processor import VisionProcessor

logger = logging.getLogger('electroninja')

class OutputMonitor(QThread):
    """
    Monitors output directories for new ASC code and images.
    Emits signals when new files are detected.
    """
    
    ascCodeDetected = pyqtSignal(str)
    imageDetected = pyqtSignal(str)
    
    def __init__(self, output_dir, prompt_id):
        super().__init__()
        self.output_dir = output_dir
        self.prompt_id = prompt_id
        self.running = False
        self.known_files = set()
        
    def run(self):
        """Run the monitor thread"""
        self.running = True
        
        # Path to prompt directory
        prompt_dir = os.path.join(self.output_dir, f"prompt{self.prompt_id}")
        
        # Initial scan
        self._scan_directory(prompt_dir)
        
        # Monitor loop
        while self.running:
            # Scan for new files
            new_files = self._scan_directory(prompt_dir)
            
            # Process new files
            for file_path in new_files:
                if file_path.endswith(".asc"):
                    try:
                        with open(file_path, "r") as f:
                            asc_code = f.read()
                        
                        # Extract iteration from path
                        iteration = 0
                        if "output" in file_path:
                            try:
                                import re
                                match = re.search(r'output(\d+)', file_path)
                                if match:
                                    iteration = int(match.group(1))
                            except:
                                pass
                        
                        # Tag as initial and emit immediately
                        if iteration > 0:
                            self.ascCodeDetected.emit(f"[ITERATION {iteration}] {asc_code}")
                        else:
                            self.ascCodeDetected.emit(f"[INITIAL] {asc_code}")
                            
                        logger.info(f"ASC code detected: {file_path}")
                    except Exception as e:
                        logger.error(f"Error reading ASC file: {str(e)}")
                elif file_path.endswith(".png"):
                    # Emit image path (full path)
                    self.imageDetected.emit(file_path)
                    logger.info(f"Image detected: {file_path}")
                    
            # Sleep before next scan
            time.sleep(0.5)
            
    def stop(self):
        """Stop the monitor thread"""
        self.running = False
        
    def _scan_directory(self, prompt_dir):
        """
        Scan directory for new files.
        
        Returns:
            list: List of new file paths
        """
        new_files = []
        
        if not os.path.exists(prompt_dir):
            return new_files
            
        # Look for output directories (output0, output1, etc.)
        for item in os.listdir(prompt_dir):
            if item.startswith("output"):
                output_dir = os.path.join(prompt_dir, item)
                
                if os.path.isdir(output_dir):
                    # Check for ASC and PNG files
                    for file_name in os.listdir(output_dir):
                        if file_name in ("code.asc", "image.png"):
                            file_path = os.path.join(output_dir, file_name)
                            
                            if file_path not in self.known_files:
                                self.known_files.add(file_path)
                                new_files.append(file_path)
                                
        return new_files

class ElectroNinjaWorker(QThread):
    """
    Worker thread for processing requests asynchronously.
    
    Signals:
        chatResponseGenerated: Emitted when a chat response is generated
        ascCodeGenerated: Emitted when ASC code is generated
        imageGenerated: Emitted when an image is generated
        visionFeedbackGenerated: Emitted when vision feedback is received
        resultReady: Emitted when the process is complete
    """
    
    # Signals for UI updates
    chatResponseGenerated = pyqtSignal(str)
    ascCodeGenerated = pyqtSignal(str)
    imageGenerated = pyqtSignal(str)
    visionFeedbackGenerated = pyqtSignal(str)
    resultReady = pyqtSignal(dict)
    
    def __init__(self, request, prompt_id):
        """
        Initialize the worker thread.
        
        Args:
            request (str): User request
            prompt_id (int): ID for the prompt (for folder structure)
        """
        super().__init__()
        self.request = request
        self.prompt_id = prompt_id
        self.config = Config()
        self.vision_processed = False
        
        # Initialize backend components
        self.llm_provider = OpenAIProvider(self.config)
        self.orchestrator = WorkflowOrchestrator(self.llm_provider, self.config)
        self.ltspice_manager = LTSpiceManager(self.config)
        self.vision_processor = VisionProcessor(self.config)
        
        # Output monitor
        self.monitor = None
        
    def run(self):
        """Execute the worker thread"""
        try:
            # Start output monitor to catch file system changes
            self._start_monitor()
            
            # Process the request
            self._process_request()
        except Exception as e:
            error_msg = f"Error in worker thread: {str(e)}"
            logger.error(error_msg)
            
            # Signal completion with error
            self.resultReady.emit({
                "success": False,
                "error": str(e),
                "vision_processed": self.vision_processed
            })
        finally:
            # Stop output monitor
            self._stop_monitor()
            
    def _start_monitor(self):
        """Start the output monitor"""
        self.monitor = OutputMonitor(self.config.OUTPUT_DIR, self.prompt_id)
        self.monitor.ascCodeDetected.connect(self.ascCodeGenerated)
        self.monitor.imageDetected.connect(self.imageGenerated)
        self.monitor.start()
        
    def _stop_monitor(self):
        """Stop the output monitor"""
        if self.monitor:
            self.monitor.stop()
            self.monitor.wait()
            
    def manual_vision_evaluation(self, image_path, original_request):
        """
        Manually trigger vision evaluation for an image
        
        Args:
            image_path (str): Path to the image file
            original_request (str): Original user request
        """
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image does not exist for vision evaluation: {image_path}")
                return
                
            logger.info(f"Manually evaluating image with vision model: {image_path}")
            
            # Run the vision analysis
            vision_feedback = self.vision_processor.analyze_circuit_image(image_path, original_request)
            
            # Emit the feedback
            self.visionFeedbackGenerated.emit(vision_feedback)
            
            # Process the feedback
            self.process_vision_feedback(vision_feedback)
            
        except Exception as e:
            logger.error(f"Error in manual vision evaluation: {str(e)}")
            
    def process_vision_feedback(self, vision_feedback):
        """
        Process vision feedback by generating a chat response
        
        Args:
            vision_feedback (str): Feedback from vision model
        """
        try:
            logger.info(f"Processing vision feedback: {'Y' if vision_feedback.strip() == 'Y' else 'Needs improvement'}")
            
            # Generate feedback response
            response = self.orchestrator.chat_generator.generate_feedback_response(vision_feedback)
            
            # Tag based on whether circuit is verified
            if vision_feedback.strip() == 'Y':
                tagged_response = "[COMPLETE] " + response
            else:
                tagged_response = "[REFINING] " + response
                
            # Emit signal
            self.chatResponseGenerated.emit(tagged_response)
            
            # Mark vision as processed
            self.vision_processed = True
            
            logger.info(f"Vision feedback processed and response generated")
            
            # If the circuit needs refinement, trigger refinement process
            if vision_feedback.strip() != 'Y':
                # This would normally happen in the iteration controller
                logger.info("Circuit needs refinement, would trigger refinement here")
                # In a full implementation, we would:
                # 1. Extract the latest history from orchestrator
                # 2. Call circuit_generator.refine_asc_code
                # 3. Send the refined code back to the UI
                
        except Exception as e:
            logger.error(f"Error processing vision feedback: {str(e)}")
            
    def _process_request(self):
        """Process a user request through the workflow orchestrator"""
        logger.info(f"Processing request: '{self.request}'")
        
        # Monkey patch the orchestrator to capture outputs in real-time
        original_generate_response = self.orchestrator.chat_generator.generate_response
        original_generate_feedback = self.orchestrator.chat_generator.generate_feedback_response
        original_generate_asc = self.orchestrator.circuit_generator.generate_asc_code
        original_refine_asc = self.orchestrator.circuit_generator.refine_asc_code
        original_analyze_circuit = self.orchestrator.vision_processor.analyze_circuit_image
        
        def generate_response_hook(prompt, is_circuit_related):
            """Hook to capture initial chat responses"""
            response = original_generate_response(prompt, is_circuit_related)
            # Add tag and emit signal
            tagged_response = "[INITIAL] " + response
            self.chatResponseGenerated.emit(tagged_response)
            logger.info(f"Generated initial chat response: {response[:50]}...")
            return response
            
        def generate_feedback_hook(vision_feedback):
            """Hook to capture feedback responses"""
            response = original_generate_feedback(vision_feedback)
            
            # Tag based on whether circuit is verified
            if vision_feedback.strip() == 'Y':
                tagged_response = "[COMPLETE] " + response
            else:
                tagged_response = "[REFINING] " + response
                
            # Emit signal
            self.chatResponseGenerated.emit(tagged_response)
            logger.info(f"Generated feedback response: {response[:50]}...")
            
            # Mark vision as processed
            self.vision_processed = True
            
            return response
            
        def generate_asc_hook(prompt):
            """Hook to capture initial ASC generation"""
            asc_code = original_generate_asc(prompt)
            
            # Skip if not circuit-related
            if asc_code == "N":
                return "N"
            
            # Tag as initial and emit
            tagged_asc = "[INITIAL] " + asc_code
            self.ascCodeGenerated.emit(tagged_asc)
            logger.info("Generated initial ASC code")
            
            return asc_code
            
        def refine_asc_hook(original_request, history):
            """Hook to capture ASC refinement"""
            refined_asc = original_refine_asc(original_request, history)
            
            # Get current iteration number
            iteration = len(history)
            
            # Tag with iteration number and emit
            tagged_asc = f"[ITERATION {iteration}] " + refined_asc
            self.ascCodeGenerated.emit(tagged_asc)
            logger.info(f"Generated refined ASC code for iteration {iteration}")
            
            return refined_asc
            
        def analyze_circuit_hook(image_path, original_request):
            """Hook to capture vision analysis"""
            analysis = original_analyze_circuit(image_path, original_request)
            
            # Emit vision feedback for UI
            self.visionFeedbackGenerated.emit(analysis)
            logger.info(f"Generated vision feedback: {'Y' if analysis.strip() == 'Y' else 'Needs improvement'}")
            
            return analysis
        
        try:
            # Apply the monkey patches
            self.orchestrator.chat_generator.generate_response = generate_response_hook
            self.orchestrator.chat_generator.generate_feedback_response = generate_feedback_hook
            self.orchestrator.circuit_generator.generate_asc_code = generate_asc_hook
            self.orchestrator.circuit_generator.refine_asc_code = refine_asc_hook
            self.orchestrator.vision_processor.analyze_circuit_image = analyze_circuit_hook
            
            # Process the request - this will run the full pipeline including vision
            result = self.orchestrator.process_request(self.request, self.prompt_id)
            
            # Check if vision was processed
            result["vision_processed"] = self.vision_processed
            
            # Check for existing image that might need vision processing
            self._check_for_existing_image()
            
            # Signal completion
            self.resultReady.emit(result)
        finally:
            # Restore the original methods
            self.orchestrator.chat_generator.generate_response = original_generate_response
            self.orchestrator.chat_generator.generate_feedback_response = original_generate_feedback
            self.orchestrator.circuit_generator.generate_asc_code = original_generate_asc
            self.orchestrator.circuit_generator.refine_asc_code = original_refine_asc
            self.orchestrator.vision_processor.analyze_circuit_image = original_analyze_circuit
            
    def _check_for_existing_image(self):
        """Check for existing image files and process them if needed"""
        try:
            # Skip if vision already processed
            if self.vision_processed:
                return
                
            # Look for the image file
            image_path = os.path.join(
                self.config.OUTPUT_DIR,
                f"prompt{self.prompt_id}", 
                "output0",  # Start with first iteration
                "image.png"
            )
            
            if os.path.exists(image_path):
                logger.info(f"Found existing image that needs processing: {image_path}")
                
                # Emit the image path
                self.imageGenerated.emit(image_path)
                
                # Run vision evaluation
                self.manual_vision_evaluation(image_path, self.request)
                
        except Exception as e:
            logger.error(f"Error checking for existing image: {str(e)}")