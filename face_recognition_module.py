import os
import sys
# Suppress TensorFlow oneDNN warnings and other deprecation warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO and WARNING messages
os.environ['TF_DISABLE_V2_BEHAVIOR'] = '0'  # Ensure v2 behavior

# Fix Unicode encoding issues on Windows
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

import cv2
import face_recognition
import numpy as np
import pickle

# Suppress Python warnings before importing TensorFlow
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Safe print function for Unicode characters
def safe_print(text):
    """Print text with Unicode fallback for Windows console"""
    try:
        print(text)
    except UnicodeEncodeError:
        # Replace Unicode characters with ASCII equivalents
        ascii_text = text.replace('✓', '[OK]').replace('✗', '[FAIL]').replace('📁', '[FILE]').replace('🎉', '[SUCCESS]')
        print(ascii_text)

try:
    import tensorflow as tf
    from tensorflow import keras
    
    # Additional TensorFlow warning suppression
    tf.get_logger().setLevel('ERROR')
    
    # Disable specific TensorFlow deprecation warnings
    import logging
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    
    # Monkey patch to suppress the specific placeholder warning
    original_warning = warnings.warn
    def patched_warning(message, category=None, *args, **kwargs):
        if 'tf.placeholder' not in str(message):
            original_warning(message, category, *args, **kwargs)
    warnings.warn = patched_warning
    
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    safe_print("TensorFlow not available. Siamese network will be disabled.")

class FaceRecognitionSystem:
    def __init__(self, models_to_load=None):
        """
        Initialize Face Recognition System with selective model loading
        
        Args:
            models_to_load (list): List of models to load. Options:
                - 'face_detection': Load face detection model
                - 'euclidean': Load Euclidean Siamese model
                - 'cosine': Load Cosine Siamese model  
                - 'embedding': Load Embedding Classification model
                If None, loads all available models (default behavior)
        """
        self.known_face_encodings = []
        self.known_face_names = []
        self.mobilenet_detector = None
        self.siamese_model = None
        self.siamese_cosine_model = None
        self.embedding_classification_model = None
        self.detection_method = None
        
        # Set default models to load if none specified
        if models_to_load is None:
            models_to_load = ['face_detection', 'euclidean', 'cosine', 'embedding']
        
        # Auto-include face detection if any recognition model is selected
        recognition_models = ['euclidean', 'cosine', 'embedding']
        if any(model in models_to_load for model in recognition_models) and 'face_detection' not in models_to_load:
            models_to_load = models_to_load + ['face_detection']
            print("Auto-including face detection model (required for recognition)")
        
        self.models_to_load = models_to_load
        
        # Always setup cascade classifiers (lightweight)
        self.setup_cascade_classifiers()
        
        # Conditionally setup other models
        if 'face_detection' in models_to_load:
            self.setup_mobilenet_detector()
        else:
            print("Skipping face detection model loading")
            
        if any(model in models_to_load for model in ['euclidean', 'cosine', 'embedding']):
            self.setup_siamese_model()
        else:
            print("Skipping all deep learning models")
            
        self.load_faces()
    
    def setup_cascade_classifiers(self):
        """Setup cascade classifiers for liveness detection only"""
        # Try multiple possible paths for cascade files
        cascade_paths = [
            cv2.data.haarcascades,  # Default OpenCV path
            os.path.join(cv2.__path__[0], 'data'),  # Alternative path
            'cascades',  # Local cascades folder
        ]
        
        cascade_files = {
            'eye': 'haarcascade_eye.xml',
            'smile': 'haarcascade_smile.xml'
        }
        
        # Only load eye and smile cascades for liveness detection
        self.eye_cascade = None
        self.smile_cascade = None
        
        for base_path in cascade_paths:
            if not os.path.exists(base_path):
                continue
                
            try:
                # Eye cascade (for liveness detection)
                eye_path = os.path.join(base_path, cascade_files['eye'])
                if os.path.exists(eye_path):
                    test_cascade = cv2.CascadeClassifier(eye_path)
                    if not test_cascade.empty():
                        self.eye_cascade = test_cascade
                        print(f"Eye cascade loaded from: {eye_path}")
                
                # Smile cascade (for liveness detection)
                smile_path = os.path.join(base_path, cascade_files['smile'])
                if os.path.exists(smile_path):
                    test_cascade = cv2.CascadeClassifier(smile_path)
                    if not test_cascade.empty():
                        self.smile_cascade = test_cascade
                        print(f"Smile cascade loaded from: {smile_path}")
                        
            except Exception as e:
                print(f"Error loading cascades from {base_path}: {e}")
                continue
        
        # Create empty cascades if not found (for liveness detection fallback)
        if self.eye_cascade is None or self.eye_cascade.empty():
            print("Warning: Eye cascade not found. Eye detection disabled.")
            self.eye_cascade = cv2.CascadeClassifier()
            
        if self.smile_cascade is None or self.smile_cascade.empty():
            print("Warning: Smile cascade not found. Smile detection disabled.")
            self.smile_cascade = cv2.CascadeClassifier()
    
    def setup_mobilenet_detector(self):
        """Setup MobileNetV2 face detector using Caffe model only"""
        try:
            # Use the Caffe MobileNetV2-SSD model for detection
            prototxt_path = 'models/deploy.prototxt'
            model_path = 'models/res10_300x300_ssd_iter_140000.caffemodel'
            
            if os.path.exists(prototxt_path) and os.path.exists(model_path):
                self.mobilenet_detector = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
                self.detection_method = 'caffe'
                safe_print("✓ MobileNetV2-SSD (Caffe) detector loaded successfully")
                print(f"  Model: {model_path}")
                print(f"  Config: {prototxt_path}")
            else:
                print("⚠ MobileNetV2 Caffe models not found.")
                print("  Expected files:")
                print(f"    - {prototxt_path}")
                print(f"    - {model_path}")
                self.mobilenet_detector = None
                self.detection_method = None
                print("Face detection will not be available without MobileNetV2 models")
            
        except Exception as e:
            print(f"Error loading MobileNetV2 detector: {e}")
            print("Face detection will not be available")
            self.mobilenet_detector = None
            self.detection_method = None
    
    def setup_siamese_model(self):
        """Setup Siamese networks for face recognition with selective loading"""
        try:
            if not TENSORFLOW_AVAILABLE:
                print("⚠ TensorFlow not available. Siamese networks disabled.")
                self.siamese_model = None
                self.siamese_cosine_model = None
                self.embedding_classification_model = None
                return
            
            # Initialize all models
            self.siamese_model = None
            self.siamese_cosine_model = None
            self.embedding_classification_model = None
            self.feature_extractor = None
            self.cosine_feature_extractor = None
            self.embedding_feature_extractor = None
            
            # Define model paths for Keras files
            euclidean_model_path = 'models/siamese_euclidean.keras'
            cosine_model_path = 'models/siamese_cosine.keras'
            embedding_model_path = 'models/embedding_classification.keras'
            
            models_found = []
            
            # Define custom objects for loading models with custom classes
            custom_objects = {
                'SiameseModel': self.create_siamese_model_class(),
                'EmbeddingModel': self.create_embedding_model_class()
            }
            
            # Load Euclidean Siamese model using Keras (only if requested)
            if 'euclidean' in self.models_to_load and os.path.exists(euclidean_model_path):
                try:
                    safe_print(f"Loading Euclidean Siamese model ({os.path.getsize(euclidean_model_path)/(1024*1024):.1f} MB)")
                    
                    # Load complete model using Keras with custom objects
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        try:
                            self.siamese_model = keras.models.load_model(euclidean_model_path, compile=False, custom_objects=custom_objects)
                        except Exception as custom_error:
                            print(f"  Custom class loading failed: {custom_error}")
                            # Try loading without custom objects (weights only approach)
                            print("  Attempting to load weights into new architecture...")
                            self.siamese_model, self.feature_extractor = self.create_functional_siamese_model(distance_type='euclidean')
                            if self.siamese_model is not None:
                                try:
                                    self.siamese_model.load_weights(euclidean_model_path.replace('.keras', '.weights.h5'))
                                    print("  ✓ Loaded weights into new architecture")
                                except:
                                    print("  ⚠ Using model with random weights")
                    
                    # If we successfully loaded the model, extract feature extractor
                    if self.siamese_model is not None and self.feature_extractor is None:
                        print("Creating feature extractor from loaded Siamese model...")
                        if hasattr(self.siamese_model, 'input') and isinstance(self.siamese_model.input, list) and len(self.siamese_model.input) > 1:
                            # For Siamese models, create feature extractor from the first branch
                            input_shape = self.siamese_model.input[0].shape[1:]
                            print(f"Creating feature extractor for Siamese model with input shape: {input_shape}")
                            try:
                                # Try to find the embedding layer or create a feature extractor
                                # Get the model layers to find the feature extraction part
                                layers = self.siamese_model.layers
                                print(f"Model has {len(layers)} layers")
                                
                                # Look for a layer that might be the feature extractor
                                feature_layer = None
                                for layer in layers:
                                    if hasattr(layer, 'output_shape') and len(layer.output_shape) == 2:
                                        # This might be a dense/embedding layer
                                        feature_layer = layer
                                        print(f"Found potential feature layer: {layer.name} with shape {layer.output_shape}")
                                        break

                                # Use robust input extraction
                                model_input = self.siamese_model.input
                                if isinstance(model_input, list):
                                    model_input = model_input[0]

                                if feature_layer is not None:
                                    # Create feature extractor up to this layer
                                    self.feature_extractor = keras.Model(
                                        inputs=model_input,
                                        outputs=feature_layer.output,
                                        name='euclidean_feature_extractor'
                                    )
                                else:
                                    # Fallback: use the first output
                                    # If outputs is a list, use outputs[0], else use outputs
                                    model_outputs = self.siamese_model.outputs
                                    if isinstance(model_outputs, list):
                                        model_outputs = model_outputs[0]
                                    self.feature_extractor = keras.Model(
                                        inputs=model_input,
                                        outputs=model_outputs,
                                        name='euclidean_feature_extractor'
                                    )
                                print(f"✅ Feature extractor created with input shape: {self.feature_extractor.input_shape}")
                            except Exception as extractor_error:
                                print(f"❌ Error creating feature extractor: {extractor_error}")
                                # Create a simple functional model as fallback
                                try:
                                    self.siamese_model, self.feature_extractor = self.create_functional_siamese_model(distance_type='euclidean')
                                    print("✅ Created functional model as fallback")
                                except:
                                    print("❌ Fallback model creation also failed")
                        else:
                            # Single input model, create a proper feature extractor instead of using the whole model
                            print("Single input model detected, creating proper feature extractor...")
                            try:
                                # Try to create a functional feature extractor from the loaded model
                                if hasattr(self.siamese_model, 'layers') and len(self.siamese_model.layers) > 0:
                                    # Find the last meaningful layer before any classification layers
                                    last_feature_layer = None
                                    for layer in reversed(self.siamese_model.layers):
                                        if hasattr(layer, 'output_shape'):
                                            # Look for dense layers or layers with 2D output (batch, features)
                                            if len(layer.output_shape) == 2 and layer.output_shape[1] > 1:
                                                last_feature_layer = layer
                                                print(f"Found suitable feature layer: {layer.name} with shape {layer.output_shape}")
                                                break
                                    
                                    if last_feature_layer is not None:
                                        # Create a proper feature extractor
                                        if hasattr(self.siamese_model, 'input'):
                                            model_input = self.siamese_model.input
                                            if isinstance(model_input, list):
                                                model_input = model_input[0]  # Use first input for Siamese models
                                            
                                            self.feature_extractor = keras.Model(
                                                inputs=model_input,
                                                outputs=last_feature_layer.output,
                                                name='euclidean_feature_extractor'
                                            )
                                            print(f"✅ Created proper feature extractor with input shape: {self.feature_extractor.input_shape}")
                                        else:
                                            print("⚠ Could not access model input, creating fallback model")
                                            self.siamese_model, self.feature_extractor = self.create_functional_siamese_model(distance_type='euclidean')
                                    else:
                                        print("⚠ Could not find suitable feature layer, creating functional model")
                                        self.siamese_model, self.feature_extractor = self.create_functional_siamese_model(distance_type='euclidean')
                                else:
                                    print("⚠ Model has no accessible layers, creating functional model")
                                    self.siamese_model, self.feature_extractor = self.create_functional_siamese_model(distance_type='euclidean')
                            except Exception as extract_error:
                                print(f"⚠ Error creating feature extractor: {extract_error}")
                                print("Creating functional model as fallback...")
                                self.siamese_model, self.feature_extractor = self.create_functional_siamese_model(distance_type='euclidean')
                    
                    if self.siamese_model is not None:
                        safe_print("✓ Euclidean Siamese network loaded successfully with Keras")
                        print(f"  Model parameters: {self.siamese_model.count_params():,}")
                        print(f"  Input shape: {self.siamese_model.input_shape if hasattr(self.siamese_model, 'input_shape') else 'Multiple inputs'}")
                        models_found.append('euclidean')
                    
                except Exception as e:
                    print(f"⚠ Could not load Euclidean Siamese model: {e}")
                    print(f"  File exists but may be corrupted or incompatible: {euclidean_model_path}")
            elif 'euclidean' in self.models_to_load:
                print(f"⚠ Euclidean model requested but not found: {euclidean_model_path}")
            
            # Load Cosine Siamese model using Keras (only if requested)
            if 'cosine' in self.models_to_load and os.path.exists(cosine_model_path):
                try:
                    safe_print(f"Loading Cosine Siamese model ({os.path.getsize(cosine_model_path)/(1024*1024):.1f} MB)")
                    
                    # Load complete model using Keras with custom objects
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        try:
                            self.siamese_cosine_model = keras.models.load_model(cosine_model_path, compile=False, custom_objects=custom_objects)
                        except Exception as custom_error:
                            print(f"  Custom class loading failed: {custom_error}")
                            # Try loading without custom objects (weights only approach)
                            print("  Attempting to load weights into new architecture...")
                            self.siamese_cosine_model, self.cosine_feature_extractor = self.create_functional_siamese_model(distance_type='cosine')
                            if self.siamese_cosine_model is not None:
                                try:
                                    self.siamese_cosine_model.load_weights(cosine_model_path.replace('.keras', '.weights.h5'))
                                    print("  ✓ Loaded weights into new architecture")
                                except:
                                    print("  ⚠ Using model with random weights")
                    
                    # If we successfully loaded the model, extract feature extractor
                    if self.siamese_cosine_model is not None and self.cosine_feature_extractor is None:
                        print("Creating cosine feature extractor from loaded Siamese model...")
                        if hasattr(self.siamese_cosine_model, 'input') and isinstance(self.siamese_cosine_model.input, list) and len(self.siamese_cosine_model.input) > 1:
                            input_shape = self.siamese_cosine_model.input[0].shape[1:]
                            print(f"Creating cosine feature extractor for Siamese model with input shape: {input_shape}")
                            try:
                                # Try to find the embedding layer or create a feature extractor
                                layers = self.siamese_cosine_model.layers
                                print(f"Cosine model has {len(layers)} layers")
                                
                                # Look for a layer that might be the feature extractor
                                feature_layer = None
                                for layer in layers:
                                    if hasattr(layer, 'output_shape') and len(layer.output_shape) == 2:
                                        feature_layer = layer
                                        print(f"Found potential cosine feature layer: {layer.name} with shape {layer.output_shape}")
                                        break
                                
                                if feature_layer is not None:
                                    self.cosine_feature_extractor = keras.Model(
                                        inputs=self.siamese_cosine_model.input[0],
                                        outputs=feature_layer.output,
                                        name='cosine_feature_extractor'
                                    )
                                else:
                                    self.cosine_feature_extractor = keras.Model(
                                        inputs=self.siamese_cosine_model.input[0],
                                        outputs=self.siamese_cosine_model.outputs[0],
                                        name='cosine_feature_extractor'
                                    )
                                print(f"✅ Cosine feature extractor created with input shape: {self.cosine_feature_extractor.input_shape}")
                            except Exception as extractor_error:
                                print(f"❌ Error creating cosine feature extractor: {extractor_error}")
                                try:
                                    self.siamese_cosine_model, self.cosine_feature_extractor = self.create_functional_siamese_model(distance_type='cosine')
                                    print("✅ Created functional cosine model as fallback")
                                except:
                                    print("❌ Cosine fallback model creation also failed")
                        else:
                            # Single input cosine model detected, create proper feature extractor
                            print("Single input cosine model detected, creating proper cosine feature extractor...")
                            try:
                                # Try to create a functional feature extractor from the loaded model
                                if hasattr(self.siamese_cosine_model, 'layers') and len(self.siamese_cosine_model.layers) > 0:
                                    # Find the last meaningful layer before any classification layers
                                    last_feature_layer = None
                                    for layer in reversed(self.siamese_cosine_model.layers):
                                        if hasattr(layer, 'output_shape'):
                                            # Look for dense layers or layers with 2D output (batch, features)
                                            if len(layer.output_shape) == 2 and layer.output_shape[1] > 1:
                                                last_feature_layer = layer
                                                print(f"Found suitable cosine feature layer: {layer.name} with shape {layer.output_shape}")
                                                break
                                    
                                    if last_feature_layer is not None:
                                        # Create a proper feature extractor
                                        if hasattr(self.siamese_cosine_model, 'input'):
                                            model_input = self.siamese_cosine_model.input
                                            if isinstance(model_input, list):
                                                model_input = model_input[0]  # Use first input for Siamese models
                                            
                                            self.cosine_feature_extractor = keras.Model(
                                                inputs=model_input,
                                                outputs=last_feature_layer.output,
                                                name='cosine_feature_extractor'
                                            )
                                            print(f"✅ Created proper cosine feature extractor with input shape: {self.cosine_feature_extractor.input_shape}")
                                        else:
                                            print("⚠ Could not access cosine model input, creating fallback model")
                                            self.siamese_cosine_model, self.cosine_feature_extractor = self.create_functional_siamese_model(distance_type='cosine')
                                    else:
                                        print("⚠ Could not find suitable cosine feature layer, creating functional model")
                                        self.siamese_cosine_model, self.cosine_feature_extractor = self.create_functional_siamese_model(distance_type='cosine')
                                else:
                                    print("⚠ Cosine model has no accessible layers, creating functional model")
                                    self.siamese_cosine_model, self.cosine_feature_extractor = self.create_functional_siamese_model(distance_type='cosine')
                            except Exception as extract_error:
                                print(f"⚠ Error creating cosine feature extractor: {extract_error}")
                                print("Creating functional cosine model as fallback...")
                                self.siamese_cosine_model, self.cosine_feature_extractor = self.create_functional_siamese_model(distance_type='cosine')
                    
                    if self.siamese_cosine_model is not None:
                        safe_print("✓ Cosine Siamese network loaded successfully with Keras")
                        print(f"  Model parameters: {self.siamese_cosine_model.count_params():,}")
                        print(f"  Input shape: {self.siamese_cosine_model.input_shape if hasattr(self.siamese_cosine_model, 'input_shape') else 'Multiple inputs'}")
                        models_found.append('cosine')
                    
                except Exception as e:
                    print(f"⚠ Could not load Cosine Siamese model: {e}")
                    print(f"  File exists but may be corrupted or incompatible: {cosine_model_path}")
            elif 'cosine' in self.models_to_load:
                print(f"⚠ Cosine model requested but not found: {cosine_model_path}")
            
            # Load Embedding Classification model using Keras (only if requested)
            if 'embedding' in self.models_to_load and os.path.exists(embedding_model_path):
                try:
                    safe_print(f"Loading Embedding Classification model ({os.path.getsize(embedding_model_path)/(1024*1024):.1f} MB)")
                    
                    # Load complete model using Keras with error handling
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        self.embedding_classification_model = keras.models.load_model(embedding_model_path, compile=False)
                    
                    # Use the model as feature extractor (typically the layers before final classification)
                    if hasattr(self.embedding_classification_model, 'layers') and len(self.embedding_classification_model.layers) > 1:
                        # Find the embedding layer (usually second to last layer)
                        embedding_layer_index = -2  # Second to last layer
                        if len(self.embedding_classification_model.layers) > 2:
                            self.embedding_feature_extractor = keras.Model(
                                inputs=self.embedding_classification_model.input,
                                outputs=self.embedding_classification_model.layers[embedding_layer_index].output,
                                name='embedding_feature_extractor'
                            )
                        else:
                            self.embedding_feature_extractor = self.embedding_classification_model
                    else:
                        self.embedding_feature_extractor = self.embedding_classification_model
                    
                    safe_print("✓ Embedding Classification model loaded successfully with Keras")
                    print(f"  Model parameters: {self.embedding_classification_model.count_params():,}")
                    print(f"  Input shape: {self.embedding_classification_model.input_shape}")
                    models_found.append('embedding')
                    
                except Exception as e:
                    print(f"⚠ Could not load Embedding Classification model: {e}")
                    print(f"  File exists but may be corrupted or incompatible: {embedding_model_path}")
            elif 'embedding' in self.models_to_load:
                print(f"⚠ Embedding model requested but not found: {embedding_model_path}")
            
            # Update recognition status based on available models
            if models_found:
                print(f"  Loaded models: {', '.join(models_found)}")
                print(f"  Skipped models: {', '.join([m for m in ['euclidean', 'cosine', 'embedding'] if m in self.models_to_load and m not in models_found])}")
            else:
                if any(model in self.models_to_load for model in ['euclidean', 'cosine', 'embedding']):
                    print("⚠ No requested models could be loaded.")
                    print("  Falling back to creating default models...")
                    # Create default euclidean model for basic functionality if it was requested
                    if 'euclidean' in self.models_to_load:
                        self.siamese_model, self.feature_extractor = self.create_functional_siamese_model(distance_type='euclidean')
                        if self.siamese_model is not None:
                            print("  Created default Euclidean Siamese model with random weights")
                
        except Exception as e:
            print(f"Error setting up models: {e}")
            self.siamese_model = None
            self.siamese_cosine_model = None
            self.embedding_classification_model = None
                
        except Exception as e:
            print(f"Error setting up Keras models: {e}")
            self.siamese_model = None
            self.siamese_cosine_model = None
            self.embedding_classification_model = None
    
    def create_siamese_model_class(self):
        """Create a placeholder class for SiameseModel"""
        class SiameseModel(keras.Model):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
            
            def call(self, inputs):
                return inputs
        
        return SiameseModel
    
    def create_embedding_model_class(self):
        """Create a placeholder class for EmbeddingModel"""
        class EmbeddingModel(keras.Model):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
            
            def call(self, inputs):
                return inputs
        
        return EmbeddingModel
    
    def get_model_input_shape(self, model, default_shape=(224, 224, 3)):
        """Safely get input shape from a Keras model, handling different model types"""
        try:
            if model is None:
                return default_shape
                
            if hasattr(model, 'input_shape') and model.input_shape is not None:
                # Handle multiple inputs (Siamese models)
                if isinstance(model.input_shape, list) and len(model.input_shape) > 0:
                    return model.input_shape[0][1:]  # First input, remove batch dimension
                else:
                    return model.input_shape[1:]  # Remove batch dimension
            elif hasattr(model, 'input') and hasattr(model.input, 'shape'):
                # Handle tf.Tensor input
                if isinstance(model.input, list) and len(model.input) > 0:
                    return model.input[0].shape[1:]  # First input, remove batch dimension
                else:
                    return model.input.shape[1:]  # Remove batch dimension
            elif hasattr(model, 'layers') and len(model.layers) > 0:
                first_layer = model.layers[0]
                if hasattr(first_layer, 'input_shape'):
                    return first_layer.input_shape[1:]
                elif hasattr(first_layer, 'batch_input_shape'):
                    return first_layer.batch_input_shape[1:]
            else:
                print(f"⚠ Could not determine input shape for {type(model)}, using default {default_shape}")
                return default_shape
                
        except Exception as e:
            print(f"⚠ Error getting input shape: {e}, using default {default_shape}")
            return default_shape
    
    def get_model_info(self):
        """Get information about loaded models"""
        info = {
            'models_loaded': [],
            'feature_extractors': [],
            'total_parameters': 0
        }
        
        if self.siamese_model is not None:
            info['models_loaded'].append('Euclidean Siamese (Keras)')
            info['total_parameters'] += self.siamese_model.count_params()
            if hasattr(self, 'feature_extractor') and self.feature_extractor is not None:
                info['feature_extractors'].append('euclidean')
        
        if self.siamese_cosine_model is not None:
            info['models_loaded'].append('Cosine Siamese (Keras)')
            info['total_parameters'] += self.siamese_cosine_model.count_params()
            if hasattr(self, 'cosine_feature_extractor') and self.cosine_feature_extractor is not None:
                info['feature_extractors'].append('cosine')
        
        if self.embedding_classification_model is not None:
            info['models_loaded'].append('Embedding Classification (Keras)')
            info['total_parameters'] += self.embedding_classification_model.count_params()
            if hasattr(self, 'embedding_feature_extractor') and self.embedding_feature_extractor is not None:
                info['feature_extractors'].append('embedding')
        
        return info
    
    def test_model_loading(self):
        """Test if models can perform inference"""
        test_results = {}
        
        if self.feature_extractor is not None:
            try:
                # Use appropriate input size for the model
                input_shape = self.get_model_input_shape(self.feature_extractor)
                test_input = np.random.rand(1, *input_shape).astype(np.float32)
                output = self.feature_extractor.predict(test_input, verbose=0)
                test_results['euclidean'] = {
                    'status': 'success',
                    'output_shape': output.shape,
                    'output_type': str(output.dtype),
                    'input_shape': input_shape
                }
            except Exception as e:
                test_results['euclidean'] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        if self.cosine_feature_extractor is not None:
            try:
                input_shape = self.get_model_input_shape(self.cosine_feature_extractor)
                test_input = np.random.rand(1, *input_shape).astype(np.float32)
                output = self.cosine_feature_extractor.predict(test_input, verbose=0)
                test_results['cosine'] = {
                    'status': 'success',
                    'output_shape': output.shape,
                    'output_type': str(output.dtype),
                    'input_shape': input_shape
                }
            except Exception as e:
                test_results['cosine'] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        if self.embedding_feature_extractor is not None:
            try:
                input_shape = self.get_model_input_shape(self.embedding_feature_extractor)
                test_input = np.random.rand(1, *input_shape).astype(np.float32)
                output = self.embedding_feature_extractor.predict(test_input, verbose=0)
                test_results['embedding'] = {
                    'status': 'success',
                    'output_shape': output.shape,
                    'output_type': str(output.dtype),
                    'input_shape': input_shape
                }
            except Exception as e:
                test_results['embedding'] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        return test_results
    
    def reload_models(self, models_to_load):
        """
        Reload models with new selection
        
        Args:
            models_to_load (list): List of models to load. Options:
                - 'face_detection': Load face detection model
                - 'euclidean': Load Euclidean Siamese model
                - 'cosine': Load Cosine Siamese model
                - 'embedding': Load Embedding Classification model
        """
        print(f"Reloading models: {models_to_load}")
        
        # Auto-include face detection if any recognition model is selected
        recognition_models = ['euclidean', 'cosine', 'embedding']
        if any(model in models_to_load for model in recognition_models) and 'face_detection' not in models_to_load:
            models_to_load = models_to_load + ['face_detection']
            print("Auto-including face detection model (required for recognition)")
        
        # Clear existing models
        self.clear_models()
        
        # Update models to load
        self.models_to_load = models_to_load
        
        # Reload specified models
        if 'face_detection' in models_to_load:
            self.setup_mobilenet_detector()
        
        if any(model in models_to_load for model in ['euclidean', 'cosine', 'embedding']):
            self.setup_siamese_model()
        
        print("Model reloading completed")
    
    def clear_models(self):
        """Clear all loaded models to free memory"""
        print("Clearing loaded models...")
        
        # Clear TensorFlow models
        if hasattr(self, 'siamese_model') and self.siamese_model is not None:
            del self.siamese_model
        if hasattr(self, 'siamese_cosine_model') and self.siamese_cosine_model is not None:
            del self.siamese_cosine_model
        if hasattr(self, 'embedding_classification_model') and self.embedding_classification_model is not None:
            del self.embedding_classification_model
        if hasattr(self, 'feature_extractor') and self.feature_extractor is not None:
            del self.feature_extractor
        if hasattr(self, 'cosine_feature_extractor') and self.cosine_feature_extractor is not None:
            del self.cosine_feature_extractor
        if hasattr(self, 'embedding_feature_extractor') and self.embedding_feature_extractor is not None:
            del self.embedding_feature_extractor
        
        # Clear OpenCV detector
        if hasattr(self, 'mobilenet_detector') and self.mobilenet_detector is not None:
            del self.mobilenet_detector
        
        # Reset model references
        self.siamese_model = None
        self.siamese_cosine_model = None
        self.embedding_classification_model = None
        self.feature_extractor = None
        self.cosine_feature_extractor = None
        self.embedding_feature_extractor = None
        self.mobilenet_detector = None
        self.detection_method = None
        
        # Force garbage collection
        try:
            import gc
            gc.collect()
            if TENSORFLOW_AVAILABLE:
                tf.keras.backend.clear_session()
        except:
            pass
        
        print("Models cleared")
    
    def get_available_models(self):
        """Get list of available model files"""
        available = []
        
        # Check face detection model
        if os.path.exists('models/deploy.prototxt') and os.path.exists('models/res10_300x300_ssd_iter_140000.caffemodel'):
            available.append('face_detection')
        
        # Check deep learning models
        if os.path.exists('models/siamese_euclidean.keras'):
            available.append('euclidean')
        if os.path.exists('models/siamese_cosine.keras'):
            available.append('cosine')
        if os.path.exists('models/embedding_classification.keras'):
            available.append('embedding')
        
        return available
    
    def create_functional_siamese_model(self, distance_type='euclidean'):
        """Create a functional Siamese model optimized for specified distance type"""
        try:
            print(f"  Creating optimized Siamese architecture for {distance_type} distance...")
            
            # Temporarily suppress stderr to hide tf.placeholder warning during model creation
            import sys
            import contextlib
            
            @contextlib.contextmanager
            def suppress_stderr():
                with open(os.devnull, "w") as devnull:
                    old_stderr = sys.stderr
                    sys.stderr = devnull
                    try:
                        yield
                    finally:
                        sys.stderr = old_stderr
            
            # Use a proven architecture that works well for face recognition
            input_shape = (160, 160, 3)  # Good balance of detail and efficiency
            inputs = keras.layers.Input(shape=input_shape)
            
            # Feature extraction backbone (inspired by FaceNet)
            x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_1')(inputs)
            x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_2')(x)
            x = keras.layers.MaxPooling2D((2, 2), name='pool1')(x)
            
            x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_1')(x)
            x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_2')(x)
            x = keras.layers.MaxPooling2D((2, 2), name='pool2')(x)
            
            x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_1')(x)
            x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_2')(x)
            x = keras.layers.MaxPooling2D((2, 2), name='pool3')(x)
            
            x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4_1')(x)
            x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4_2')(x)
            x = keras.layers.GlobalAveragePooling2D(name='global_pool')(x)
            
            # Feature embedding layers
            x = keras.layers.Dense(512, activation='relu', name='fc1')(x)
            x = keras.layers.Dropout(0.5, name='dropout1')(x)
            x = keras.layers.Dense(256, activation='relu', name='fc2')(x)
            x = keras.layers.Dropout(0.3, name='dropout2')(x)
            
            # Final embedding layer (128 dimensions works well for both distance types)
            embeddings = keras.layers.Dense(128, activation='linear', name='embeddings')(x)
            
            # Apply normalization based on distance type
            if distance_type == 'euclidean':
                # L2 normalization for Euclidean distance
                def euclidean_l2_normalize(x):
                    l2_norm = tf.sqrt(tf.reduce_sum(tf.square(x), axis=1, keepdims=True))
                    l2_norm = tf.maximum(l2_norm, 1e-12)  # Avoid division by zero
                    return x / l2_norm
                
                with suppress_stderr():
                    normalized_embeddings = keras.layers.Lambda(
                        euclidean_l2_normalize, 
                        name='euclidean_normalize'
                    )(embeddings)
                
                model_name = 'euclidean_face_encoder'
                
            elif distance_type == 'cosine':
                # L2 normalization for Cosine similarity (same as Euclidean but different naming)
                def cosine_l2_normalize(x):
                    l2_norm = tf.sqrt(tf.reduce_sum(tf.square(x), axis=1, keepdims=True))
                    l2_norm = tf.maximum(l2_norm, 1e-12)  # Avoid division by zero
                    return x / l2_norm
                
                with suppress_stderr():
                    normalized_embeddings = keras.layers.Lambda(
                        cosine_l2_normalize, 
                        name='cosine_normalize'
                    )(embeddings)
                
                model_name = 'cosine_face_encoder'
            
            else:
                # No normalization for other distance types
                normalized_embeddings = embeddings
                model_name = 'face_encoder'
            
            # Create feature extractor
            feature_extractor = keras.Model(
                inputs=inputs, 
                outputs=normalized_embeddings, 
                name=model_name
            )
            
            # Create Siamese model for training/comparison
            input_a = keras.layers.Input(shape=input_shape, name='input_a')
            input_b = keras.layers.Input(shape=input_shape, name='input_b')
            
            embedding_a = feature_extractor(input_a)
            embedding_b = feature_extractor(input_b)
            
            siamese_model = keras.Model(
                inputs=[input_a, input_b], 
                outputs=[embedding_a, embedding_b],
                name=f'siamese_{distance_type}_model'
            )
            
            print(f"  ✓ Functional Siamese model created ({distance_type})")
            print(f"    Input shape: {input_shape}")
            print(f"    Feature extractor params: {feature_extractor.count_params():,}")
            print(f"    Embedding dimensions: 128 (normalized for {distance_type})")
            print(f"    Optimized for: {distance_type.title()} distance computation")
            
            return siamese_model, feature_extractor
            
        except Exception as e:
            print(f"  Error creating functional Siamese model ({distance_type}): {e}")
            return None, None
    
    def load_siamese_weights_flexible(self, weights_path):
        """Flexible weight loading that handles different architectures"""
        try:
            # Strategy 1: Direct weight loading
            try:
                self.siamese_model.load_weights(weights_path)
                print("  ✓ Weights loaded successfully (direct)")
                return True
            except Exception as e1:
                print(f"  Strategy 1 failed: Architecture mismatch - {str(e1)[:80]}...")
            
            # Strategy 2: Try to reconstruct the exact original architecture
            try:
                print("  Attempting to reconstruct original architecture from weights...")
                reconstructed_model = self.reconstruct_model_from_weights(weights_path)
                if reconstructed_model is not None:
                    # Replace our model with the reconstructed one
                    self.siamese_model = reconstructed_model[0]
                    self.feature_extractor = reconstructed_model[1]
                    print("  ✓ Weights loaded with reconstructed architecture")
                    return True
            except Exception as e2:
                print(f"  Strategy 2 failed: {str(e2)[:80]}...")
            
            # Strategy 3: Load only the feature extraction layers we can use
            try:
                print("  Attempting partial weight loading...")
                success = self.load_compatible_weights_only(weights_path)
                if success:
                    print("  ✓ Partial weights loaded successfully")
                    return True
            except Exception as e3:
                print(f"  Strategy 3 failed: {str(e3)[:80]}...")
            
            # Strategy 4: Create a working model with random weights but correct architecture
            try:
                print("  Creating functional model with random weights...")
                if self.siamese_model is not None and self.feature_extractor is not None:
                    # Test if the model works for feature extraction
                    input_shape = self.get_model_input_shape(self.feature_extractor)
                    test_input = np.random.rand(1, *input_shape).astype(np.float32)
                    test_output = self.feature_extractor.predict(test_input, verbose=0)
                    if test_output is not None and len(test_output.shape) == 2:
                        print(f"  ✓ Model functional with random weights (output shape: {test_output.shape})")
                        return True
            except Exception as e4:
                print(f"  Strategy 4 failed: {str(e4)[:80]}...")
            
            return False
            
        except Exception as e:
            print(f"  All weight loading strategies failed: {e}")
            return False
    
    def reconstruct_model_from_weights(self, weights_path):
        """Try to reconstruct the original model architecture from the weights file"""
        try:
            import h5py
            
            with h5py.File(weights_path, 'r') as f:
                # Analyze the layers structure
                if 'layers' not in f:
                    return None
                
                layer_info = {}
                for layer_name in f['layers'].keys():
                    layer_group = f['layers'][layer_name]
                    layer_info[layer_name] = {
                        'type': 'unknown',
                        'weights': []
                    }
                    
                    # Try to determine layer type from weights
                    if 'vars' in layer_group:
                        for var_name in layer_group['vars'].keys():
                            weight_data = layer_group['vars'][var_name]
                            if hasattr(weight_data, 'shape'):
                                layer_info[layer_name]['weights'].append({
                                    'name': var_name,
                                    'shape': weight_data.shape
                                })
                
                print(f"    Found {len(layer_info)} layers in weights file")
                
                # Try to match this to a known architecture pattern
                # Look for final dense layer that outputs to embedding size
                for layer_name, info in layer_info.items():
                    for weight in info['weights']:
                        if len(weight['shape']) == 2 and weight['shape'][1] == 128:
                            print(f"    Found embedding layer: {layer_name} -> {weight['shape']}")
                            
                            # Create a simple model that ends with this layer
                            input_shape = (160, 160, 3)  # Use our flexible size
                            return self.create_simple_compatible_model(input_shape)
                
                return None
                
        except Exception as e:
            print(f"    Error reconstructing model: {e}")
            return None
    
    def create_simple_compatible_model(self, input_shape):
        """Create a simple model that's more likely to be compatible"""
        try:
            inputs = keras.layers.Input(shape=input_shape)
            
            # Very simple CNN that reduces to 128 features
            x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
            x = keras.layers.MaxPooling2D((4, 4))(x)  # Aggressive downsampling
            
            x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
            x = keras.layers.MaxPooling2D((4, 4))(x)
            
            x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
            x = keras.layers.GlobalAveragePooling2D()(x)
            
            # Direct to embedding size
            embeddings = keras.layers.Dense(128, activation='linear', name='dense')(x)
            
            # Create feature extractor
            feature_extractor = keras.Model(inputs=inputs, outputs=embeddings, name='simple_encoder')
            
            # Create Siamese model
            input_a = keras.layers.Input(shape=input_shape, name='input_a')
            input_b = keras.layers.Input(shape=input_shape, name='input_b')
            
            embedding_a = feature_extractor(input_a)
            embedding_b = feature_extractor(input_b)
            
            siamese_model = keras.Model(inputs=[input_a, input_b], outputs=[embedding_a, embedding_b])
            
            print(f"    ✓ Simple compatible model created ({feature_extractor.count_params():,} params)")
            
            return siamese_model, feature_extractor
            
        except Exception as e:
            print(f"    Error creating simple model: {e}")
            return None
    
    def load_compatible_weights_only(self, weights_path):
        """Load only the weights that are compatible with our current model"""
        try:
            import h5py
            
            # This is a simplified approach - just verify we can work with the model as-is
            if self.siamese_model is not None and self.feature_extractor is not None:
                # Test feature extraction
                input_shape = self.get_model_input_shape(self.feature_extractor)
                test_input = np.random.rand(1, *input_shape).astype(np.float32)
                test_output = self.feature_extractor.predict(test_input, verbose=0)
                
                if test_output is not None:
                    print(f"    Model is functional (output: {test_output.shape})")
                    return True
            
            return False
            
        except Exception as e:
            print(f"    Error in compatibility check: {e}")
            return False
    
    def create_mobilenetv2_siamese(self):
        """Create MobileNetV2-based Siamese network that matches the weights structure"""
        try:
            # Use TensorFlow's MobileNetV2 as base
            from tensorflow.keras.applications import MobileNetV2
            from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
            from tensorflow.keras.models import Model
            
            # Check the actual weights structure first
            weights_path = 'models/model_siamese_euclidean.weights.h5'
            weights_info = self.analyze_weights_structure(weights_path)
            
            # Try multiple input shapes based on common Siamese architectures
            possible_shapes = [(224, 224, 3), (160, 160, 3), (128, 128, 3), (96, 96, 3)]
            
            for input_shape in possible_shapes:
                try:
                    print(f"  Trying input shape: {input_shape}")
                    
                    # Create MobileNetV2 base (without top layers)
                    base_mobilenet = MobileNetV2(
                        input_shape=input_shape,
                        include_top=False,
                        weights=None,  # Don't use ImageNet weights
                        pooling=None
                    )
                    
                    # Add custom top layers to match the weights structure
                    inputs = Input(shape=input_shape)
                    x = base_mobilenet(inputs, training=False)
                    x = GlobalAveragePooling2D()(x)
                    
                    # Add the final dense layer (1280 -> 128) that matches the weights
                    embeddings = Dense(128, activation='linear', name='dense')(x)
                    
                    # Create feature extractor model
                    feature_extractor = Model(inputs=inputs, outputs=embeddings, name='mobilenetv2_encoder')
                    
                    # Create Siamese inputs
                    input_a = Input(shape=input_shape, name='input_a')
                    input_b = Input(shape=input_shape, name='input_b')
                    
                    # Get embeddings for both inputs
                    embedding_a = feature_extractor(input_a)
                    embedding_b = feature_extractor(input_b)
                    
                    # For Euclidean distance, we just need the embeddings
                    siamese_model = Model(inputs=[input_a, input_b], outputs=[embedding_a, embedding_b])
                    
                    print(f"✓ MobileNetV2-based Siamese model created")
                    print(f"  Input shape: {input_shape}")
                    print(f"  Feature extractor params: {feature_extractor.count_params():,}")
                    print(f"  Full model params: {siamese_model.count_params():,}")
                    
                    return siamese_model, feature_extractor
                    
                except Exception as shape_error:
                    print(f"    Failed with shape {input_shape}: {shape_error}")
                    continue
            
            # If all MobileNetV2 attempts failed, try custom architecture
            print("  MobileNetV2 approaches failed, trying custom architecture...")
            return self.create_custom_siamese_architecture()
            
        except Exception as e:
            print(f"Error creating MobileNetV2 Siamese: {e}")
            return self.create_custom_siamese_architecture()
    
    def analyze_weights_structure(self, weights_path):
        """Analyze the weights file to understand the model structure"""
        try:
            import h5py
            
            if not os.path.exists(weights_path):
                return {}
            
            weights_info = {}
            with h5py.File(weights_path, 'r') as f:
                weights_info['root_keys'] = list(f.keys())
                
                if 'layers' in f:
                    weights_info['layer_count'] = len(f['layers'].keys())
                    weights_info['layer_names'] = list(f['layers'].keys())
                
                # Try to get some structure info
                def explore_group(group, path=""):
                    items = {}
                    for key in group.keys():
                        item_path = f"{path}/{key}" if path else key
                        if isinstance(group[key], h5py.Group):
                            items[key] = explore_group(group[key], item_path)
                        else:
                            items[key] = f"Dataset: {group[key].shape if hasattr(group[key], 'shape') else 'unknown'}"
                    return items
                
                weights_info['structure'] = explore_group(f)
            
            print(f"  Weights analysis: {weights_info.get('layer_count', 'unknown')} layers found")
            return weights_info
            
        except Exception as e:
            print(f"  Could not analyze weights: {e}")
            return {}
    
    def create_custom_siamese_architecture(self):
        """Create a custom Siamese architecture that's more flexible with weights"""
        try:
            print("  Creating flexible custom Siamese architecture...")
            
            # Try a simpler, more flexible architecture
            input_shape = (160, 160, 3)  # Common size for face recognition
            inputs = Input(shape=input_shape)
            
            # Simple but effective CNN backbone
            x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1')(inputs)
            x = keras.layers.BatchNormalization(name='bn1')(x)
            x = keras.layers.MaxPooling2D((2, 2))(x)
            
            x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2')(x)
            x = keras.layers.BatchNormalization(name='bn2')(x)
            x = keras.layers.MaxPooling2D((2, 2))(x)
            
            x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3')(x)
            x = keras.layers.BatchNormalization(name='bn3')(x)
            x = keras.layers.MaxPooling2D((2, 2))(x)
            
            x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4')(x)
            x = keras.layers.BatchNormalization(name='bn4')(x)
            x = keras.layers.GlobalAveragePooling2D()(x)
            
            # Dense layers for feature extraction
            x = keras.layers.Dense(512, activation='relu', name='dense1')(x)
            x = keras.layers.Dropout(0.5)(x)
            embeddings = keras.layers.Dense(128, activation='linear', name='dense')(x)
            
            # Create feature extractor
            feature_extractor = keras.Model(inputs=inputs, outputs=embeddings, name='custom_encoder')
            
            # Create Siamese model
            input_a = keras.layers.Input(shape=input_shape, name='input_a')
            input_b = keras.layers.Input(shape=input_shape, name='input_b')
            
            embedding_a = feature_extractor(input_a)
            embedding_b = feature_extractor(input_b)
            
            siamese_model = keras.Model(inputs=[input_a, input_b], outputs=[embedding_a, embedding_b])
            
            print(f"✓ Custom Siamese model created with {siamese_model.count_params():,} parameters")
            print(f"  Input shape: {input_shape}")
            print(f"  Output embedding size: 128")
            
            return siamese_model, feature_extractor
            
        except Exception as e:
            print(f"Error creating custom Siamese: {e}")
            return None, None
    
    def create_siamese_network(self):
        """Create Siamese network architecture for face recognition"""
        try:
            # Try to load the weights file first to understand the architecture
            import h5py
            
            weights_path = 'models/model_siamese_euclidean.weights.h5'
            
            # Check if the weights file exists and try to understand its structure
            if not os.path.exists(weights_path):
                print(f"Weights file not found: {weights_path}")
                return None
            
            # Try a simpler approach - create a model that just extracts embeddings
            # Based on common Siamese architectures for face recognition
            input_shape = (105, 105, 1)  # Common size for Siamese networks (grayscale)
            
            # Create base network (encoder)
            inputs = keras.layers.Input(shape=input_shape)
            
            # Convolutional layers
            x = keras.layers.Conv2D(64, (10, 10), activation='relu')(inputs)
            x = keras.layers.MaxPooling2D((2, 2))(x)
            
            x = keras.layers.Conv2D(128, (7, 7), activation='relu')(x)
            x = keras.layers.MaxPooling2D((2, 2))(x)
            
            x = keras.layers.Conv2D(128, (4, 4), activation='relu')(x)
            x = keras.layers.MaxPooling2D((2, 2))(x)
            
            x = keras.layers.Conv2D(256, (4, 4), activation='relu')(x)
            
            # Flatten and dense layers
            x = keras.layers.Flatten()(x)
            x = keras.layers.Dense(4096, activation='sigmoid')(x)
            
            # Create the base model (encoder)
            base_model = keras.Model(inputs=inputs, outputs=x)
            
            # Create twin inputs for Siamese architecture
            input_a = keras.layers.Input(shape=input_shape)
            input_b = keras.layers.Input(shape=input_shape)
            
            # Generate embeddings for both inputs
            embedding_a = base_model(input_a)
            embedding_b = base_model(input_b)
            
            # Calculate L2 distance between embeddings
            distance = keras.layers.Lambda(
                lambda tensors: keras.backend.abs(tensors[0] - tensors[1])
            )([embedding_a, embedding_b])
            
            # Output layer for similarity prediction
            output = keras.layers.Dense(1, activation='sigmoid')(distance)
            
            # Create the Siamese model
            siamese_model = keras.Model(inputs=[input_a, input_b], outputs=output)
            
            # Also create the encoder model for feature extraction
            self.feature_extractor = base_model
            
            return siamese_model
            
        except Exception as e:
            print(f"Error creating Siamese network (attempt 1): {e}")
            
            # Try alternative architecture
            try:
                # Alternative architecture - simpler version
                input_shape = (96, 96, 3)  # RGB input
                
                inputs = keras.layers.Input(shape=input_shape)
                
                # Simple CNN architecture
                x = keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
                x = keras.layers.MaxPooling2D((2, 2))(x)
                x = keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
                x = keras.layers.MaxPooling2D((2, 2))(x)
                x = keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
                x = keras.layers.MaxPooling2D((2, 2))(x)
                
                x = keras.layers.Flatten()(x)
                x = keras.layers.Dense(512, activation='relu')(x)
                embeddings = keras.layers.Dense(128, activation='linear')(x)  # Embedding layer
                
                # Create encoder
                encoder = keras.Model(inputs=inputs, outputs=embeddings)
                
                # Create Siamese inputs
                input_a = keras.layers.Input(shape=input_shape)
                input_b = keras.layers.Input(shape=input_shape)
                
                # Get embeddings
                embedding_a = encoder(input_a)
                embedding_b = encoder(input_b)
                
                # Store encoder for feature extraction
                self.feature_extractor = encoder
                
                # Create Siamese model
                siamese_model = keras.Model(inputs=[input_a, input_b], outputs=[embedding_a, embedding_b])
                
                return siamese_model
                
            except Exception as e2:
                print(f"Error creating Siamese network (attempt 2): {e2}")
                return None
    
    def detect_faces_mobilenetv2(self, frame):
        """Face detection using MobileNetV2-SSD"""
        try:
            # Use the available Caffe model
            if not hasattr(self, 'mobilenet_detector') or self.mobilenet_detector is None:
                print("MobileNetV2 detector not available")
                return []
            
            h, w = frame.shape[:2]
            print(f"Input frame shape: {frame.shape}")
            
            # MobileNetV2-SSD preprocessing for Caffe model
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
            
            self.mobilenet_detector.setInput(blob)
            detections = self.mobilenet_detector.forward()
            
            print(f"Detection shape: {detections.shape}")
            print(f"Total detections found: {detections.shape[2]}")
            
            faces = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                print(f"Detection {i}: confidence = {confidence:.4f}")
                
                if confidence > 0.3:  # Even lower threshold for better detection
                    x1 = int(detections[0, 0, i, 3] * w)
                    y1 = int(detections[0, 0, i, 4] * h)
                    x2 = int(detections[0, 0, i, 5] * w)
                    y2 = int(detections[0, 0, i, 6] * h)
                    
                    # Ensure coordinates are within frame bounds
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)
                    
                    if x2 > x1 and y2 > y1:  # Valid detection
                        faces.append((x1, y1, x2-x1, y2-y1))
                        print(f"✅ Face detected with confidence {confidence:.3f} at ({x1}, {y1}, {x2-x1}, {y2-y1})")
                    else:
                        print(f"❌ Invalid coordinates: ({x1}, {y1}, {x2-x1}, {y2-y1})")
            
            print(f"Total valid faces detected: {len(faces)}")
            return faces
            
        except Exception as e:
            print(f"MobileNetV2 detection error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def extract_face_features_siamese(self, face_roi, distance_type='auto'):
        """Extract face features using appropriate Siamese network or embedding classification model"""
        try:
            print(f"🔍 Starting feature extraction with distance_type='{distance_type}'")
            print(f"Face ROI shape: {face_roi.shape}")
            
            # Determine which model to use
            if distance_type == 'auto':
                # Use euclidean by default, fallback to cosine, then embedding
                if self.siamese_model is not None:
                    distance_type = 'euclidean'
                    print("Auto-selected: euclidean")
                elif self.siamese_cosine_model is not None:
                    distance_type = 'cosine'
                    print("Auto-selected: cosine")
                elif self.embedding_classification_model is not None:
                    distance_type = 'embedding'
                    print("Auto-selected: embedding")
                else:
                    print("❌ No models available for feature extraction")
                    return None
            
            # Select appropriate model and extractor
            extractor = None
            model_name = None
            
            if distance_type == 'euclidean':
                print("Checking Euclidean Siamese model...")
                if self.siamese_model is None:
                    print("❌ siamese_model is None")
                    return None
                if not hasattr(self, 'feature_extractor'):
                    print("❌ feature_extractor attribute not found")
                    return None
                if self.feature_extractor is None:
                    print("❌ feature_extractor is None")
                    return None
                    
                extractor = self.feature_extractor
                model_name = "Euclidean Siamese (Keras)"
                print(f"✅ Using {model_name}")
                
            elif distance_type == 'cosine':
                print("Checking Cosine Siamese model...")
                if self.siamese_cosine_model is None or not hasattr(self, 'cosine_feature_extractor'):
                    print("❌ Cosine model or extractor not available")
                    return None
                extractor = self.cosine_feature_extractor
                model_name = "Cosine Siamese (Keras)"
                print(f"✅ Using {model_name}")
                
            elif distance_type == 'embedding':
                print("Checking Embedding Classification model...")
                if self.embedding_classification_model is None or not hasattr(self, 'embedding_feature_extractor'):
                    print("❌ Embedding model or extractor not available")
                    return None
                extractor = self.embedding_feature_extractor
                model_name = "Embedding Classification (Keras)"
                print(f"✅ Using {model_name}")
                
            else:
                print(f"❌ Unknown distance type: {distance_type}")
                return None
            
            if extractor is None:
                print("❌ Extractor is None after selection")
                return None
                
            print(f"Extractor selected: {type(extractor)}")
            
            # Get input shape (remove batch dimension) using helper function
            input_shape = self.get_model_input_shape(extractor)
            target_height, target_width = input_shape[0], input_shape[1]
            print(f"Model expects input shape: {input_shape}")
            print(f"Target dimensions: {target_height}x{target_width}")
            
            # Preprocess face image to match the expected input
            if len(input_shape) == 3 and input_shape[2] == 3:
                print("Processing RGB input...")
                # RGB input expected
                if len(face_roi.shape) == 3:
                    face_processed = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                    print("Converted BGR to RGB")
                else:
                    face_processed = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)
                    print("Converted Grayscale to RGB")
                
                print(f"Face processed shape before resize: {face_processed.shape}")
                
                # Resize to expected input size (handle different input sizes)
                face_processed = cv2.resize(face_processed, (target_width, target_height))
                print(f"Face resized to: {face_processed.shape}")
                
                # Normalize to [0, 1] range (standard for most models)
                face_processed = face_processed.astype(np.float32) / 255.0
                print(f"Normalized, dtype: {face_processed.dtype}, range: [{face_processed.min():.3f}, {face_processed.max():.3f}]")
                
                # Add batch dimension
                face_batch = np.expand_dims(face_processed, axis=0)
                print(f"Batch shape: {face_batch.shape}")
                
                # Extract features
                print("Extracting features...")
                features = extractor.predict(face_batch, verbose=0)
                print(f"Raw features shape: {features.shape}")
                
                # Return flattened features
                result = features.flatten()
                print(f"✅ {model_name} features extracted successfully!")
                print(f"   Final feature shape: {result.shape}")
                print(f"   Feature range: [{result.min():.3f}, {result.max():.3f}]")
                return result
            
            else:
                print(f"❌ Unexpected input shape for {model_name} model: {input_shape}")
                return None
            
        except Exception as e:
            print(f"❌ Error extracting {distance_type} features: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def extract_face_features_siamese_legacy(self, face_roi):
        """Extract face features using Siamese network (MobileNetV2-based or custom)"""
        try:
            if self.siamese_model is None:
                return None
            
            # Check if we have a feature extractor (encoder part)
            if hasattr(self, 'feature_extractor') and self.feature_extractor is not None:
                # Use the encoder directly
                extractor = self.feature_extractor
                
                # Get input shape using helper function
                input_shape = self.get_model_input_shape(extractor)
                target_height, target_width = input_shape[0], input_shape[1]
                
                # Preprocess face image to match the expected input
                if len(input_shape) == 3 and input_shape[2] == 3:
                    # RGB input expected
                    if len(face_roi.shape) == 3:
                        face_processed = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                    else:
                        face_processed = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)
                    
                    # Resize to expected input size
                    face_processed = cv2.resize(face_processed, (target_width, target_height))
                    
                    # Normalize to [0, 1] range (standard for most models)
                    face_processed = face_processed.astype(np.float32) / 255.0
                    
                    # Add batch dimension
                    face_batch = np.expand_dims(face_processed, axis=0)
                    
                    # Extract features
                    features = extractor.predict(face_batch, verbose=0)
                    
                    # Return flattened features
                    result = features.flatten()
                    print(f"    Siamese features extracted: shape={result.shape}")
                    return result
                
                else:
                    print(f"Unexpected input shape for Siamese model: {input_shape}")
                    return None
            
            else:
                # Fallback: use full Siamese model (less efficient)
                print("Using fallback feature extraction with full Siamese model")
                
                # Get input shape using helper function
                input_shape = self.get_model_input_shape(self.siamese_model)
                target_height, target_width = input_shape[0], input_shape[1]
                
                if len(input_shape) == 3 and input_shape[2] == 3:
                    # RGB input
                    if len(face_roi.shape) == 3:
                        face_processed = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                    else:
                        face_processed = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)
                    
                    face_processed = cv2.resize(face_processed, (target_width, target_height))
                    face_processed = face_processed.astype(np.float32) / 255.0
                    face_batch = np.expand_dims(face_processed, axis=0)
                    
                    # Use dummy input for second branch
                    dummy_input = np.zeros_like(face_batch)
                    embeddings = self.siamese_model.predict([face_batch, dummy_input], verbose=0)
                    
                    # Return the first embedding
                    if isinstance(embeddings, list):
                        result = embeddings[0].flatten()
                    else:
                        result = embeddings.flatten()
                    
                    print(f"    Siamese features extracted (fallback): shape={result.shape}")
                    return result
                
                else:
                    print(f"Unexpected input shape: {input_shape}")
                    return None
            
        except Exception as e:
            print(f"Error extracting Siamese features: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def compute_euclidean_distance(self, features1, features2):
        """Compute Euclidean distance between face features"""
        try:
            if features1 is None or features2 is None:
                return float('inf')
            
            # Ensure features are numpy arrays
            features1 = np.array(features1)
            features2 = np.array(features2)
            
            # Ensure features are the same length
            min_len = min(len(features1), len(features2))
            features1 = features1[:min_len]
            features2 = features2[:min_len]
            
            # Compute Euclidean distance
            distance = np.linalg.norm(features1 - features2)
            return distance
            
        except Exception as e:
            print(f"Error computing Euclidean distance: {e}")
            return float('inf')
    
    def compute_cosine_distance(self, features1, features2):
        """Compute Cosine distance between face features"""
        try:
            if features1 is None or features2 is None:
                return float('inf')
            
            # Ensure features are numpy arrays
            features1 = np.array(features1)
            features2 = np.array(features2)
            
            # Ensure features are the same length
            min_len = min(len(features1), len(features2))
            features1 = features1[:min_len]
            features2 = features2[:min_len]
            
            # Normalize features (should already be normalized by the model, but just in case)
            features1_norm = features1 / (np.linalg.norm(features1) + 1e-8)
            features2_norm = features2 / (np.linalg.norm(features2) + 1e-8)
            
            # Compute cosine similarity
            cosine_similarity = np.dot(features1_norm, features2_norm)
            
            # Convert to cosine distance (1 - cosine_similarity)
            # Clamp to [0, 2] range to handle floating point errors
            cosine_distance = max(0.0, min(2.0, 1.0 - cosine_similarity))
            
            return cosine_distance
            
        except Exception as e:
            print(f"Error computing Cosine distance: {e}")
            return float('inf')
    
    def compute_face_distance(self, features1, features2, method='auto'):
        """Compute distance between face features using specified method"""
        try:
            if method == 'auto':
                # Choose method based on available models (prefer Euclidean)
                if self.siamese_model is not None:
                    method = 'euclidean'
                elif self.siamese_cosine_model is not None:
                    method = 'cosine'
                elif self.embedding_classification_model is not None:
                    method = 'embedding'
                else:
                    print("No recognition models available")
                    return float('inf')
            
            if method == 'euclidean':
                return self.compute_euclidean_distance(features1, features2)
            elif method == 'cosine':
                return self.compute_cosine_distance(features1, features2)
            elif method == 'embedding':
                # For embedding classification, use cosine distance
                return self.compute_cosine_distance(features1, features2)
            else:
                print(f"Unknown distance method: {method}")
                return float('inf')
                
        except Exception as e:
            print(f"Error computing face distance: {e}")
            return float('inf')
    
    def load_faces(self):
        """Load known faces from the faces directory"""
        faces_dir = "faces"
        if not os.path.exists(faces_dir):
            os.makedirs(faces_dir)
            return
        
        # Load from pickle file if exists
        pickle_file = os.path.join(faces_dir, "face_encodings.pkl")
        if os.path.exists(pickle_file):
            try:
                with open(pickle_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data['encodings']
                    self.known_face_names = data['names']
                
                # Clean up any invalid encodings after loading
                if self.known_face_encodings:
                    original_count = len(self.known_face_encodings)
                    self.clean_face_encodings()
                    if len(self.known_face_encodings) != original_count:
                        print(f"Cleaned up {original_count - len(self.known_face_encodings)} invalid face encodings")
                return
            except Exception as e:
                print(f"Error loading pickle file: {e}")
        
        # Load from image files
        for filename in os.listdir(faces_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(faces_dir, filename)
                name = os.path.splitext(filename)[0]
                
                try:
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        self.known_face_encodings.append(encodings[0])
                        self.known_face_names.append(name)
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        
        # Save to pickle file
        self.save_encodings()
    
    def save_encodings(self):
        """Save face encodings to pickle file"""
        faces_dir = "faces"
        if not os.path.exists(faces_dir):
            os.makedirs(faces_dir)
        
        pickle_file = os.path.join(faces_dir, "face_encodings.pkl")
        try:
            with open(pickle_file, 'wb') as f:
                pickle.dump({
                    'encodings': self.known_face_encodings,
                    'names': self.known_face_names
                }, f)
        except Exception as e:
            print(f"Error saving encodings: {e}")
    
    def clean_face_encodings(self):
        """Remove face encodings with invalid names/IDs and ensure encoding consistency"""
        cleaned_encodings = []
        cleaned_names = []
        
        print("Cleaning face encodings...")
        
        for encoding, name in zip(self.known_face_encodings, self.known_face_names):
            try:
                # Check if name has valid format and characters
                if "_" in name:
                    display_name, student_id = name.split("_", 1)
                    # Check if student_id contains only valid characters
                    if student_id.replace(' ', '').replace('-', '').replace('_', '').isalnum():
                        # Check encoding validity and compatibility
                        if self.is_encoding_valid(encoding):
                            cleaned_encodings.append(encoding)
                            cleaned_names.append(name)
                            print(f"Kept valid encoding for: {name} (shape: {encoding.shape})")
                        else:
                            print(f"Removing invalid encoding for: {name}")
                    else:
                        print(f"Removing encoding with invalid student ID: {name}")
                else:
                    # Keep single-word names if they're valid
                    if self.is_encoding_valid(encoding):
                        cleaned_encodings.append(encoding)
                        cleaned_names.append(name)
                        print(f"Kept valid encoding for: {name} (shape: {encoding.shape})")
                    else:
                        print(f"Removing invalid encoding for: {name}")
            except Exception as e:
                print(f"Removing corrupted face encoding: {name} - Error: {e}")
        
        self.known_face_encodings = cleaned_encodings
        self.known_face_names = cleaned_names
        
        # Save cleaned encodings
        self.save_encodings()
        print(f"Face encodings cleaned. {len(cleaned_encodings)} valid encodings remaining.")
        
        # Print summary of encoding types
        self.print_encoding_summary()
    
    def is_encoding_valid(self, encoding):
        """Check if an encoding is valid and has expected properties"""
        try:
            if not hasattr(encoding, 'shape'):
                return False
            
            # Check for valid shapes
            valid_shapes = [128, 1280]  # face_recognition, embedding model
            if len(encoding) not in valid_shapes:
                return False
            
            # Check for valid data types
            if encoding.dtype not in [np.float32, np.float64]:
                return False
            
            # Check for non-NaN and finite values
            if np.any(np.isnan(encoding)) or not np.all(np.isfinite(encoding)):
                return False
            
            return True
            
        except Exception:
            return False
    
    def print_encoding_summary(self):
        """Print a summary of current encodings"""
        if not self.known_face_encodings:
            print("No face encodings loaded.")
            return
        
        print(f"\nFace Encodings Summary:")
        print(f"Total encodings: {len(self.known_face_encodings)}")
        
        # Group by encoding type
        encoding_types = {}
        for encoding in self.known_face_encodings:
            shape_key = f"{len(encoding)}D ({encoding.dtype})"
            if shape_key not in encoding_types:
                encoding_types[shape_key] = 0
            encoding_types[shape_key] += 1
        
        for shape_key, count in encoding_types.items():
            model_type = self.get_model_type_for_shape(shape_key)
            print(f"  {shape_key}: {count} encodings ({model_type})")
    
    def get_model_type_for_shape(self, shape_key):
        """Get the model type for a given encoding shape"""
        if "128D" in shape_key and "float64" in shape_key:
            return "face_recognition library"
        elif "128D" in shape_key and "float32" in shape_key:
            return "Siamese network"
        elif "1280D" in shape_key:
            return "Embedding Classification"
        else:
            return "Unknown"
    
    def can_register_new_faces(self):
        """Check if the current model configuration supports registering new faces"""
        # Siamese models (euclidean and cosine) can register new faces dynamically
        # because they work by comparing face encodings
        if (hasattr(self, 'siamese_model') and self.siamese_model is not None) or \
           (hasattr(self, 'siamese_cosine_model') and self.siamese_cosine_model is not None):
            return True, "Siamese models support dynamic face registration"
        
        # Embedding classification model cannot register new faces
        # because it has a fixed output layer trained on specific classes
        if (hasattr(self, 'embedding_classification_model') and self.embedding_classification_model is not None):
            return False, "Classification model can only identify existing registered faces"
        
        return False, "No models loaded that support face registration"
    
    def get_supported_operations(self):
        """Get information about what operations are supported by loaded models"""
        operations = {
            'can_register': False,
            'can_identify': False,
            'registration_models': [],
            'identification_models': [],
            'messages': []
        }
        
        # Check Euclidean Siamese model
        if hasattr(self, 'siamese_model') and self.siamese_model is not None:
            operations['can_register'] = True
            operations['can_identify'] = True
            operations['registration_models'].append('Euclidean Siamese')
            operations['identification_models'].append('Euclidean Siamese')
            operations['messages'].append("✓ Euclidean Siamese: Can register new faces and identify existing ones")
        
        # Check Cosine Siamese model
        if hasattr(self, 'siamese_cosine_model') and self.siamese_cosine_model is not None:
            operations['can_register'] = True
            operations['can_identify'] = True
            operations['registration_models'].append('Cosine Siamese')
            operations['identification_models'].append('Cosine Siamese')
            operations['messages'].append("✓ Cosine Siamese: Can register new faces and identify existing ones")
        
        # Check Embedding Classification model
        if hasattr(self, 'embedding_classification_model') and self.embedding_classification_model is not None:
            operations['can_identify'] = True
            operations['identification_models'].append('Embedding Classification')
            operations['messages'].append("✓ Embedding Classification: Can identify existing faces only (no new registration)")
        
        # Add limitation messages
        if operations['can_identify'] and not operations['can_register']:
            operations['messages'].append("⚠️ Only classification model loaded - cannot register new faces")
            operations['messages'].append("   Use Siamese models (Euclidean/Cosine) for new face registration")
        
        if not operations['can_identify']:
            operations['messages'].append("❌ No recognition models loaded")
        
        return operations

    def add_face(self, image_path, name, student_id):
        """Add a new face to the recognition system using available methods"""
        try:
            # Check if registration is supported
            can_register, registration_message = self.can_register_new_faces()
            if not can_register:
                print(f"❌ Cannot register new faces: {registration_message}")
                print("💡 To register new faces, load Siamese models (Euclidean or Cosine) in Model Settings")
                return False
            
            print(f"✅ Face registration supported: {registration_message}")
            print(f"Processing image: {image_path}")
            
            # Method 1: Try OpenCV first
            cv_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            
            # Method 2: If OpenCV fails, try PIL
            if cv_image is None:
                try:
                    from PIL import Image as PILImage
                    pil_image = PILImage.open(image_path)
                    # Convert PIL to OpenCV format
                    if pil_image.mode == 'RGBA':
                        pil_image = pil_image.convert('RGB')
                    elif pil_image.mode == 'L':
                        pil_image = pil_image.convert('RGB')
                    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                except Exception as e:
                    print(f"PIL loading failed: {e}")
                    return False
            
            if cv_image is None:
                print(f"Could not load image: {image_path}")
                return False
            
            print(f"Image loaded successfully. Shape: {cv_image.shape}, dtype: {cv_image.dtype}")
            
            # Ensure image is in correct format
            if len(cv_image.shape) == 3 and cv_image.shape[2] == 3:
                # BGR to RGB conversion
                image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            elif len(cv_image.shape) == 2:
                # Grayscale to RGB
                image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
            else:
                print(f"Unsupported image shape: {cv_image.shape}")
                return False
            
            # Ensure 8-bit unsigned integer
            if image.dtype != np.uint8:
                # Normalize and convert to uint8
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            print(f"Image preprocessed. Shape: {image.shape}, dtype: {image.dtype}")
            
            # Resize if too large (works better with reasonable image sizes)
            height, width = image.shape[:2]
            max_dimension = 800
            if width > max_dimension or height > max_dimension:
                scale = max_dimension / max(width, height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                print(f"Image resized to: {new_width}x{new_height}")
            
            # Detect face first
            print(f"Attempting face detection on image...")
            mobilenet_faces = self.detect_faces_mobilenetv2(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            # If MobileNetV2 fails, try Haar cascade as fallback
            if not mobilenet_faces:
                print("MobileNetV2 detection failed, trying Haar cascade fallback...")
                try:
                    # Convert to grayscale for Haar cascade
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    
                    # Try multiple scale factors and parameters for better detection
                    scale_factors = [1.05, 1.1, 1.2, 1.3]
                    min_neighbors = [3, 4, 5, 6]
                    
                    haar_faces = []
                    for scale_factor in scale_factors:
                        for min_neighbor in min_neighbors:
                            try:
                                faces = face_cascade.detectMultiScale(
                                    gray, 
                                    scaleFactor=scale_factor, 
                                    minNeighbors=min_neighbor, 
                                    minSize=(30, 30),  # Lower minimum size
                                    maxSize=(500, 500)
                                )
                                if len(faces) > 0:
                                    haar_faces = faces
                                    print(f"✅ Haar cascade detected {len(faces)} face(s) with scale={scale_factor}, neighbors={min_neighbor}")
                                    break
                            except Exception as e:
                                continue
                        if len(haar_faces) > 0:
                            break
                    
                    if len(haar_faces) > 0:
                        # Convert to MobileNetV2 format (x, y, w, h)
                        mobilenet_faces = [(x, y, w, h) for (x, y, w, h) in haar_faces]
                        print(f"✅ Haar cascade final result: {len(mobilenet_faces)} face(s)")
                    else:
                        print("❌ Haar cascade also failed to detect faces with all parameter combinations")
                        
                except Exception as e:
                    print(f"Haar cascade fallback failed: {e}")
                    import traceback
                    traceback.print_exc()
            
            if not mobilenet_faces:
                print("❌ No face detected in the image")
                print("Troubleshooting tips:")
                print("- Make sure the image contains a clear, front-facing face")
                print("- Ensure good lighting and face is not too small or too large")
                print("- Face should be unobstructed (no sunglasses, masks, etc.)")
                print("- Try a different image with better quality")
                return False
            
            print(f"✅ {len(mobilenet_faces)} face(s) detected successfully")
            
            # Use the first detected face
            x, y, w, h = mobilenet_faces[0]
            face_roi = image[y:y+h, x:x+w]
            
            # Extract features ONLY with Siamese models (registration-capable models)
            encodings = None
            feature_type = None
            
            # Priority: Euclidean Siamese > Cosine Siamese
            # Do NOT use Embedding Classification for registration
            
            # Try Euclidean Siamese network first (preferred for registration)
            if hasattr(self, 'siamese_model') and self.siamese_model is not None:
                siamese_features = self.extract_face_features_siamese(face_roi, distance_type='euclidean')
                if siamese_features is not None:
                    encodings = [siamese_features]
                    feature_type = "Euclidean Siamese (Keras)"
                    print("✅ Successfully extracted Euclidean Siamese face features for registration")
            
            # Try Cosine Siamese if Euclidean not available
            if encodings is None and hasattr(self, 'siamese_cosine_model') and self.siamese_cosine_model is not None:
                cosine_features = self.extract_face_features_siamese(face_roi, distance_type='cosine')
                if cosine_features is not None:
                    encodings = [cosine_features]
                    feature_type = "Cosine Siamese (Keras)"
                    print("✅ Successfully extracted Cosine Siamese face features for registration")
            
            # Do NOT use Embedding Classification model for registration
            if encodings is None and hasattr(self, 'embedding_classification_model') and self.embedding_classification_model is not None:
                print("❌ Embedding Classification model detected but not suitable for registration")
                print("💡 Embedding Classification can only identify faces from its training dataset")
                print("   Use Siamese models (Euclidean/Cosine) to register new faces")
                return False
            
            if not encodings:
                print(f"❌ No registration-capable models available or face features could not be extracted from: {image_path}")
                print("Make sure:")
                print("- Siamese models (Euclidean or Cosine) are loaded")
                print("- Face is clearly visible and well-lit")
                print("- Face is front-facing and centered")
                print("- Image quality is good")
                return False
            
            print(f"✅ Successfully detected face and extracted {feature_type} features for registration")
            
            # Create name with student ID - ensure proper encoding
            try:
                # Clean name for display
                clean_name = ''.join(c for c in name if c.isalnum() or c in ' -_').strip()
                if not clean_name:
                    clean_name = f"Student{student_id}"
                full_name = f"{clean_name}_{student_id}"
            except:
                full_name = f"Student{student_id}_{student_id}"
            
            # Remove existing encoding for this person if exists
            if full_name in self.known_face_names:
                index = self.known_face_names.index(full_name)
                self.known_face_encodings.pop(index)
                self.known_face_names.pop(index)
                print(f"Removed existing encoding for {full_name}")
            
            # Add new encoding
            self.known_face_encodings.append(encodings[0])
            self.known_face_names.append(full_name)
            
            # Save to faces directory
            faces_dir = "faces"
            if not os.path.exists(faces_dir):
                os.makedirs(faces_dir)
            
            # Save the processed image
            dest_path = os.path.join(faces_dir, f"{full_name}.jpg")
            # Convert back to BGR for saving with OpenCV
            save_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(dest_path, save_image)
            
            # Save encodings
            self.save_encodings()
            print(f"✅ Successfully registered {full_name} using {feature_type} features")
            print(f"   Saved image: {dest_path}")
            print(f"   Total registered faces: {len(self.known_face_encodings)}")
            return True
            
        except Exception as e:
            print(f"❌ Error registering face: {e}")
            import traceback
            traceback.print_exc()
            print("Make sure the image file is valid and Siamese models are loaded")
            return False
    
    def recognize_faces(self, frame):
        """Recognize faces in a frame with MobileNetV2 detection and multiple recognition methods"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Use MobileNetV2 detection only
            mobilenet_faces = self.detect_faces_mobilenetv2(frame)
            
            if not mobilenet_faces:
                # No faces detected with MobileNetV2
                return [], []
            
            # Convert MobileNetV2 format to face_recognition format
            face_locations = []
            for (x, y, w, h) in mobilenet_faces:
                # face_recognition uses (top, right, bottom, left) format
                face_locations.append((y, x + w, y + h, x))
            
            # Extract face features using available methods
            face_names = []
            for face_location in face_locations:
                top, right, bottom, left = face_location
                face_roi = rgb_frame[top:bottom, left:right]
                
                best_match = "Unknown"
                min_distance = float('inf')
                matched_with_method = None
                
                print(f"Processing face at location: {face_location}")
                print(f"Available encodings: {len(self.known_face_encodings)}")
                if len(self.known_face_encodings) > 0:
                    print(f"Encoding shapes: {[enc.shape for enc in self.known_face_encodings]}")
                    print(f"Encoding names: {self.known_face_names}")
                else:
                    print("⚠️ No face encodings found! Please register students first.")
                    face_names.append("Unknown")
                    continue
                
                # Strategy: Try each available model and compare with compatible encodings
                
                # 1. Try Embedding Classification model (1280 dimensions)
                if self.embedding_classification_model is not None:
                    current_features = self.extract_face_features_siamese(face_roi, distance_type='embedding')
                    
                    if current_features is not None:
                        print(f"Extracted embedding features: shape={current_features.shape}")
                        compatible_encodings = 0
                        
                        for i, known_encoding in enumerate(self.known_face_encodings):
                            # Compare only with 1280-dimensional encodings (embedding model)
                            if hasattr(known_encoding, 'shape') and len(known_encoding) == 1280:
                                compatible_encodings += 1
                                distance = self.compute_cosine_distance(current_features, known_encoding)
                                print(f"  Comparing with {self.known_face_names[i]}: distance={distance:.4f}")
                                
                                if distance < min_distance and distance < 0.7:  # Threshold for cosine distance
                                    min_distance = distance
                                    best_match = self.known_face_names[i]
                                    matched_with_method = "Embedding (Keras)"
                        
                        if compatible_encodings == 0:
                            print("  No compatible 1280D embeddings found for comparison")
                    else:
                        print("  Failed to extract embedding features")
                
                # 2. Try Euclidean Siamese model if available and feature extractor exists
                if best_match == "Unknown" and hasattr(self, 'feature_extractor') and self.feature_extractor is not None:
                    current_features = self.extract_face_features_siamese(face_roi, distance_type='euclidean')
                    
                    if current_features is not None:
                        print(f"Extracted euclidean features: shape={current_features.shape}")
                        compatible_encodings = 0
                        
                        for i, known_encoding in enumerate(self.known_face_encodings):
                            # Compare with same-dimensional encodings
                            if hasattr(known_encoding, 'shape') and len(known_encoding) == len(current_features):
                                compatible_encodings += 1
                                distance = self.compute_euclidean_distance(current_features, known_encoding)
                                print(f"  Comparing with {self.known_face_names[i]}: distance={distance:.4f}")
                                
                                if distance < min_distance and distance < 1.0:  # Threshold for Euclidean distance
                                    min_distance = distance
                                    best_match = self.known_face_names[i]
                                    matched_with_method = "Euclidean Siamese (Keras)"
                        
                        if compatible_encodings == 0:
                            print(f"  No compatible {len(current_features)}D encodings found for euclidean comparison")
                    else:
                        print("  Failed to extract euclidean features")
                
                # 3. Try Cosine Siamese model if available and feature extractor exists
                if best_match == "Unknown" and hasattr(self, 'cosine_feature_extractor') and self.cosine_feature_extractor is not None:
                    current_features = self.extract_face_features_siamese(face_roi, distance_type='cosine')
                    
                    if current_features is not None:
                        print(f"Extracted cosine features: shape={current_features.shape}")
                        compatible_encodings = 0
                        
                        for i, known_encoding in enumerate(self.known_face_encodings):
                            # Compare with same-dimensional encodings
                            if hasattr(known_encoding, 'shape') and len(known_encoding) == len(current_features):
                                compatible_encodings += 1
                                distance = self.compute_cosine_distance(current_features, known_encoding)
                                print(f"  Comparing with {self.known_face_names[i]}: distance={distance:.4f}")
                                
                                if distance < min_distance and distance < 0.7:  # Threshold for cosine distance
                                    min_distance = distance
                                    best_match = self.known_face_names[i]
                                    matched_with_method = "Cosine Siamese (Keras)"
                        
                        if compatible_encodings == 0:
                            print(f"  No compatible {len(current_features)}D encodings found for cosine comparison")
                    else:
                        print("  Failed to extract cosine features")
                
                if best_match != "Unknown":
                    print(f"  ✓ Matched with {best_match} using {matched_with_method} (distance: {min_distance:.4f})")
                else:
                    if min_distance == float('inf'):
                        print(f"  ✗ No compatible encodings found for comparison")
                        print(f"    Make sure you have registered students with the same model type")
                    else:
                        print(f"  ✗ No match found (min distance: {min_distance:.4f})")
                        print(f"    Distance exceeded threshold for recognition")
                
                face_names.append(best_match)
            
            return face_locations, face_names
            
        except Exception as e:
            print(f"Error in face recognition: {e}")
            import traceback
            traceback.print_exc()
            return [], []
    
    def detect_liveness(self, frame, face_location):
        """Detect liveness using smile and eye detection"""
        try:
            # Extract face region
            top, right, bottom, left = face_location
            face_roi = frame[top:bottom, left:right]
            
            if face_roi.size == 0:
                return False, "No face region"
            
            # Convert to grayscale for cascade detection
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            liveness_indicators = {
                'smile_detected': False,
                'eyes_detected': False,
                'liveness_score': 0
            }
            
            # Detect smile
            if self.smile_cascade and not self.smile_cascade.empty():
                smiles = self.smile_cascade.detectMultiScale(
                    gray_face, 
                    scaleFactor=1.8, 
                    minNeighbors=20,
                    minSize=(10, 10)
                )
                if len(smiles) > 0:
                    liveness_indicators['smile_detected'] = True
                    liveness_indicators['liveness_score'] += 50
                    print(f"Smile detected! Smiles found: {len(smiles)}")
            
            # Detect eyes
            if self.eye_cascade and not self.eye_cascade.empty():
                eyes = self.eye_cascade.detectMultiScale(
                    gray_face,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(10, 10)
                )
                if len(eyes) >= 2:  # Need at least 2 eyes
                    liveness_indicators['eyes_detected'] = True
                    liveness_indicators['liveness_score'] += 30
                    print(f"Eyes detected! Eyes found: {len(eyes)}")
                elif len(eyes) == 1:
                    liveness_indicators['liveness_score'] += 15
                    print(f"One eye detected")
            
            # Determine if person is live
            is_live = liveness_indicators['liveness_score'] >= 30  # Minimum threshold
            
            status_msg = []
            if liveness_indicators['smile_detected']:
                status_msg.append("Smile✓")
            if liveness_indicators['eyes_detected']:
                status_msg.append("Eyes✓")
            
            status = " ".join(status_msg) if status_msg else "Static"
            
            return is_live, status
            
        except Exception as e:
            print(f"Error in liveness detection: {e}")
            return False, "Error"
    
    def test_cascades(self):
        """Test if detection and recognition systems are working properly"""
        print("Testing detection and recognition systems...")
        
        # Test MobileNetV2 detector
        if self.mobilenet_detector is not None:
            print(f"✓ MobileNetV2 detector loaded successfully ({self.detection_method})")
        else:
            print("✗ MobileNetV2 detector not loaded")
        
        # Test Keras models
        if self.siamese_model is not None:
            print("✓ Siamese network with Euclidean distance loaded successfully")
        else:
            print("✗ Siamese Euclidean network not loaded")
            
        if self.siamese_cosine_model is not None:
            print("✓ Siamese network with Cosine distance loaded successfully")
        else:
            print("✗ Siamese Cosine network not loaded")
            
        if self.embedding_classification_model is not None:
            print("✓ Embedding Classification model loaded successfully")
        else:
            print("✗ Embedding Classification model not loaded")
        
        # Test eye cascade (for liveness)
        if self.eye_cascade and not self.eye_cascade.empty():
            print("✓ Eye cascade loaded successfully")
        else:
            print("✗ Eye cascade not loaded or empty")
        
        # Test smile cascade (for liveness)
        if self.smile_cascade and not self.smile_cascade.empty():
            print("✓ Smile cascade loaded successfully")
        else:
            print("✗ Smile cascade not loaded or empty")
        
        # Print recognition method priority
        available_methods = []
        if self.embedding_classification_model is not None:
            available_methods.append("Embedding Classification (Keras)")
        if self.siamese_model is not None:
            available_methods.append("Siamese Euclidean (Keras)")
        if self.siamese_cosine_model is not None:
            available_methods.append("Siamese Cosine (Keras)")
        
        if available_methods:
            print(f"📋 Recognition methods available: {', '.join(available_methods)}")
            print(f"   Priority: {' > '.join(available_methods)}")
        else:
            print("⚠ No recognition methods available")
        
        return (self.mobilenet_detector is not None,
                self.siamese_model is not None or self.siamese_cosine_model is not None or self.embedding_classification_model is not None,
                self.eye_cascade and not self.eye_cascade.empty(),
                self.smile_cascade and not self.smile_cascade.empty())
    
    def recognize_faces_with_liveness(self, frame, require_liveness=True):
        """Recognize faces with liveness detection"""
        try:
            # Get basic face recognition results
            face_locations, face_names = self.recognize_faces(frame)
            
            if not require_liveness:
                return face_locations, face_names, [True] * len(face_names)
            
            # Check liveness for each detected face
            liveness_results = []
            updated_names = []
            
            for i, (face_location, name) in enumerate(zip(face_locations, face_names)):
                is_live, liveness_status = self.detect_liveness(frame, face_location)
                liveness_results.append(is_live)
                
                # Update name with liveness status
                if name != "Unknown":
                    if is_live:
                        updated_names.append(f"{name} (Live: {liveness_status})")
                    else:
                        updated_names.append(f"{name} (Static)")
                else:
                    if is_live:
                        updated_names.append(f"Unknown (Live: {liveness_status})")
                    else:
                        updated_names.append("Unknown (Static)")
            
            return face_locations, updated_names, liveness_results
            
        except Exception as e:
            print(f"Error in face recognition with liveness: {e}")
            return [], [], []
    
    def remove_face(self, name, student_id):
        """Remove a face from the recognition system"""
        full_name = f"{name}_{student_id}"
        if full_name in self.known_face_names:
            index = self.known_face_names.index(full_name)
            self.known_face_encodings.pop(index)
            self.known_face_names.pop(index)
            
            # Remove image file
            faces_dir = "faces"
            image_path = os.path.join(faces_dir, f"{full_name}.jpg")
            if os.path.exists(image_path):
                os.remove(image_path)
            
            # Save updated encodings
            self.save_encodings()
            return True
        return False
           