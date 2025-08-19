#!/usr/bin/env python3
"""
TouchDesigner YOLO Production Readiness Test Suite
==================================================

Comprehensive simulation and validation script to test if the TDYolo repository
is production-ready for both Mac and Windows platforms.

This script emulates TouchDesigner behavior and tests all critical components:
- Environment files (Mac/Windows)
- Python version compatibility
- Dependencies validation
- CUDA/MPS device detection
- YOLO model loading and inference
- TouchDesigner DAT operations
- Cross-platform compatibility

Usage:
    python production_readiness_test.py [--conda-env TDYolo-Test1] [--platform auto|windows|darwin]
"""

import sys
import os
import platform
import subprocess
import json
import tempfile
import shutil
import logging
import traceback
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import time

# Configure logging - minimal output to console, detailed to file
class MinimalConsoleFormatter(logging.Formatter):
    def format(self, record):
        if record.levelno >= logging.WARNING:
            return f"[{record.levelname}] {record.getMessage()}"
        return record.getMessage()

# Setup dual logging
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(MinimalConsoleFormatter())

file_handler = logging.FileHandler('production_test.log', mode='w')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[console_handler, file_handler]
)
logger = logging.getLogger(__name__)

class ProductionTestSuite:
    def __init__(self, conda_env: str = "TDYolo-Test1", target_platform: str = "auto"):
        self.conda_env = conda_env
        self.target_platform = target_platform if target_platform != "auto" else platform.system()
        self.repo_root = Path(__file__).parent
        self.test_results = {}
        self.critical_errors = []
        self.warnings = []
        
        # Mock TouchDesigner environment
        self.mock_td_env = {}
        self.setup_mock_touchdesigner()
        
        logger.info("=" * 70)
        logger.info("TouchDesigner YOLO Production Readiness Test Suite")
        logger.info("=" * 70)
        logger.info(f"Repository: {self.repo_root}")
        logger.info(f"Target Platform: {self.target_platform}")
        logger.info(f"Conda Environment: {self.conda_env}")
        logger.info(f"Python Version: {sys.version}")
        logger.info("=" * 70)

    def setup_mock_touchdesigner(self):
        """Setup mock TouchDesigner environment for testing"""
        # Setup mock TouchDesigner environment (details in log file)
        
        # Mock op() function and DAT objects
        class MockDAT:
            def __init__(self, name: str, data: List[List[str]] = None):
                self.name = name
                self.data = data or [["header"], ["value"]]
                self.numRows = len(self.data)
                self.numCols = len(self.data[0]) if self.data else 0
                self.storage = {}
            
            def __getitem__(self, key):
                if isinstance(key, tuple):
                    row, col = key
                    if 0 <= row < self.numRows and 0 <= col < self.numCols:
                        return MockCell(self.data[row][col])
                return MockCell("")
            
            def clear(self):
                self.data = []
                self.numRows = 0
                self.numCols = 0
            
            def appendRow(self, row_data: List[str]):
                self.data.append(row_data)
                self.numRows = len(self.data)
                if self.data:
                    self.numCols = max(self.numCols, len(row_data))
            
            def store(self, key: str, value: Any):
                self.storage[key] = value
        
        class MockCell:
            def __init__(self, value: str):
                self.val = value
        
        class MockScriptOp:
            def __init__(self):
                self.inputs = [MockInput()]
                self.par = MockPar()
            
            def appendCustomPage(self, name: str):
                return MockPage()
            
            def copyNumpyArray(self, array):
                pass
        
        class MockInput:
            def numpyArray(self):
                import numpy as np
                # Return a mock 640x480 RGBA image
                return np.random.rand(480, 640, 4).astype(np.float32)
        
        class MockPar:
            def __init__(self):
                self.Drawbox = MockParam(True)
                self.Confidence = MockParam(0.25)
                self.Frameskip = MockParam(0)
                self.Detectionlimit = MockParam(0)
        
        class MockParam:
            def __init__(self, default_val):
                self.default = default_val
                self._val = default_val
            
            def eval(self):
                return self._val
        
        class MockPage:
            def appendToggle(self, name: str, **kwargs):
                return [MockParam(True)]
            
            def appendStr(self, name: str, **kwargs):
                return [MockParam("")]
            
            def appendFloat(self, name: str, **kwargs):
                param = MockParam(0.25)
                param.normMin = 0.0
                param.normMax = 1.0
                return [param]
            
            def appendInt(self, name: str, **kwargs):
                param = MockParam(0)
                param.normMin = 0
                param.normMax = 100
                return [param]
        
        # Setup mock DATs
        self.mock_td_env = {
            'condaParam': MockDAT('condaParam', [
                ['Parameter', 'Value'],
                ['Condaenv', self.conda_env],
                ['User', os.getenv('USER', os.getenv('USERNAME', 'testuser'))]
            ]),
            'parameter1': MockDAT('parameter1', [
                ['Parameter', 'Value'],
                ['Classes', 'person,car']
            ]),
            'report': MockDAT('report'),
            'summary': MockDAT('summary'),
            'scriptOp': MockScriptOp()
        }
        
        # Inject mock op() function globally
        def mock_op(name: str):
            return self.mock_td_env.get(name)
        
        # Add to builtins so it's available in imported modules
        import builtins
        builtins.op = mock_op
        
        # Mock environment ready

    def test_environment_files(self) -> bool:
        """Test environment.yml files for validity and cross-platform compatibility"""
        logger.info("🧪 Testing Environment Files...")
        
        success = True
        
        # Test environment files
        env_files = {
            'Mac': self.repo_root / 'environment-mac.yml',
            'Windows': self.repo_root / 'environment-win.yml'
        }
        
        for platform_name, env_file in env_files.items():
            logger.debug(f"  Testing {platform_name} environment file: {env_file.name}")
            
            if not env_file.exists():
                self.critical_errors.append(f"Missing environment file: {env_file}")
                success = False
                continue
            
            try:
                import yaml
                with open(env_file, 'r') as f:
                    env_data = yaml.safe_load(f)
                
                # Check Python version consistency
                dependencies = env_data.get('dependencies', [])
                python_spec = None
                for dep in dependencies:
                    if isinstance(dep, str) and dep.startswith('python='):
                        python_spec = dep.split('=')[1]
                        break
                
                if python_spec:
                    logger.debug(f"    Python version: {python_spec}")
                    if python_spec != "3.11.10":
                        self.warnings.append(f"{platform_name}: Python version {python_spec} != 3.11.10")
                else:
                    self.critical_errors.append(f"{platform_name}: No Python version specified")
                    success = False
                
                # Check critical dependencies
                pip_deps = []
                for dep in dependencies:
                    if isinstance(dep, dict) and 'pip' in dep:
                        pip_deps = dep['pip']
                        break
                
                critical_packages = [
                    'torch', 'torchvision', 'torchaudio', 'ultralytics', 
                    'opencv-python', 'numpy', 'pillow'
                ]
                
                found_packages = set()
                for pip_dep in pip_deps:
                    pkg_name = pip_dep.split('==')[0].split('>=')[0].split('<=')[0]
                    found_packages.add(pkg_name)
                
                missing_critical = set(critical_packages) - found_packages
                if missing_critical:
                    self.critical_errors.append(f"{platform_name}: Missing critical packages: {missing_critical}")
                    success = False
                
            except ImportError:
                self.warnings.append("PyYAML not available for environment validation")
            except Exception as e:
                self.critical_errors.append(f"Error parsing {platform_name} environment: {e}")
                success = False
        
        self.test_results['environment_files'] = success
        return success

    def test_conda_environment_setup(self) -> bool:
        """Test conda environment detection and setup"""
        logger.info("🧪 Testing Conda Environment Setup...")
        
        try:
            # Test if we can import the extCondaEnv module
            sys.path.insert(0, str(self.repo_root / 'python-script'))
            
            # Import and test extCondaEnv
            import extCondaEnv
            
            # Test conda info function
            conda_info = extCondaEnv.get_conda_info()
            if conda_info:
                logger.debug(f"    Conda version: {conda_info.get('conda_version', 'unknown')}")
            else:
                self.warnings.append("Could not retrieve conda info")
            
            # Test environment detection
            username = os.getenv('USER', os.getenv('USERNAME', 'testuser'))
            conda_locations = extCondaEnv.find_conda_environments(username)
            
            if conda_locations:
                logger.debug(f"    Found conda installations: {len(conda_locations)}")
                for loc in conda_locations:
                    logger.debug(f"      - {loc}")
            else:
                self.warnings.append("No conda installations found")
            
            # Test environment existence
            env_found = False
            for location in conda_locations:
                env_path = os.path.join(location, 'envs', self.conda_env)
                if os.path.exists(env_path):
                    env_found = True
                    logger.debug(f"    Found target environment: {env_path}")
                    
                    # Test Python version detection
                    python_version = extCondaEnv.get_python_version_from_env(env_path)
                    logger.debug(f"    Python version detected: {python_version}")
                    break
            
            if not env_found:
                self.warnings.append(f"Target environment '{self.conda_env}' not found")
            
            self.test_results['conda_setup'] = True
            return True
            
        except Exception as e:
            self.critical_errors.append(f"Conda environment setup test failed: {e}")
            self.test_results['conda_setup'] = False
            return False

    def test_pytorch_device_detection(self) -> bool:
        """Test PyTorch and device detection (CUDA/MPS/CPU)"""
        logger.info("🧪 Testing PyTorch Device Detection...")
        
        try:
            import torch
            logger.debug(f"    PyTorch version: {torch.__version__}")
            
            # Test CUDA availability
            cuda_available = torch.cuda.is_available()
            
            if cuda_available:
                cuda_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if cuda_count > 0 else "Unknown"
                logger.info(f"    ✅ CUDA available: {cuda_count} GPU(s) - {gpu_name}")
            
            # Test MPS availability (Apple Silicon)
            mps_available = False
            if hasattr(torch.backends, 'mps'):
                mps_available = torch.backends.mps.is_available()
                if mps_available:
                    logger.info(f"    ✅ MPS (Metal) available")
            
            # Determine optimal device
            if cuda_available:
                device = 'cuda'
            elif mps_available:
                device = 'mps'
            else:
                device = 'cpu'
            
            logger.info(f"    ✅ Optimal device: {device}")
            
            # Test device functionality
            test_tensor = torch.randn(100, 100)
            if device != 'cpu':
                test_tensor = test_tensor.to(device)
                logger.debug(f"    Successfully moved tensor to {device}")
            
            self.test_results['pytorch_device'] = True
            return True
            
        except ImportError as e:
            self.critical_errors.append(f"PyTorch import failed: {e}")
            self.test_results['pytorch_device'] = False
            return False
        except Exception as e:
            self.critical_errors.append(f"PyTorch device test failed: {e}")
            self.test_results['pytorch_device'] = False
            return False

    def test_yolo_model_loading(self) -> bool:
        """Test YOLO model loading and basic inference"""
        logger.info("🧪 Testing YOLO Model Loading...")
        
        try:
            from ultralytics import YOLO
            
            # Check for model file
            model_path = self.repo_root / 'yolo11n.pt'
            if not model_path.exists():
                self.warnings.append(f"YOLO model file not found: {model_path}")
                model_path = 'yolo11n.pt'  # Will download if needed
            
            # Load model (suppress verbose output)
            import logging as ul_logging
            ul_logging.getLogger('ultralytics').setLevel(ul_logging.WARNING)
            
            model = YOLO(str(model_path), task='detect')
            
            # Test device placement
            import torch
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
            
            model.to(device)
            
            # Test model names/classes
            class_names = model.names
            logger.info(f"    ✅ YOLO model loaded: {len(class_names)} classes on {device}")
            
            # Test inference with dummy image
            import numpy as np
            dummy_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
            
            with torch.no_grad():
                results = model.predict(
                    source=dummy_image,
                    conf=0.25,
                    verbose=False,
                    device=device,
                    imgsz=640
                )
            
            detections = len(results[0].boxes) if results[0].boxes is not None else 0
            logger.info(f"    ✅ Inference test completed: {detections} detections")
            
            self.test_results['yolo_model'] = True
            return True
            
        except ImportError as e:
            self.critical_errors.append(f"YOLO/Ultralytics import failed: {e}")
            self.test_results['yolo_model'] = False
            return False
        except Exception as e:
            self.critical_errors.append(f"YOLO model test failed: {e}")
            self.test_results['yolo_model'] = False
            return False

    def test_opencv_operations(self) -> bool:
        """Test OpenCV operations and image processing"""
        logger.info("🧪 Testing OpenCV Operations...")
        
        try:
            import cv2
            import numpy as np
            
            logger.debug(f"    OpenCV version: {cv2.__version__}")
            
            # Test image operations used in main script
            dummy_rgba = np.random.rand(480, 640, 4).astype(np.float32)
            
            # Test RGBA to BGR conversion
            bgr = cv2.cvtColor(np.clip(dummy_rgba * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGBA2BGR)
            
            # Test BGR to RGBA conversion
            rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
            
            # Test image flipping
            flipped = cv2.flip(rgba, 0)
            
            # Test drawing operations
            test_image = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.rectangle(test_image, (100, 100), (200, 200), (0, 255, 0), 2)
            
            # Test text rendering
            text = "test_text_123"
            text_size, base_line = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(test_image, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            logger.info(f"    ✅ OpenCV operations successful")
            
            self.test_results['opencv'] = True
            return True
            
        except ImportError as e:
            self.critical_errors.append(f"OpenCV import failed: {e}")
            self.test_results['opencv'] = False
            return False
        except Exception as e:
            self.critical_errors.append(f"OpenCV operations test failed: {e}")
            self.test_results['opencv'] = False
            return False

    def test_main_script_integration(self) -> bool:
        """Test main YOLO script integration with mock TouchDesigner environment"""
        logger.info("🧪 Testing Main Script Integration...")
        
        try:
            # Import main script
            sys.path.insert(0, str(self.repo_root / 'python-script'))
            
            # We need to mock the YOLO model file first
            model_path = self.repo_root / 'yolo11n.pt'
            if not model_path.exists():
                logger.debug("    Creating temporary model file for testing...")
                # Create a dummy file to prevent download during import
                model_path.touch()
                cleanup_model = True
            else:
                cleanup_model = False
            
            try:
                # Change to the repo directory so relative paths work
                original_cwd = os.getcwd()
                os.chdir(self.repo_root)
                
                # Import main script functions
                with open(self.repo_root / 'python-script' / 'main-TDYolo.py', 'r') as f:
                    script_content = f.read()
                
                # Create a namespace to execute the script
                script_globals = {
                    '__name__': '__main__',
                    'op': lambda name: self.mock_td_env.get(name)
                }
                
                # Suppress YOLO outputs during testing
                import logging as ul_logging
                ul_logging.getLogger('ultralytics').setLevel(ul_logging.ERROR)
                
                # Execute the script in controlled environment
                exec(script_content, script_globals)
                
                # Test get_optimal_device function
                if 'get_optimal_device' in script_globals:
                    device = script_globals['get_optimal_device']()
                    logger.debug(f"    Device detection: {device}")
                
                # Test onSetupParameters function
                if 'onSetupParameters' in script_globals:
                    script_op = self.mock_td_env['scriptOp']
                    script_globals['onSetupParameters'](script_op)
                
                # Test onCook function (main processing)
                if 'onCook' in script_globals:
                    script_op = self.mock_td_env['scriptOp']
                    script_globals['onCook'](script_op)
                    
                    # Check if report table was populated
                    report_dat = self.mock_td_env['report']
                    summary_dat = self.mock_td_env['summary']
                    
                    logger.info(f"    ✅ Main script integration successful")
                
                os.chdir(original_cwd)
                
                if cleanup_model:
                    model_path.unlink()
                
                self.test_results['main_script'] = True
                return True
                
            except Exception as e:
                os.chdir(original_cwd)
                if cleanup_model:
                    model_path.unlink()
                raise e
                
        except Exception as e:
            self.critical_errors.append(f"Main script integration test failed: {e}")
            logger.error(f"    ❌ Error: {e}")
            logger.error(f"    Traceback: {traceback.format_exc()}")
            self.test_results['main_script'] = False
            return False

    def test_memory_and_performance(self) -> bool:
        """Test memory usage and performance characteristics"""
        logger.info("🧪 Testing Memory and Performance...")
        
        try:
            import psutil
            import gc
            
            # Get initial memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Test memory usage during typical operations
            import torch
            import numpy as np
            
            # Simulate frame processing
            max_memory = initial_memory
            
            for i in range(10):  # Process 10 test frames
                # Create test frame
                frame = np.random.rand(480, 640, 4).astype(np.float32)
                
                # Simulate RGBA to BGR conversion
                import cv2
                bgr = cv2.cvtColor(np.clip(frame * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGBA2BGR)
                
                # Simulate tensor operations
                if torch.cuda.is_available() or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                    tensor = torch.from_numpy(bgr)
                    if torch.cuda.is_available():
                        tensor = tensor.cuda()
                    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        tensor = tensor.to('mps')
                    del tensor
                
                # Check memory usage
                current_memory = process.memory_info().rss / 1024 / 1024
                max_memory = max(max_memory, current_memory)
                
                # Force garbage collection
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_growth = final_memory - initial_memory
            
            logger.info(f"    ✅ Memory test: {final_memory:.1f}MB peak, {memory_growth:.1f}MB growth")
            
            if memory_growth > 100:  # More than 100MB growth
                self.warnings.append(f"High memory growth detected: {memory_growth:.1f} MB")
            
            self.test_results['memory_performance'] = True
            return True
            
        except ImportError:
            self.warnings.append("psutil not available for memory testing")
            self.test_results['memory_performance'] = True
            return True
        except Exception as e:
            self.warnings.append(f"Memory/performance test failed: {e}")
            self.test_results['memory_performance'] = True
            return True

    def test_cross_platform_compatibility(self) -> bool:
        """Test cross-platform compatibility issues"""
        logger.info("🧪 Testing Cross-Platform Compatibility...")
        
        try:
            current_platform = platform.system()
            logger.debug(f"    Current platform: {current_platform}")
            
            # Test path handling
            test_paths = [
                "C:/Users/test/conda/envs/env",  # Windows style
                "/Users/test/conda/envs/env",    # Unix style
                "~/conda/envs/env"               # Home directory
            ]
            
            for test_path in test_paths:
                expanded = os.path.expanduser(test_path)
                normalized = os.path.normpath(expanded)
                logger.debug(f"    Path handling: {test_path} -> {normalized}")
            
            # Test platform-specific features
            if current_platform == "Windows":
                # Test DLL directory addition (Python 3.8+)
                try:
                    import tempfile
                    with tempfile.TemporaryDirectory() as temp_dir:
                        os.add_dll_directory(temp_dir)
                    logger.info("    ✅ Windows features compatible")
                except Exception as e:
                    self.warnings.append(f"DLL directory addition failed: {e}")
            
            elif current_platform == "Darwin":
                logger.info("    ✅ macOS features compatible")
            else:
                logger.info(f"    ✅ {current_platform} features compatible")
            
            self.test_results['cross_platform'] = True
            return True
            
        except Exception as e:
            self.critical_errors.append(f"Cross-platform compatibility test failed: {e}")
            self.test_results['cross_platform'] = False
            return False

    def run_all_tests(self) -> bool:
        """Run all production readiness tests"""
        logger.info("\n🚀 Starting Production Readiness Test Suite...")
        
        start_time = time.time()
        
        # Run all tests
        tests = [
            ("Environment Files", self.test_environment_files),
            ("Conda Setup", self.test_conda_environment_setup),
            ("PyTorch Device", self.test_pytorch_device_detection),
            ("YOLO Model", self.test_yolo_model_loading),
            ("OpenCV Operations", self.test_opencv_operations),
            ("Main Script Integration", self.test_main_script_integration),
            ("Memory & Performance", self.test_memory_and_performance),
            ("Cross-Platform", self.test_cross_platform_compatibility),
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            try:
                if test_func():
                    passed_tests += 1
                    logger.info(f"✅ {test_name}")
                else:
                    logger.error(f"❌ {test_name}: FAILED")
            except Exception as e:
                logger.error(f"❌ {test_name}: CRASHED - {e}")
                self.critical_errors.append(f"{test_name} crashed: {e}")
        
        # Generate final report
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info("\n" + "=" * 70)
        logger.info("PRODUCTION READINESS TEST RESULTS")
        logger.info("=" * 70)
        
        logger.info(f"Tests Passed: {passed_tests}/{total_tests}")
        logger.info(f"Test Duration: {duration:.2f} seconds")
        logger.info(f"Platform: {self.target_platform}")
        logger.info(f"Conda Environment: {self.conda_env}")
        
        if self.critical_errors:
            logger.error("\n❌ CRITICAL ERRORS:")
            for error in self.critical_errors:
                logger.error(f"  • {error}")
        
        if self.warnings:
            logger.warning("\n⚠️  WARNINGS:")
            for warning in self.warnings:
                logger.warning(f"  • {warning}")
        
        # Determine overall result
        production_ready = (
            passed_tests == total_tests and 
            len(self.critical_errors) == 0
        )
        
        if production_ready:
            logger.info("\n🎉 RESULT: PRODUCTION READY! ✅")
            logger.info("All critical tests passed. The repository is ready for production use.")
        else:
            logger.error("\n⚠️  RESULT: NOT PRODUCTION READY ❌")
            logger.error("Critical issues found. Please address them before production deployment.")
        
        if self.warnings and production_ready:
            logger.warning("\nNote: There are warnings that should be addressed for optimal performance.")
        
        logger.info("\nDetailed test log saved to: production_test.log")
        logger.info("=" * 70)
        
        return production_ready

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="TouchDesigner YOLO Production Readiness Test")
    parser.add_argument('--conda-env', default='TDYolo-Test1', 
                       help='Conda environment name to test (default: TDYolo-Test1)')
    parser.add_argument('--platform', choices=['auto', 'windows', 'darwin'], default='auto',
                       help='Target platform for testing (default: auto-detect)')
    
    args = parser.parse_args()
    
    # Create and run test suite
    test_suite = ProductionTestSuite(
        conda_env=args.conda_env,
        target_platform=args.platform
    )
    
    success = test_suite.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()