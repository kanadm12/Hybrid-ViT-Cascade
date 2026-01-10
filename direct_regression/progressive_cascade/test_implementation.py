"""
Test Suite for Progressive Cascade Implementation
Validates all components before training
"""
import torch
import sys
import json
from pathlib import Path

# Color codes for pretty printing
GREEN = '\033[92m'
RED = '\033[91m'
BLUE = '\033[94m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def print_header(text):
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{text:^60}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

def print_test(name, passed):
    status = f"{GREEN}✓ PASS{RESET}" if passed else f"{RED}✗ FAIL{RESET}"
    print(f"  {status} {name}")
    return passed

def test_imports():
    """Test that all required imports work"""
    print_header("Testing Imports")
    
    tests_passed = 0
    total_tests = 0
    
    # Test model imports
    total_tests += 1
    try:
        from model_progressive import (
            ProgressiveCascadeModel, Stage1Base64, 
            Stage2Refiner128, Stage3Refiner256
        )
        tests_passed += print_test("Model imports", True)
    except Exception as e:
        print_test(f"Model imports: {e}", False)
    
    # Test loss imports
    total_tests += 1
    try:
        from loss_multiscale import (
            MultiScaleLoss, Stage1Loss, Stage2Loss, Stage3Loss,
            compute_psnr, compute_ssim_metric
        )
        tests_passed += print_test("Loss imports", True)
    except Exception as e:
        print_test(f"Loss imports: {e}", False)
    
    # Test utility imports
    total_tests += 1
    try:
        from utils import (
            count_parameters, check_gpu_memory, 
            estimate_memory_usage, validate_config
        )
        tests_passed += print_test("Utility imports", True)
    except Exception as e:
        print_test(f"Utility imports: {e}", False)
    
    return tests_passed, total_tests


def test_model_architecture():
    """Test model instantiation and forward pass"""
    print_header("Testing Model Architecture")
    
    tests_passed = 0
    total_tests = 0
    
    try:
        from model_progressive import ProgressiveCascadeModel
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Test model creation
        total_tests += 1
        try:
            model = ProgressiveCascadeModel().to(device)
            tests_passed += print_test("Model instantiation", True)
        except Exception as e:
            print_test(f"Model instantiation: {e}", False)
            return tests_passed, total_tests
        
        # Test Stage 1 forward
        total_tests += 1
        try:
            xrays = torch.randn(1, 2, 1, 512, 512).to(device)
            with torch.no_grad():
                out1 = model(xrays, max_stage=1)
            assert out1.shape == (1, 1, 64, 64, 64), f"Wrong shape: {out1.shape}"
            tests_passed += print_test("Stage 1 forward (64³)", True)
        except Exception as e:
            print_test(f"Stage 1 forward: {e}", False)
        
        # Test Stage 2 forward
        total_tests += 1
        try:
            with torch.no_grad():
                out2 = model(xrays, max_stage=2)
            assert out2.shape == (1, 1, 128, 128, 128), f"Wrong shape: {out2.shape}"
            tests_passed += print_test("Stage 2 forward (128³)", True)
        except Exception as e:
            print_test(f"Stage 2 forward: {e}", False)
        
        # Test Stage 3 forward
        total_tests += 1
        try:
            with torch.no_grad():
                out3 = model(xrays, max_stage=3)
            assert out3.shape == (1, 1, 256, 256, 256), f"Wrong shape: {out3.shape}"
            tests_passed += print_test("Stage 3 forward (256³)", True)
        except Exception as e:
            print_test(f"Stage 3 forward: {e}", False)
        
        # Test intermediate outputs
        total_tests += 1
        try:
            with torch.no_grad():
                outputs = model(xrays, return_intermediate=True, max_stage=3)
            assert 'stage1' in outputs and 'stage2' in outputs and 'stage3' in outputs
            tests_passed += print_test("Intermediate outputs", True)
        except Exception as e:
            print_test(f"Intermediate outputs: {e}", False)
        
        # Test stage freezing
        total_tests += 1
        try:
            model.freeze_stage(1)
            stage1_frozen = not any(p.requires_grad for p in model.stage1.parameters())
            model.unfreeze_stage(1)
            stage1_unfrozen = any(p.requires_grad for p in model.stage1.parameters())
            assert stage1_frozen and stage1_unfrozen
            tests_passed += print_test("Stage freeze/unfreeze", True)
        except Exception as e:
            print_test(f"Stage freeze/unfreeze: {e}", False)
        
    except Exception as e:
        print(f"{RED}Fatal error in model testing: {e}{RESET}")
    
    return tests_passed, total_tests


def test_loss_functions():
    """Test all loss functions"""
    print_header("Testing Loss Functions")
    
    tests_passed = 0
    total_tests = 0
    
    try:
        from loss_multiscale import MultiScaleLoss, compute_psnr, compute_ssim_metric
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        loss_fn = MultiScaleLoss().to(device)
        
        # Test Stage 1 loss
        total_tests += 1
        try:
            pred = torch.randn(2, 1, 64, 64, 64).to(device)
            target = torch.randn(2, 1, 64, 64, 64).to(device)
            loss_dict = loss_fn(pred, target, stage=1)
            assert 'total_loss' in loss_dict and 'l1_loss' in loss_dict and 'ssim_loss' in loss_dict
            tests_passed += print_test("Stage 1 loss (L1 + SSIM)", True)
        except Exception as e:
            print_test(f"Stage 1 loss: {e}", False)
        
        # Test Stage 2 loss
        total_tests += 1
        try:
            pred = torch.randn(2, 1, 128, 128, 128).to(device)
            target = torch.randn(2, 1, 128, 128, 128).to(device)
            loss_dict = loss_fn(pred, target, stage=2)
            assert 'vgg_loss' in loss_dict
            tests_passed += print_test("Stage 2 loss (+ VGG)", True)
        except Exception as e:
            print_test(f"Stage 2 loss: {e}", False)
        
        # Test Stage 3 loss
        total_tests += 1
        try:
            pred = torch.randn(2, 1, 256, 256, 256).to(device)
            target = torch.randn(2, 1, 256, 256, 256).to(device)
            xrays = torch.randn(2, 2, 1, 512, 512).to(device)
            loss_dict = loss_fn(pred, target, stage=3, input_xrays=xrays)
            assert 'gradient_loss' in loss_dict and 'drr_loss' in loss_dict
            tests_passed += print_test("Stage 3 loss (+ Grad + DRR)", True)
        except Exception as e:
            print_test(f"Stage 3 loss: {e}", False)
        
        # Test PSNR metric
        total_tests += 1
        try:
            pred = torch.randn(2, 1, 64, 64, 64).to(device)
            target = torch.randn(2, 1, 64, 64, 64).to(device)
            psnr = compute_psnr(pred, target)
            assert isinstance(psnr, float) and psnr > 0
            tests_passed += print_test("PSNR metric", True)
        except Exception as e:
            print_test(f"PSNR metric: {e}", False)
        
        # Test SSIM metric
        total_tests += 1
        try:
            ssim = compute_ssim_metric(pred, target)
            assert isinstance(ssim, float) and 0 <= ssim <= 1
            tests_passed += print_test("SSIM metric", True)
        except Exception as e:
            print_test(f"SSIM metric: {e}", False)
        
    except Exception as e:
        print(f"{RED}Fatal error in loss testing: {e}{RESET}")
    
    return tests_passed, total_tests


def test_configuration():
    """Test configuration file"""
    print_header("Testing Configuration")
    
    tests_passed = 0
    total_tests = 0
    
    config_path = Path(__file__).parent / "config_progressive.json"
    
    # Test config file exists
    total_tests += 1
    if config_path.exists():
        tests_passed += print_test("Config file exists", True)
    else:
        print_test("Config file exists", False)
        return tests_passed, total_tests
    
    # Test config loads
    total_tests += 1
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        tests_passed += print_test("Config loads successfully", True)
    except Exception as e:
        print_test(f"Config loading: {e}", False)
        return tests_passed, total_tests
    
    # Test required keys
    required_keys = ['model', 'training', 'loss', 'data', 'checkpoints']
    for key in required_keys:
        total_tests += 1
        if key in config:
            tests_passed += print_test(f"Config has '{key}' section", True)
        else:
            print_test(f"Config missing '{key}' section", False)
    
    # Test stage configs
    for stage in [1, 2, 3]:
        stage_key = f'stage{stage}'
        total_tests += 1
        if stage_key in config['training']:
            tests_passed += print_test(f"Training config for {stage_key}", True)
        else:
            print_test(f"Training config for {stage_key}", False)
    
    return tests_passed, total_tests


def test_utilities():
    """Test utility functions"""
    print_header("Testing Utilities")
    
    tests_passed = 0
    total_tests = 0
    
    try:
        from utils import count_parameters, estimate_memory_usage
        from model_progressive import ProgressiveCascadeModel
        
        # Test parameter counting
        total_tests += 1
        try:
            model = ProgressiveCascadeModel()
            params = count_parameters(model)
            assert 'total' in params and 'trainable' in params
            tests_passed += print_test("Parameter counting", True)
        except Exception as e:
            print_test(f"Parameter counting: {e}", False)
        
        # Test memory estimation
        total_tests += 1
        try:
            mem_gb = estimate_memory_usage(batch_size=2, resolution=(256, 256, 256))
            assert isinstance(mem_gb, float) and mem_gb > 0
            tests_passed += print_test("Memory estimation", True)
        except Exception as e:
            print_test(f"Memory estimation: {e}", False)
        
    except Exception as e:
        print(f"{RED}Fatal error in utility testing: {e}{RESET}")
    
    return tests_passed, total_tests


def test_gpu_availability():
    """Test GPU availability and memory"""
    print_header("Testing GPU Availability")
    
    tests_passed = 0
    total_tests = 1
    
    cuda_available = torch.cuda.is_available()
    
    if cuda_available:
        num_gpus = torch.cuda.device_count()
        print(f"  {GREEN}✓{RESET} CUDA available with {num_gpus} GPU(s)")
        
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory / 1024**3
            print(f"    GPU {i}: {props.name} ({mem_gb:.1f} GB)")
        
        tests_passed += 1
    else:
        print(f"  {YELLOW}⚠{RESET} CUDA not available (CPU only)")
        print(f"    Training will be very slow on CPU")
    
    return tests_passed, total_tests


def main():
    """Run all tests"""
    print(f"\n{BLUE}{'#'*60}{RESET}")
    print(f"{BLUE}{'Progressive Cascade Test Suite':^60}{RESET}")
    print(f"{BLUE}{'#'*60}{RESET}")
    
    total_passed = 0
    total_tests = 0
    
    # Run all test suites
    passed, tests = test_imports()
    total_passed += passed
    total_tests += tests
    
    passed, tests = test_model_architecture()
    total_passed += passed
    total_tests += tests
    
    passed, tests = test_loss_functions()
    total_passed += passed
    total_tests += tests
    
    passed, tests = test_configuration()
    total_passed += passed
    total_tests += tests
    
    passed, tests = test_utilities()
    total_passed += passed
    total_tests += tests
    
    passed, tests = test_gpu_availability()
    total_passed += passed
    total_tests += tests
    
    # Print summary
    print_header("Test Summary")
    
    percentage = (total_passed / total_tests * 100) if total_tests > 0 else 0
    
    if total_passed == total_tests:
        color = GREEN
        status = "ALL TESTS PASSED ✓"
    elif total_passed >= total_tests * 0.8:
        color = YELLOW
        status = "MOST TESTS PASSED ⚠"
    else:
        color = RED
        status = "SOME TESTS FAILED ✗"
    
    print(f"{color}  {status}{RESET}")
    print(f"  Passed: {total_passed}/{total_tests} ({percentage:.1f}%)")
    
    if total_passed == total_tests:
        print(f"\n{GREEN}{'='*60}{RESET}")
        print(f"{GREEN}{'Implementation validated successfully!':^60}{RESET}")
        print(f"{GREEN}{'Ready to start training.':^60}{RESET}")
        print(f"{GREEN}{'='*60}{RESET}\n")
    else:
        print(f"\n{YELLOW}{'='*60}{RESET}")
        print(f"{YELLOW}{'Some tests failed. Review errors above.':^60}{RESET}")
        print(f"{YELLOW}{'='*60}{RESET}\n")
    
    return total_passed == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
