"""
Quick fix script for syntax errors in train_spatial_clustering.py
Run this on RunPod to fix the file before testing
"""

import re

# Read the file
with open('train_spatial_clustering.py', 'r') as f:
    content = f.read()

# Fix 1: Remove any malformed return statements with mixed brackets
content = re.sub(
    r"return epoch_losses\['total_loss', 'position_loss', 'intensity_loss', 'contrast_loss', 'cluster_consistency'\]\}",
    "return epoch_losses",
    content
)

# Fix 2: Remove duplicate validation metric appends
content = re.sub(
    r"(\s+self\.val_metrics\[k\]\.append\(val_losses\[k\]\)\))\s+# Average validation losses.*?\1",
    r"\1\n\n        # Average validation losses",
    content,
    flags=re.DOTALL
)

# Write back
with open('train_spatial_clustering.py', 'w') as f:
    f.write(content)

print("✓ Fixed syntax errors in train_spatial_clustering.py")
print("Now run: python test_spatial_clustering.py")
