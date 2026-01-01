"""
Quick fix script for syntax errors in train_spatial_clustering.py
Run this on RunPod to fix the file before testing
"""

import re

# Read the file
with open('train_spatial_clustering.py', 'r') as f:
    lines = f.readlines()

# Find and fix issues line by line
fixed_lines = []
skip_next = 0

for i, line in enumerate(lines):
    if skip_next > 0:
        skip_next -= 1
        continue
    
    # Fix 1: Unterminated string on line ~415
    if "checkpoint['globametrics" in line:
        # This should be: self.global_step = checkpoint['global_step']
        fixed_lines.append("        self.global_step = checkpoint['global_step']\n")
        # Skip any junk lines that follow
        continue
    
    # Fix 2: Malformed return with mixed brackets
    if "return epoch_losses['total_loss', 'position_loss'" in line and '}' in line:
        fixed_lines.append("        return epoch_losses\n")
        continue
    
    # Fix 3: Duplicate self.val_metrics append with closing paren
    if "self.val_metrics[k].append(val_losses[k]))" in line:
        fixed_lines.append(line.replace("))", ")"))
        continue
    
    # Fix 4: Remove duplicate "# Average validation losses" blocks
    if i > 0 and "# Average validation losses" in line:
        # Check if we already have this section
        prev_lines_text = ''.join(fixed_lines[-20:])
        if prev_lines_text.count("# Average validation losses") >= 1:
            continue
    
    # Fix 5: Skip lines that look like code fragments from bad merge
    if line.strip() and not line.strip().startswith('#'):
        # Check for junk patterns
        if 'for k, v in train_losses.items():' in line and i > 0 and 'def train(' not in ''.join(lines[max(0,i-10):i]):
            # This is likely a duplicate fragment, skip it and following related lines
            j = i
            while j < len(lines) and j < i + 10:
                if 'else:' in lines[j] or 'self.load_checkpoint' in lines[j]:
                    break
                j += 1
            skip_next = j - i - 1
            continue
    
    fixed_lines.append(line)

# Write back
with open('train_spatial_clustering.py', 'w') as f:
    f.writelines(fixed_lines)

print("✓ Fixed syntax errors in train_spatial_clustering.py")
print("  - Fixed unterminated string literal")
print("  - Fixed malformed return statement")
print("  - Removed duplicate code blocks")
print("\nNow run: python test_spatial_clustering.py")
