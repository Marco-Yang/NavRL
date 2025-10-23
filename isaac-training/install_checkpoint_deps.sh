#!/bin/bash
# Install LZ4 compression library for NavRL checkpoint optimization

echo "========================================"
echo "NavRL Checkpoint Optimization Setup"
echo "========================================"
echo ""

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "❌ Error: pip not found. Please install Python pip first."
    exit 1
fi

echo "📦 Installing LZ4 compression library..."
pip install lz4

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Installation successful!"
    echo ""
    echo "You can now use checkpoint compression:"
    echo "  python training/scripts/train.py compress_checkpoint=True"
    echo ""
else
    echo ""
    echo "❌ Installation failed. Please check your Python environment."
    exit 1
fi

# Test installation
echo "🔍 Testing LZ4 installation..."
python -c "import lz4.frame; print('LZ4 version:', lz4.VERSION)" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ LZ4 is working correctly!"
else
    echo "⚠️  Warning: LZ4 import test failed. Please check your installation."
fi

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
