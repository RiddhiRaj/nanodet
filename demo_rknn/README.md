# NanoDet RKNN Demo for RK3588

This folder contains tools and demos for running NanoDet on Rockchip RK3588 NPU using RKNN Toolkit.

## Overview

The workflow consists of two main steps:

1. **Convert ONNX → RKNN** (on PC with x86_64)
2. **Run Inference** (on RK3588 device)

## Requirements

### On PC (for conversion):
```bash
pip install rknn-toolkit2
pip install opencv-python numpy
```

Download RKNN Toolkit 2 from: https://github.com/rockchip-linux/rknn-toolkit2

### On RK3588 Device (for inference):
```bash
pip install rknn-toolkit2-lite
pip install opencv-python numpy
```

## Step 1: Convert ONNX to RKNN (on PC)

### Basic Conversion (Float16 - Recommended)

```bash
# First, export ONNX model if you haven't already
source venv-nanodet/bin/activate
python tools/export_onnx.py \
  --cfg_path config/nanodet-plus-m-1.5x_416.yml \
  --model_path nanodet-plus-m-1.5x_416.pth \
  --out_path nanodet-plus-m-1.5x_416.onnx \
  --input_shape 416,416

# Convert ONNX to RKNN (Float16 - best balance)
python tools/export_rknn.py \
  --config config/nanodet-plus-m-1.5x_416.yml \
  --onnx_model nanodet-plus-m-1.5x_416.onnx \
  --out_path nanodet-plus-m-1.5x_416.rknn \
  --quantize_type float16 \
  --target_platform rk3588
```

### INT8 Quantization (Fastest, requires calibration dataset)

For maximum speed with minimal accuracy loss:

```bash
# Prepare calibration dataset
# Create a folder with ~100-500 representative images
mkdir calibration_images
# Copy images to calibration_images/

# Convert with INT8 quantization
python tools/export_rknn.py \
  --config config/nanodet-plus-m-1.5x_416.yml \
  --onnx_model nanodet-plus-m-1.5x_416.onnx \
  --out_path nanodet-plus-m-1.5x_416_int8.rknn \
  --quantize_type int8 \
  --target_platform rk3588 \
  --quantize_dataset calibration_images \
  --dataset_size 200
```

### Available Quantization Types

| Type | Speed | Accuracy | Size | Recommendation |
|------|-------|----------|------|----------------|
| float32 | Slowest | Best | Largest | Not recommended for NPU |
| float16 | Fast | Excellent | Medium | **Recommended** |
| int8 | Fastest | Good | Smallest | For maximum performance |
| uint8 | Fastest | Good | Smallest | Alternative to int8 |

## Step 2: Transfer Model to RK3588

```bash
# Copy RKNN model to your RK3588 device
scp nanodet-plus-m-1.5x_416.rknn root@<RK3588_IP>:/path/to/models/
scp demo_rknn/nanodet_rknn.py root@<RK3588_IP>:/path/to/inference/
```

## Step 3: Run Inference on RK3588

### Image Inference

```bash
# On RK3588 device
python3 nanodet_rknn.py \
  --model nanodet-plus-m-1.5x_416.rknn \
  --image test.jpg \
  --input_size 416,416 \
  --score_threshold 0.35 \
  --save
```

### Benchmark Performance

```bash
# Test inference speed (100 iterations)
python3 nanodet_rknn.py \
  --model nanodet-plus-m-1.5x_416.rknn \
  --input_size 416,416 \
  --benchmark
```

### Custom Parameters

```bash
python3 nanodet_rknn.py \
  --model your_model.rknn \
  --image test.jpg \
  --input_size 320,320 \
  --num_classes 10 \
  --reg_max 7 \
  --strides 8,16,32,64 \
  --score_threshold 0.4 \
  --nms_threshold 0.5 \
  --save
```

## Expected Performance on RK3588

| Model | Resolution | Quantization | Latency | FPS | mAP |
|-------|------------|--------------|---------|-----|-----|
| NanoDet-Plus-m | 320×320 | Float16 | ~8ms | ~125 | 27.0 |
| NanoDet-Plus-m | 416×416 | Float16 | ~12ms | ~83 | 30.4 |
| NanoDet-Plus-m-1.5x | 320×320 | Float16 | ~10ms | ~100 | 29.9 |
| NanoDet-Plus-m-1.5x | 416×416 | Float16 | ~15ms | ~67 | 34.1 |
| NanoDet-Plus-m | 416×416 | INT8 | ~6ms | ~167 | 29.5 |
| NanoDet-Plus-m-1.5x | 416×416 | INT8 | ~9ms | ~111 | 33.2 |

*Note: Actual performance may vary based on RK3588 firmware version and NPU driver.*

## Configuration Files

Different NanoDet models require different config files:

| Model File | Config File | Input Size |
|------------|-------------|------------|
| nanodet-plus-m_320.pth | config/nanodet-plus-m_320.yml | 320×320 |
| nanodet-plus-m_416.pth | config/nanodet-plus-m_416.yml | 416×416 |
| nanodet-plus-m-1.5x_320.pth | config/nanodet-plus-m-1.5x_320.yml | 320×320 |
| nanodet-plus-m-1.5x_416.pth | config/nanodet-plus-m-1.5x_416.yml | 416×416 |

## Troubleshooting

### Issue: "RKNN model load failed"
- Ensure you're using the correct RKNN Toolkit version (2.0.0 or later)
- Check that the model was converted for the correct platform (rk3588)

### Issue: "Init runtime failed"
- Make sure rknn-toolkit2-lite is installed on RK3588
- Check NPU driver: `cat /sys/kernel/debug/rknpu/version`

### Issue: Low FPS
- Try INT8 quantization for better performance
- Ensure NPU cores are being used (check with `top` or `htop`)
- Use all NPU cores: modify `core_mask=RKNNLite.NPU_CORE_0_1_2` in code

### Issue: Poor accuracy after INT8 quantization
- Use more calibration images (500-1000 recommended)
- Ensure calibration images are representative of your use case
- Try different quantization algorithms in export_rknn.py

## Advanced: NPU Core Selection

RK3588 has 3 NPU cores. You can select which cores to use:

```python
# In nanodet_rknn.py, modify init_runtime:
core_mask = RKNNLite.NPU_CORE_0       # Use core 0 only
core_mask = RKNNLite.NPU_CORE_0_1     # Use cores 0 and 1
core_mask = RKNNLite.NPU_CORE_0_1_2   # Use all 3 cores (recommended)
```

## Model Optimization Tips

1. **Input Resolution**: Lower resolution = faster inference
   - 320×320: Best for real-time applications
   - 416×416: Good balance of speed and accuracy

2. **Quantization**: 
   - Float16: Best for most use cases (no calibration needed)
   - INT8: When you need maximum speed and have calibration data

3. **Batch Processing**: RKNN supports batch inference for even better throughput

4. **Pre/Post Processing**: Consider moving preprocessing to CPU to pipeline with NPU

## Complete Example Workflow

```bash
# On PC: Export and convert
source venv-nanodet/bin/activate

# Step 1: Export ONNX
python tools/export_onnx.py \
  --cfg_path config/nanodet-plus-m-1.5x_416.yml \
  --model_path nanodet-plus-m-1.5x_416.pth \
  --out_path nanodet-plus-m-1.5x_416.onnx

# Step 2: Convert to RKNN Float16
python tools/export_rknn.py \
  --config config/nanodet-plus-m-1.5x_416.yml \
  --onnx_model nanodet-plus-m-1.5x_416.onnx \
  --quantize_type float16 \
  --target_platform rk3588

# Step 3: Transfer to RK3588
scp nanodet-plus-m-1.5x_416.rknn root@192.168.1.100:~/models/
scp demo_rknn/nanodet_rknn.py root@192.168.1.100:~/
scp demo/teddy.png root@192.168.1.100:~/

# On RK3588: Run inference
ssh root@192.168.1.100
cd ~/
python3 nanodet_rknn.py \
  --model models/nanodet-plus-m-1.5x_416.rknn \
  --image teddy.png \
  --input_size 416,416 \
  --save

# Check performance
python3 nanodet_rknn.py \
  --model models/nanodet-plus-m-1.5x_416.rknn \
  --input_size 416,416 \
  --benchmark
```

## Resources

- RKNN Toolkit 2: https://github.com/rockchip-linux/rknn-toolkit2
- RK3588 Documentation: https://www.rock-chips.com/
- NanoDet GitHub: https://github.com/RangiLyu/nanodet

## License

Same as NanoDet project - Apache 2.0 License
