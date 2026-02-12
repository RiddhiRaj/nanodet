# NanoDet RKNN Demo for Vicharak Axon (Quick Setup)

This is a minimal flow to go from NanoDet weights to RKNN and run inference on the Vicharak Axon.

## 1. Create two environments

### `venv-rknn` (done with Python 3.12.12)

Follow this guide for RKNN Toolkit setup:
https://github.com/vicharak-in/Axon-NPU-Guide/tree/main/examples/yolov8-11_model_conversion_n_deployment#create-environment-2-venv-rknn

### `venv-nanodet` (Python 3.8 only)

NanoDet install must be done with Python 3.8.

```bash
python3.8 -m venv venv-nanodet
source venv-nanodet/bin/activate

git clone https://github.com/RangiLyu/nanodet.git
pip install --upgrade pip
pip install -r /path/to/this/repo/requirements.txt

cd nanodet
python setup.py develop
python -c "import nanodet; print('NanoDet OK')"
```

Do not use NanoDet official requirements from:
https://raw.githubusercontent.com/RangiLyu/nanodet/refs/heads/main/requirements.txt

Use this repo's `requirements.txt` instead (it avoids known compatibility issues).

## 2. Download weights and export ONNX

1. Download NanoDet weights from:
   https://github.com/RangiLyu/nanodet?tab=readme-ov-file#model-zoo
2. Export ONNX using official steps:
   https://github.com/RangiLyu/nanodet?tab=readme-ov-file#export-model-to-onnx

Important:
- Use the matching config file from NanoDet for your chosen weight file.
- Keep model/input size consistent (example: `416x416` weight with `416x416` config/input).

## 3. Convert ONNX to RKNN

Use `tools/nanodet2rknn.py` from this repo.

### FP model (no quantization)

```bash
source venv-rknn/bin/activate
cd /path/to/this/repo

python nanodet2rknn.py \
  --onnx /path/to/model.onnx \
  --config /path/to/nanodet_config.yml \
  --output /path/to/model.rknn \
  --platform rk3588
```

### INT8 quantization (optional)

```bash
python nanodet2rknn.py \
  --onnx /path/to/model.onnx \
  --config /path/to/nanodet_config.yml \
  --output /path/to/model_int8.rknn \
  --platform rk3588 \
  --quantize \
  --dataset /path/to/dataset.txt
```

Notes:
- `--config` should be the correct NanoDet YAML from your NanoDet repo clone (for example `nanodet/config/...`) corresponding to your weights/ONNX.
- If `--output` is omitted, the script auto-generates an RKNN filename.

## 4. Run inference

Use `nanodet_rknn_inference.py`.

Always pass the correct `--input_size` (`W,H`) for the model and add `--save` so output image is written.

```bash
python3 nanodet_rknn_inference.py \
  --model /path/to/model.rknn \
  --image /path/to/test.jpg \
  --input_size 416,416 \
  --save
```

Optional benchmark (500 iterations on the same image):

```bash
python3 nanodet_rknn_inference.py \
  --model /path/to/model.rknn \
  --image /path/to/test.jpg \
  --input_size 416,416 \
  --save \
  --benchmark
```
