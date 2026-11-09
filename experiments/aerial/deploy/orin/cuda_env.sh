# Source on Orin before Tello/Lumos Python. Torch's bundled cuDNN 9.3 must
# precede the extracted CUDA 12.6 tree (which has cuDNN 9.0).
export LD_LIBRARY_PATH="$HOME/.local/lib/python3.10/site-packages/torch/lib:$HOME/cuda-runtime/usr/local/cuda-12.6/lib64:$HOME/cuda-runtime/usr/local/cuda-12.6/targets/aarch64-linux/lib:$HOME/cuda-runtime/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}"
export PATH="$HOME/.local/bin:$PATH"
export AERIAL_DA3_FP16=1
