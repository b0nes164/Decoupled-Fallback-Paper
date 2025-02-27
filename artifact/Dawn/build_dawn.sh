#!/bin/bash
set -e
set -o pipefail

NUM_THREADS=${1:-4}
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
ROOT_DIR="$(realpath "$SCRIPT_DIR/../../..")"

echo "Script directory: $SCRIPT_DIR"
echo "Project root: $ROOT_DIR"

command -v git >/dev/null 2>&1 || { echo "Error: git is required but not installed."; exit 1; }
command -v cmake >/dev/null 2>&1 || { echo "Error: cmake is required but not installed."; exit 1; }
command -v clang++ >/dev/null 2>&1 || { echo "Error: clang++ is required but not installed."; exit 1; }

echo "=== Installing dependencies ==="
sudo apt update
sudo apt install -y libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libx11-xcb-dev mesa-common-dev

echo "=== Cloning the Dawn repository ==="
if [ ! -d "$ROOT_DIR/dawn" ]; then
    git clone https://dawn.googlesource.com/dawn "$ROOT_DIR/dawn"
else
    echo "Dawn repository already exists; skipping clone."
fi

cd "$ROOT_DIR/dawn"
git fetch --all
git checkout chromium/6950

echo "=== Building and installing Dawn ==="
cmake -S . -B out/Release \
  -DDAWN_FETCH_DEPENDENCIES=ON \
  -DDAWN_ENABLE_INSTALL=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DDAWN_BUILD_TESTS=OFF \
  -DDAWN_BUILD_SAMPLES=OFF \
  -DTINT_BUILD_TESTS=OFF \
  -DTINT_BUILD_CMD_TOOLS=OFF \
  -DCMAKE_CXX_COMPILER=clang++
cmake --build out/Release --parallel "${NUM_THREADS}"
cmake --install out/Release --prefix install/Release

echo "=== Building the artifact ==="
cd "$SCRIPT_DIR"

export CMAKE_PREFIX_PATH="$(realpath "$ROOT_DIR/dawn/install/Release")"

cmake -S . -B out/Release -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++
cmake --build out/Release --parallel "${NUM_THREADS}"

echo "=== Build complete! ==="

