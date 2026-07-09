# NicheCpp

NicheCpp is an experimental libtorch implementation of the `nicheformer` framework.  See [theislab github](https://github.com/theislab/nicheformer)
for the python code base.  See also the [paper](https://www.nature.com/articles/s41592-025-02814-z)
by Tejada-Lapuerta et al.
  
## Requirements  
  
- **CMake** 3.18+  
- **LibTorch** (CPU version, 2.7.0) – [download here](https://pytorch.org/get-started/locally/) (select `LibTorch`, `C++/Java`, `CPU`)  

### 1. Install Libtorch

Download the **CPU** shared version for Windows from the [official PyTorch website](https://pytorch.org/get-started/locally/).    

Example link for version 2.7.0 (CPU)(Windows):    
https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.7.0%2Bcpu.zip    

Example link for version 2.7.0 (CPU)(MacOS):    
https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.13.0.zip  

Extract the archive so that you have a folder named `libtorch` inside the project root.  

### 2. Configure torchscript  

Build a environment with the following requirements.

```cmd
conda activate -f environment.yml
```

Run `make_torchscript.ipynb` to generate `nicheformer_embeddings.pt`

### 3. Build with CMake  

#### For Windows 

Open a **x64 Native Tools Command Prompt for VS 2022**, navigate to the project root, and run:  

```cmd  
mkdir build  
cd build  
cmake -G "Visual Studio 17 2022" -A x64 -T v143 -DCMAKE_PREFIX_PATH="../libtorch" ..  
cmake --build . --config Release 
```

#### For MacOS 

```cmd  
cmake -S . -B build   -DCMAKE_PREFIX_PATH="$PWD/libtorch"
cmake --build build
```

### 4. Run

Input any image as an argument as you like.

```
./build/run_nicheformer
```
