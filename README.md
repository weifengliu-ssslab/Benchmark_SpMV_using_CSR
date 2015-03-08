# SpMV on Heterogeneous Processors using the CSR Format

This is the source code of paper "Speculative Segmented Sum for Sparse Matrix-Vector Multiplication on Heterogeneous Processors" submitted to Elsevier journal "Parallel Computing".

Contact: Weifeng Liu (weifeng.liu _at_ nbi.ku.dk) and/or Brian Vinter (vinter _at_ nbi.ku.dk).

Our algorithm has been implemented on three heterogeneous processors from Intel, AMD and nVidia. See below for a guide on how to benchmark our code.

<br><hr>
<h3>Intel platform</h3>

- Prerequisites

1. processor: Intel Broadwell (the 5th generation processor) or above, 

2. OS: Microsoft Windows 7 or above, 

3. Other tools: Intel OpenCL SDK with OpenCL 2.0 support, Microsoft Visual Studio 2012 or above.

- Benchmarking

1. Open visual studio solution ``spmv_opencl_intel.sln`` in folder ``spmv_opencl_intel``. 

2. Make sure `Debug` and `x64` in the Visual Studio IDE are selected.

3. Build the project.

4. Got to folder ``x64/Debug`` and run the generated executable file with an argument (filename of the benchmark matrix in the Matrix Market format). E.g. ``spmv.exe D:\matrices\filename.mtx``

<br><hr>
<h3>AMD platform</h3>

- Prerequisites

1. Processor: AMD Kaveri or above, 

2. OS: Ubuntu or other Linux versions, 

3. Other tools: AMD GPU driver with OpenCL 2.0 support, AMD APP SDK 3.0 Beta or above.

- Benchmarking

1. Make sure ``Makefile`` in folder ``spmv_opencl_amd`` has corrected paths. 

2. Run ``make USE_DOUBLE=0`` or ``make USE_DOUBLE=1`` for building single precision or double precision SpMV.

3. Run the generated executable file with an argument (filename of the benchmark matrix in the Matrix Market format). E.g. ``./spmv /home/user/Downloads/matrices/filename.mtx``
 
<br><hr>
<h3>nVidia platform</h3>

- Prerequisites

1. Processor: nVidia Tegra K1 or above, 

2. OS: Ubuntu Linux 14.04 or above, 

3. Other tools: nVidia GPU driver r19.2 or above, CUDA SDK 6.0 or above.

- Benchmarking

1. Make sure ``Makefile`` in folder ``spmv_cuda`` has corrected paths. 

2. Make sure ``Makefile`` has proper shader model (e.g., ``-arch=sm_32``) for nvcc compiler.

3. Run ``make USE_DOUBLE=0`` or ``make USE_DOUBLE=1`` for building single precision or double precision SpMV.

4. Run the generated executable file with an argument (filename of the benchmark matrix in the Matrix Market format). E.g. ``./spmv /home/user/Downloads/matrices/filename.mtx``
