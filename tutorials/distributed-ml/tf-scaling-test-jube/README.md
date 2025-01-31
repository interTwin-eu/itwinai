# Benchmarking tutorial using JUBE

Benchmarking of itwinai can also be performed with the JUBE Benchmarking Environment from JSC.
The JUBE benchmarking tool is already setup in the environment files provided under `env-files`.

## Source the environment

Find the location of your environment file along with the module load commands, such as:

```bash
ml Stages/2024 GCC/12.3.0 OpenMPI CUDA/12 MPI-settings/CUDA Python HDF5 PnetCDF libaio mpi4py CMake  cuDNN/8.9.5.29-CUDA-12
source envAI_hdfml/bin/activate
```

## Run benchmark

The benchmarks are defined in the `general_jobsys.xml` file.
One can specify the configurations in terms of parameters such as the number of nodes.
The benchmark can be simply launched with the command:

```bash
jube run general_jobsys.xml
```

## Monitor status of benchmark run

The status of the run can be monitored with:

```bash
jube continue bench_run --id last
```

## Check results of the benchmark run

The results can be viewed with:

```bash
jube result -a bench_run --id last
```

This will create `result-csv.dat` file in the `results` folder.

The scaling and efficiency plots can be generated with the `bench_plot.ipynb` file
which takes the `result-csv.dat` file as input.
