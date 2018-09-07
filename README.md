# DoomSinglePlayer

This repo contains the code used to train an agent for the [Vizdoom 2018 competition track 1 singleplayer](http://vizdoom.cs.put.edu.pl/competition-cig-2018/competition-results). An agent trained with this code completes a randomly generated, previously unseen level of Doom in the competition below. 

<img src="example.gif" width="400">

## Requirements:
For now, this code assumes you are using two machines one with 8 cores and 1 gpu and another with 4 cores and 1 gpu. This can be updated to other configurations by updating the MPI comm split commands and updating references to ranks 0 and 8 to ranks on which you have GPUs. I intend to make this more dynamic in the future.

All machines must have the following installed:
- Vizdoom
- PyOblige
- OpenMPI compiled with multithreading support
- Mpi4Py (installed after OpenMPI)
- Horovod (installed after OpenMPI)
- Tensorflow

To run the code, edit RunTraining.py with parameters appropriate for your machines and then execute the following command:

```bash
$ mpiexec -n 12 \
    -hostfile hosts \
    -bind-to none -map-by slot \
    -mca pml ob1 -mca btl ^openib \
    python RunTraining.py
```

where the file hosts contains the hostnames and number of processes to run per machine. 
