Distributed Data Parallelism
=================================
Deep neural networks (DNN) are often extremely large and are trained on massive amounts of data, more than most computers have memory for.
Even smaller DNNs can take days to train. 
Distributed Data Parallelisation (DDP) adresses these two issues, long training times and limited memory, by using multiple machines to host and train both model and data.

Data parallelisation is an easy way for a developer to vastly reduce training times.
Rather than using single-node parallelism, Distributed Data Parallelism (DDP) scales to multiple machnies. 
This scaling maximises parallelisability of your model and drastically reduces training times.

Another benefit of DDP is removal of single-machine memory constraints. Since a dataset or model can be stored across several machines, it becomes possible to analyse much larger datasets or models.

Below is a list of resources expanding on theoretical aspects and practical implementations of DDP:

* Introduction to DP: https://siboehm.com/articles/22/data-parallel-training

* https://pytorch.org/tutorials/beginner/ddp_series_theory.html

* https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

* https://huggingface.co/blog/pytorch-ddp-accelerate-transformers


Investigation of expected performance improvement: 

* https://www.mdpi.com/2079-9292/11/10/1525




