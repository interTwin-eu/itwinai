Profiling Lattice QCD with the ``py-spy`` Profiler—an Example
=============================================================

The following is an example of how we used the ``py-spy`` profiler to find a bottleneck in the
Lattice QCD use case. 

Introduction
------------

Lattice QCD is a well-established non-perturbative approach to solving the quantum
chromodynamics (QCD) theory of quarks and gluons. This is a very technical field, but the
details of the use case are not very important for the sake of this tutorial. You can read
more about the use case on the
`GitHub page for normflow <https://github.com/jkomijani/normflow_/>`_. 

Using our integration of the ``py-spy`` profiler, we were able to find an important bottleneck
that shows up when running this use case in a distributed manner. This document will show how
the profiling outputs were created and how they were used to spot and fix a large bottleneck.

.. note::

   The code used in this example is from a private repository called normflow, which differs
   slightly from the publicly accessible version linked above. However, these differences will
   not affect your understanding of the profiling and optimization workflow presented here.

Searching for Bottlenecks
-------------------------

The first thing we did was to run the code using a single A100 GPU for around 10 epochs and
profile it with the ``py-spy`` profiler. We set the ``--library-name`` flag to ``normflow``,
since this is the name of the library that we want to check for bottlenecks. The exact
command looks like this:

.. code-block:: bash

   itwinai generate-py-spy-report --file profiling-output/lattice_qcd_profile.txt --library-name normflow --aggregate-leaf-paths

This resulted in the following table:

.. code-block::

     name                      | path                                        |   line | library_function_name       | library_function_path                        | library_function_line   | proportion (n)
    ---------------------------+---------------------------------------------+--------+-----------------------------+----------------------------------------------+-------------------------+------------------
     __torch_function__        | torch/utils/_device.py                      |     79 | haar_qr                     | normflow/lib/linalg/qr_decomposition.py      | 54                      | 80.53% (14447)
     __torch_function__        | torch/utils/_device.py                      |     79 | modal2antihermitian2unitary | normflow/lib/matrix_handles/flow_handle.py   | 106                     | 7.27% (1304)
     __torch_function__        | torch/utils/_device.py                      |     79 | special_svd                 | normflow/lib/linalg/__init__.py              | 47                      | 1.07% (192)
     __torch_function__        | torch/utils/_device.py                      |     79 | matrix2phase_               | normflow/lib/matrix_handles/matrix_handle.py | 45                      | 0.62% (112)
     __torch_function__        | torch/utils/_device.py                      |     79 | modal2antihermitian2unitary | normflow/lib/matrix_handles/flow_handle.py   | 104                     | 0.60% (108)
     __torch_function__        | torch/utils/_device.py                      |     79 | modal2antihermitian2unitary | normflow/lib/matrix_handles/flow_handle.py   | 105                     | 0.50% (90)
     is_available              | torch/cuda/__init__.py                      |    128 | <module>                    | normflow/device/__init__.py                  | 7                       | 0.41% (73)
     backward                  | lattice_ml/linalg/_autograd/eig_autograd.py |    107 | Not Found                   | Not Found                                    | Not Found               | 0.33% (60)
     _call_with_frames_removed | built-in                                    |     -1 | Not Found                   | Not Found                                    | Not Found               | 0.27% (48)
     calc_eig_delta            | lattice_ml/linalg/_autograd/eig_autograd.py |    206 | Not Found                   | Not Found                                    | Not Found               | 0.26% (47)

Reading this table, we can see that the function ``haar_qr()`` in the file 
``normflow/lib/linalg/qr_decomposition.py`` on line 54 is the most time consuming. Typically
in a machine learning context you expect the most time-consuming functions to be related
to the backpropagation in the gradient descent algorithm. However, in this case, we see that
this is not the case. This tells us that there might be a bottleneck in the ``haar_qr()``
function. At this point we decided to investigate the function further. The function looks
like this:

.. code-block:: python

    def haar_qr(x, q_only=False):
        """Return a phase corrected version of qr decomposition that can be used to
        generate unitary matrices with the so-called haar measure.

        Performing the phase correction on q & r matrices, the diagonal terms of
        the r matrix are going to be real and positive.
        For further discussion see `[Mezzadri]`_.

        .. _[Mezzadri]:
            F. Mezzadri,
            "How to generate random matrices from the classical compact groups",
            :arXiv:`math-ph/0609050`.

        Parameters
        ----------
        x : tensor,
            the set of matrices for qr decomposition

        q_only : boolean (otpional)
            if True, only q will be returned rather than q & r (default is False).
        """
        q, r = torch.linalg.qr(x, mode='complete')  # <-- line 54 from the table
        # we now "correct" the phase of columns & rows in q & r, respectively
        d = torch.diagonal(r, dim1=-2, dim2=-1)
        phase = d / torch.abs(d)
        q = q * phase.unsqueeze(-2)  # correct the phase of columns
        if q_only:
            return q
        r = r * (1 / phase).unsqueeze(-1)  # correct the phase of rows
        # Note that x = q @ r before & after the pahse correction
        return q, r

Notice the comment pointing out the most time-consuming line from the table, i.e. this one:

.. code-block:: python

    q, r = torch.linalg.qr(x, mode='complete')

Optimizing the Code
-------------------
       
We see that the most time-consuming line is
a ``PyTorch`` function, so we can expect the implementation to be well-optimized. Now, to 
figure out why this is so slow, we have to read up a bit on the QR decomposition. We will spare
you the details, but one of the key insights is that this decomposition has sequential
dependencies, in the sense that it consists of a bunch of steps where each step depends on the
previous one. 


If you are familiar with GPUs vs CPUs, you will know that a key distinction
is that GPUs are very good at parallelizing while a CPU is very fast at sequential tasks.
Let's see if moving the computation to the CPU is improves performance. To test this in a way
that's relevant to our use case, we will create a tensor of the same size as the one that gets
passed to the ``haar_qr()`` function. After a quick debugging session, we find that the size is
``(64, 4, 4, 4, 4, 4, 3, 3)``, where the first number, 64, comes from the batch size. Now that
we know the size, we can do the following quick test in a Python shell:

.. code-block:: python

    import torch
    from time import time

    a = torch.rand([64, 4, 4, 4, 4, 4, 3, 3])
    start_time = time();res = torch.linalg.qr(a, mode='complete');end_time = time()
    end_time - start_time
    > 0.032

    a = a.to('cuda')
    start_time = time();res = torch.linalg.qr(a, mode='complete');end_time = time()
    end_time - start_time
    > 3.6

We can see that running it on the GPU takes ~100 times as long. This confirms that performing
this computation on the GPU indeed is a bottleneck.

After moving this computation to the CPU—while leaving the rest of the code on the GPU, of
course—we reduce the total training time of this example from ~77 seconds to ~23 seconds.
The optimized version now runs at **more than 300% of the original speed**, showing that our
optimization indeed proved fruitful. A new run of profiling also shows that the
time spent doing the QR-decomposition was reduced to ~8%, further substantiating that the
bottleneck has been resolved. 

Conclusion
----------

Using the data aggregation from ``itwinai`` made it possible to find a massive bottleneck
and optimize it away in a short amount of time. After our optimization, our program runs
in less than a third of the initial time when using GPUs. 
