# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

---

# 3.1 and 3.2 Diagnostics
<details>
<summary>Click to expand</summary>

```
(.venv) C:\Users\Owner\MLEworkspace\mod3-kpan02>python project/parallel_check.py
MAP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map,
C:\Users\Owner\MLEworkspace\mod3-kpan02\minitorch\fast_ops.py (164)
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, C:\Users\Owner\MLEworkspace\mod3-kpan02\minitorch\fast_ops.py (164)
-----------------------------------------------------------------------------|loop #ID
    def _map(                                                                |
        out: Storage,                                                        |
        out_shape: Shape,                                                    |
        out_strides: Strides,                                                |
        in_storage: Storage,                                                 |
        in_shape: Shape,                                                     |
        in_strides: Strides,                                                 |
    ) -> None:                                                               |
        strides_match = np.array_equal(in_strides, out_strides)              |
        size = int(np.prod(out_shape))---------------------------------------| #2
                                                                             |
        for i in prange(size):-----------------------------------------------| #3
            if strides_match:                                                |
                out[i] = fn(in_storage[i])                                   |
                                                                             |
            else:                                                            |
                out_index = np.zeros(MAX_DIMS, np.int32)---------------------| #0
                in_index = np.zeros(MAX_DIMS, np.int32)----------------------| #1
                to_index(i, out_shape, out_index)                            |
                broadcast_index(out_index, out_shape, in_shape, in_index)    |
                o = index_to_position(out_index, out_strides)                |
                j = index_to_position(in_index, in_strides)                  |
                out[o] = fn(in_storage[j])                                   |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...

Fused loop summary:
+--0 has the following loops fused into it:
   +--1 (fused)
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #2, #3).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--3 is a parallel loop
   +--0 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (parallel)
   +--1 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (serial, fused with loop(s): 1)



Parallel region 0 (loop #3) had 1 loop(s) fused and 1 loop(s) serialized as part
 of the larger parallel loop (#3).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
C:\Users\Owner\MLEworkspace\mod3-kpan02\minitorch\fast_ops.py (180) is hoisted
out of the parallel loop labelled #3 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: out_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
C:\Users\Owner\MLEworkspace\mod3-kpan02\minitorch\fast_ops.py (181) is hoisted
out of the parallel loop labelled #3 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: in_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip,
C:\Users\Owner\MLEworkspace\mod3-kpan02\minitorch\fast_ops.py (214)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, C:\Users\Owner\MLEworkspace\mod3-kpan02\minitorch\fast_ops.py (214)
--------------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                         |
        out: Storage,                                                                 |
        out_shape: Shape,                                                             |
        out_strides: Strides,                                                         |
        a_storage: Storage,                                                           |
        a_shape: Shape,                                                               |
        a_strides: Strides,                                                           |
        b_storage: Storage,                                                           |
        b_shape: Shape,                                                               |
        b_strides: Strides,                                                           |
    ) -> None:                                                                        |
        strides_match = np.array_equal(a_strides, out_strides) and np.array_equal(    |
            b_strides, out_strides                                                    |
        )                                                                             |
        shape_match = np.array_equal(a_shape, b_shape)                                |
        size = int(np.prod(out_shape))------------------------------------------------| #7
                                                                                      |
        for i in prange(size):--------------------------------------------------------| #8
            if strides_match and shape_match:                                         |
                out[i] = fn(a_storage[i], b_storage[i])                               |
            else:                                                                     |
                out_index = np.zeros(MAX_DIMS, np.int32)------------------------------| #4
                a_pos = np.zeros(MAX_DIMS, np.int32)----------------------------------| #5
                b_pos = np.zeros(MAX_DIMS, np.int32)----------------------------------| #6
                to_index(i, out_shape, out_index)                                     |
                o = index_to_position(out_index, out_strides)                         |
                broadcast_index(out_index, out_shape, a_shape, a_pos)                 |
                j = index_to_position(a_pos, a_strides)                               |
                broadcast_index(out_index, out_shape, b_shape, b_pos)                 |
                k = index_to_position(b_pos, b_strides)                               |
                out[o] = fn(a_storage[j], b_storage[k])                               |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...

Fused loop summary:
+--4 has the following loops fused into it:
   +--5 (fused)
   +--6 (fused)
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #7, #8).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--8 is a parallel loop
   +--4 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--4 (parallel)
   +--5 (parallel)
   +--6 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--4 (serial, fused with loop(s): 5, 6)



Parallel region 0 (loop #8) had 2 loop(s) fused and 1 loop(s) serialized as part
 of the larger parallel loop (#8).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
C:\Users\Owner\MLEworkspace\mod3-kpan02\minitorch\fast_ops.py (235) is hoisted
out of the parallel loop labelled #8 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: out_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
C:\Users\Owner\MLEworkspace\mod3-kpan02\minitorch\fast_ops.py (236) is hoisted
out of the parallel loop labelled #8 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: a_pos = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
C:\Users\Owner\MLEworkspace\mod3-kpan02\minitorch\fast_ops.py (237) is hoisted
out of the parallel loop labelled #8 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: b_pos = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce,
C:\Users\Owner\MLEworkspace\mod3-kpan02\minitorch\fast_ops.py (270)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, C:\Users\Owner\MLEworkspace\mod3-kpan02\minitorch\fast_ops.py (270)
---------------------------------------------------------------|loop #ID
    def _reduce(                                               |
        out: Storage,                                          |
        out_shape: Shape,                                      |
        out_strides: Strides,                                  |
        a_storage: Storage,                                    |
        a_shape: Shape,                                        |
        a_strides: Strides,                                    |
        reduce_dim: int,                                       |
    ) -> None:                                                 |
        size = int(np.prod(out_shape))-------------------------| #10
        reduce_size = a_shape[reduce_dim]                      |
                                                               |
        for i in prange(size):---------------------------------| #12
            out_index: Index = np.zeros(MAX_DIMS, np.int32)----| #9
            to_index(i, out_shape, out_index)                  |
            o = index_to_position(out_index, out_strides)      |
            for j in prange(reduce_size):----------------------| #11
                out_index[reduce_dim] = j                      |
                p = index_to_position(out_index, a_strides)    |
                out[o] = fn(out[o], a_storage[p])              |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #10, #12, #9).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--12 is a parallel loop
   +--9 --> rewritten as a serial loop
   +--11 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--12 (parallel)
   +--9 (parallel)
   +--11 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--12 (parallel)
   +--9 (serial)
   +--11 (serial)



Parallel region 0 (loop #12) had 0 loop(s) fused and 2 loop(s) serialized as
part of the larger parallel loop (#12).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
C:\Users\Owner\MLEworkspace\mod3-kpan02\minitorch\fast_ops.py (283) is hoisted
out of the parallel loop labelled #12 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: out_index: Index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY

================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply,
C:\Users\Owner\MLEworkspace\mod3-kpan02\minitorch\fast_ops.py (294)
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, C:\Users\Owner\MLEworkspace\mod3-kpan02\minitorch\fast_ops.py (294)
------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                              |
    out: Storage,                                                                         |
    out_shape: Shape,                                                                     |
    out_strides: Strides,                                                                 |
    a_storage: Storage,                                                                   |
    a_shape: Shape,                                                                       |
    a_strides: Strides,                                                                   |
    b_storage: Storage,                                                                   |
    b_shape: Shape,                                                                       |
    b_strides: Strides,                                                                   |
) -> None:                                                                                |
    """NUMBA tensor matrix multiply function.                                             |
                                                                                          |
    Should work for any tensor shapes that broadcast as long as                           |
                                                                                          |
    ```                                                                                   |
    assert a_shape[-1] == b_shape[-2]                                                     |
    ```                                                                                   |
                                                                                          |
    Optimizations:                                                                        |
                                                                                          |
    * Outer loop in parallel                                                              |
    * No index buffers or function calls                                                  |
    * Inner loop should have no global writes, 1 multiply.                                |
                                                                                          |
                                                                                          |
    Args:                                                                                 |
    ----                                                                                  |
        out (Storage): storage for `out` tensor                                           |
        out_shape (Shape): shape for `out` tensor                                         |
        out_strides (Strides): strides for `out` tensor                                   |
        a_storage (Storage): storage for `a` tensor                                       |
        a_shape (Shape): shape for `a` tensor                                             |
        a_strides (Strides): strides for `a` tensor                                       |
        b_storage (Storage): storage for `b` tensor                                       |
        b_shape (Shape): shape for `b` tensor                                             |
        b_strides (Strides): strides for `b` tensor                                       |
                                                                                          |
    Returns:                                                                              |
    -------                                                                               |
        None : Fills in `out`                                                             |
                                                                                          |
    """                                                                                   |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                |
                                                                                          |
    # TODO: Implement for Task 3.2.                                                       |
    for n in prange(out_shape[0]):--------------------------------------------------------| #13
        for i in range(out_shape[1]):  # Rows of a                                        |
            for j in range(out_shape[2]):  # Columns of b                                 |
                sum = 0                                                                   |
                a_pos = n * a_batch_stride + i * a_strides[1]                             |
                b_pos = n * b_batch_stride + j * b_strides[2]                             |
                out_pos = n * out_strides[0] + i * out_strides[1] + j * out_strides[2]    |
                                                                                          |
                for k in range(a_shape[-1]):  # Columns of a and rows of b                |
                    sum += a_storage[a_pos] * b_storage[b_pos]                            |
                    a_pos += a_strides[2]                                                 |
                    b_pos += b_strides[1]                                                 |
                                                                                          |
                out[out_pos] = sum                                                        |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #13).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```
</details>

---

# 3.4 Comparison Graph
```
Timing summary
Size: 64
    fast: 0.00288
    gpu: 0.00607
Size: 128
    fast: 0.01536
    gpu: 0.0112
Size: 256
    fast: 0.08769
    gpu: 0.04953
Size: 512
    fast: 1.25781
    gpu: 0.26183
Size: 1024
    fast: 12.62239
    gpu: 0.90193
```

![Alt Text](images/graph.png)

---

# 3.5 Training
<details>
<summary>Simple (CPU, Small)</summary>
```
Epoch  0  loss  5.573065485081717 correct 45 Epoch Time 4.854224920272827
Epoch  10  loss  2.5318949866793448 correct 46 Epoch Time 0.05585908889770508
Epoch  20  loss  1.5167487233687145 correct 47 Epoch Time 0.05934596061706543
Epoch  30  loss  1.9668496060101601 correct 46 Epoch Time 0.05504012107849121
Epoch  40  loss  2.4647674681887297 correct 50 Epoch Time 0.0521697998046875
Epoch  50  loss  0.8169726288953983 correct 50 Epoch Time 0.04962778091430664
Epoch  60  loss  0.04611731251531294 correct 50 Epoch Time 0.05942988395690918
Epoch  70  loss  1.5852846217770222 correct 50 Epoch Time 0.05120205879211426
Epoch  80  loss  1.0238782595366702 correct 50 Epoch Time 0.05246305465698242
Epoch  90  loss  0.044722780552547324 correct 49 Epoch Time 0.06364893913269043
Epoch  100  loss  0.5383882083858323 correct 50 Epoch Time 0.05435514450073242
Epoch  110  loss  2.3465879884051346 correct 48 Epoch Time 0.05413198471069336
Epoch  120  loss  1.443456596283392 correct 50 Epoch Time 0.05104994773864746
Epoch  130  loss  0.7814191960193986 correct 50 Epoch Time 0.05220293998718262
Epoch  140  loss  0.8577317729433281 correct 50 Epoch Time 0.058226823806762695
Epoch  150  loss  0.4490481827021465 correct 49 Epoch Time 0.05678391456604004
Epoch  160  loss  0.38684792442435817 correct 50 Epoch Time 0.05252194404602051
Epoch  170  loss  1.4203508249989507 correct 48 Epoch Time 0.055290937423706055
Epoch  180  loss  0.22507901904223518 correct 50 Epoch Time 0.053724050521850586
Epoch  190  loss  0.7255494123626367 correct 50 Epoch Time 0.05923032760620117
Epoch  200  loss  1.0185094002657362 correct 50 Epoch Time 0.18589019775390625
Epoch  210  loss  0.12808207431333224 correct 50 Epoch Time 0.05611610412597656
Epoch  220  loss  0.3344119260856736 correct 50 Epoch Time 0.05352306365966797
Epoch  230  loss  2.253923437094458 correct 48 Epoch Time 0.05530095100402832
Epoch  240  loss  0.5081063894421629 correct 49 Epoch Time 0.054025888442993164
Epoch  250  loss  1.363575976023527 correct 50 Epoch Time 0.05588722229003906
Epoch  260  loss  0.8473330303288928 correct 50 Epoch Time 0.06096673011779785
Epoch  270  loss  0.31491470901451146 correct 49 Epoch Time 0.05280327796936035
Epoch  280  loss  0.035666721815018496 correct 50 Epoch Time 0.05517911911010742
Epoch  290  loss  1.0531471667879353 correct 50 Epoch Time 0.05776476860046387
Epoch  300  loss  0.8383028296943864 correct 50 Epoch Time 0.05624198913574219
Epoch  310  loss  0.056797301738307625 correct 50 Epoch Time 0.05689191818237305
Epoch  320  loss  0.3438614916032325 correct 50 Epoch Time 0.05584406852722168
Epoch  330  loss  0.6661939423275057 correct 50 Epoch Time 0.08422303199768066
Epoch  340  loss  0.00338109081924766 correct 50 Epoch Time 0.052618980407714844
Epoch  350  loss  0.21543201369638518 correct 50 Epoch Time 0.05861687660217285
Epoch  360  loss  0.0030306136106111685 correct 50 Epoch Time 0.060307979583740234
Epoch  370  loss  0.36282774831574777 correct 50 Epoch Time 0.05699896812438965
Epoch  380  loss  0.04181584771448184 correct 50 Epoch Time 0.059394121170043945
Epoch  390  loss  0.12197333011734425 correct 49 Epoch Time 0.0627889633178711
Epoch  400  loss  0.5088543093520141 correct 50 Epoch Time 0.061506032943725586
Epoch  410  loss  0.29070737529366 correct 49 Epoch Time 0.08820295333862305
Epoch  420  loss  0.2746599625891394 correct 50 Epoch Time 0.06296610832214355
Epoch  430  loss  0.507081867266153 correct 50 Epoch Time 0.061585187911987305
Epoch  440  loss  0.0008374388748473868 correct 50 Epoch Time 0.05957198143005371
Epoch  450  loss  0.057406284398479286 correct 50 Epoch Time 0.058897972106933594
Epoch  460  loss  0.6811589094303107 correct 50 Epoch Time 0.06078004837036133
Epoch  470  loss  0.2043376630679062 correct 49 Epoch Time 0.05555319786071777
Epoch  480  loss  1.116256550348641 correct 50 Epoch Time 0.05304288864135742
Epoch  490  loss  0.6886489718466784 correct 50 Epoch Time 0.052488088607788086
```
</details>


<details>
<summary>Simple (GPU, Small) </summary>
```
Epoch  0  loss  9.379384874120115 correct 38 Epoch Time 4.376723051071167
Epoch  10  loss  3.4001390496068673 correct 46 Epoch Time 1.6061482429504395
Epoch  20  loss  2.1328400007744817 correct 45 Epoch Time 1.6760098934173584
Epoch  30  loss  1.3999501096757454 correct 49 Epoch Time 1.5974607467651367
Epoch  40  loss  1.612827896066129 correct 49 Epoch Time 2.131270408630371
Epoch  50  loss  0.15731623808861003 correct 49 Epoch Time 1.5955500602722168
Epoch  60  loss  2.6096119529423265 correct 49 Epoch Time 1.872527837753296
Epoch  70  loss  0.18516333008887906 correct 50 Epoch Time 1.599280834197998
Epoch  80  loss  0.7059543669305273 correct 49 Epoch Time 1.6090402603149414
Epoch  90  loss  1.3106537893643089 correct 50 Epoch Time 1.6976828575134277
Epoch  100  loss  0.6153131376530239 correct 49 Epoch Time 1.61808443069458
Epoch  110  loss  0.8200131738226593 correct 49 Epoch Time 1.984994649887085
Epoch  120  loss  1.4476568080904468 correct 49 Epoch Time 1.6003332138061523
Epoch  130  loss  0.935765451120826 correct 50 Epoch Time 2.14292311668396
Epoch  140  loss  0.13920674284636592 correct 50 Epoch Time 1.6045007705688477
Epoch  150  loss  0.13280408920443595 correct 50 Epoch Time 1.608168601989746
Epoch  160  loss  0.3236947686100008 correct 50 Epoch Time 1.602752447128296
Epoch  170  loss  0.6704050477449128 correct 49 Epoch Time 1.5929014682769775
Epoch  180  loss  0.9026447217823985 correct 49 Epoch Time 1.7775564193725586
Epoch  190  loss  0.4017878755771605 correct 49 Epoch Time 1.6069695949554443
Epoch  200  loss  1.1708762584899717 correct 50 Epoch Time 2.184173822402954
Epoch  210  loss  1.3531649115876696 correct 50 Epoch Time 1.6110482215881348
Epoch  220  loss  0.5676757958231994 correct 50 Epoch Time 1.6929931640625
Epoch  230  loss  0.15128131708810463 correct 50 Epoch Time 1.633424997329712
Epoch  240  loss  0.1805282655668305 correct 50 Epoch Time 1.6017358303070068
Epoch  250  loss  0.6503390275462717 correct 50 Epoch Time 1.6199617385864258
Epoch  260  loss  0.1220011436238115 correct 50 Epoch Time 1.625246524810791
Epoch  270  loss  0.38807740235048116 correct 50 Epoch Time 2.222126007080078
Epoch  280  loss  0.3659204038824231 correct 50 Epoch Time 1.5986011028289795
Epoch  290  loss  0.696306077487429 correct 50 Epoch Time 1.595517873764038
Epoch  300  loss  0.2740084669212856 correct 49 Epoch Time 1.6130366325378418
Epoch  310  loss  0.26248263451274895 correct 50 Epoch Time 1.6726875305175781
Epoch  320  loss  0.9974264262142667 correct 49 Epoch Time 1.6237430572509766
Epoch  330  loss  0.781746379579948 correct 50 Epoch Time 1.7039175033569336
Epoch  340  loss  0.034939218804162044 correct 50 Epoch Time 2.2248375415802
Epoch  350  loss  0.32428817563606127 correct 50 Epoch Time 1.6018555164337158
Epoch  360  loss  0.0020064937434752517 correct 50 Epoch Time 1.591590404510498
Epoch  370  loss  0.01121450152420688 correct 50 Epoch Time 1.5920560359954834
Epoch  380  loss  0.004037594672985417 correct 50 Epoch Time 1.5890798568725586
Epoch  390  loss  0.17280894200593241 correct 50 Epoch Time 2.5109360218048096
Epoch  400  loss  0.0014042334047128253 correct 50 Epoch Time 1.6077816486358643
Epoch  410  loss  0.003947880061806977 correct 50 Epoch Time 1.977407455444336
Epoch  420  loss  0.000349981505975818 correct 50 Epoch Time 1.650658130645752
Epoch  430  loss  0.08371330380495502 correct 50 Epoch Time 1.605372667312622
Epoch  440  loss  0.012269654504486778 correct 50 Epoch Time 1.7228906154632568
Epoch  450  loss  0.8263741068816751 correct 50 Epoch Time 1.6102113723754883
Epoch  460  loss  0.16768951068763696 correct 50 Epoch Time 2.0997564792633057
Epoch  470  loss  0.1352846265808947 correct 50 Epoch Time 1.6071655750274658
Epoch  480  loss  0.09990697059867848 correct 50 Epoch Time 1.8440306186676025
Epoch  490  loss  0.059077680344070696 correct 50 Epoch Time 1.584028959274292
```
</details>


<details>
<summary>Split (CPU, Small)</summary>
```
Epoch  0  loss  6.722958261193735 correct 32 Epoch Time 5.268651008605957
Epoch  10  loss  5.8122000582215705 correct 46 Epoch Time 0.058557987213134766
Epoch  20  loss  7.000907975426973 correct 48 Epoch Time 0.056874990463256836
Epoch  30  loss  2.6742159410104933 correct 43 Epoch Time 0.05357217788696289
Epoch  40  loss  4.283984314376306 correct 49 Epoch Time 0.05856585502624512
Epoch  50  loss  1.841422606890445 correct 49 Epoch Time 0.055967092514038086
Epoch  60  loss  2.420595421457326 correct 50 Epoch Time 0.06127500534057617
Epoch  70  loss  1.5749952711476478 correct 50 Epoch Time 0.05464887619018555
Epoch  80  loss  0.6121027622930473 correct 50 Epoch Time 0.06494307518005371
Epoch  90  loss  0.8209394706265357 correct 50 Epoch Time 0.060530900955200195
Epoch  100  loss  0.8257930086390888 correct 50 Epoch Time 0.0635380744934082
Epoch  110  loss  0.8543699600386802 correct 50 Epoch Time 0.059886932373046875
Epoch  120  loss  0.5740434243856232 correct 50 Epoch Time 0.0681910514831543
Epoch  130  loss  0.5930688163708876 correct 50 Epoch Time 0.06990408897399902
Epoch  140  loss  0.3894881087996025 correct 50 Epoch Time 0.05563497543334961
Epoch  150  loss  0.609286116482566 correct 50 Epoch Time 0.06295394897460938
Epoch  160  loss  0.34619002060847726 correct 50 Epoch Time 0.05628800392150879
Epoch  170  loss  0.8308765650396308 correct 50 Epoch Time 0.05792593955993652
Epoch  180  loss  0.42748352456700633 correct 50 Epoch Time 0.0513310432434082
Epoch  190  loss  0.49520169372778794 correct 50 Epoch Time 0.05902528762817383
Epoch  200  loss  0.08384174788627634 correct 50 Epoch Time 0.05618715286254883
Epoch  210  loss  0.3410975646223826 correct 50 Epoch Time 0.05762791633605957
Epoch  220  loss  0.4416169832603927 correct 50 Epoch Time 0.05452394485473633
Epoch  230  loss  0.39703487948110605 correct 50 Epoch Time 0.05260014533996582
Epoch  240  loss  0.38442242483022104 correct 50 Epoch Time 0.05538296699523926
Epoch  250  loss  0.47901561673611726 correct 50 Epoch Time 0.05112195014953613
Epoch  260  loss  0.25820625786221457 correct 50 Epoch Time 0.06299495697021484
Epoch  270  loss  0.1266282593299946 correct 50 Epoch Time 0.06609368324279785
Epoch  280  loss  0.30610273714268377 correct 50 Epoch Time 0.05533003807067871
Epoch  290  loss  0.13743468226653466 correct 50 Epoch Time 0.059267282485961914
Epoch  300  loss  0.03245557179020509 correct 50 Epoch Time 0.06814908981323242
Epoch  310  loss  0.29149875267174563 correct 50 Epoch Time 0.06288576126098633
Epoch  320  loss  0.2110310360665736 correct 50 Epoch Time 0.06482100486755371
Epoch  330  loss  0.16095279809612673 correct 50 Epoch Time 0.059736013412475586
Epoch  340  loss  0.15708318412238242 correct 50 Epoch Time 0.05148005485534668
Epoch  350  loss  0.12766875807836747 correct 50 Epoch Time 0.05129289627075195
Epoch  360  loss  0.22383498118373932 correct 50 Epoch Time 0.06163191795349121
Epoch  370  loss  0.2543843933814496 correct 50 Epoch Time 0.053768157958984375
Epoch  380  loss  0.09385338188334663 correct 50 Epoch Time 0.06172919273376465
Epoch  390  loss  0.19123320446295344 correct 50 Epoch Time 0.0609891414642334
Epoch  400  loss  0.06837593548020546 correct 50 Epoch Time 0.05405306816101074
Epoch  410  loss  0.0579565682080933 correct 50 Epoch Time 0.05453372001647949
Epoch  420  loss  0.15420842739795 correct 50 Epoch Time 0.06869292259216309
Epoch  430  loss  0.1296098300049636 correct 50 Epoch Time 0.059861183166503906
Epoch  440  loss  0.21877419402340312 correct 50 Epoch Time 0.05292487144470215
Epoch  450  loss  0.13294260061936566 correct 50 Epoch Time 0.05111503601074219
Epoch  460  loss  0.09788239698867339 correct 50 Epoch Time 0.05294203758239746
Epoch  470  loss  0.14363897891256272 correct 50 Epoch Time 0.0542759895324707
Epoch  480  loss  0.0917840596844384 correct 50 Epoch Time 0.0516049861907959
Epoch  490  loss  0.1707805148615788 correct 50 Epoch Time 0.05251502990722656
```
</details>


<details>
<summary>Split (GPU, Small)</summary>
```
Epoch  0  loss  8.236565649546995 correct 36 Epoch Time 3.9555044174194336
Epoch  10  loss  5.478304718998793 correct 45 Epoch Time 2.3373823165893555
Epoch  20  loss  4.720182002095934 correct 46 Epoch Time 1.6765191555023193
Epoch  30  loss  3.719131090884679 correct 49 Epoch Time 1.5828673839569092
Epoch  40  loss  3.150136778436046 correct 48 Epoch Time 1.6413156986236572
Epoch  50  loss  2.6145418967426566 correct 47 Epoch Time 1.5920495986938477
Epoch  60  loss  3.798206782762928 correct 48 Epoch Time 2.265087842941284
Epoch  70  loss  2.7349642154065847 correct 48 Epoch Time 1.6050729751586914
Epoch  80  loss  1.7909330590797545 correct 50 Epoch Time 1.5853846073150635
Epoch  90  loss  2.804315413566799 correct 50 Epoch Time 1.6741597652435303
Epoch  100  loss  0.8613556789417878 correct 48 Epoch Time 1.565528154373169
Epoch  110  loss  0.9665791903958285 correct 48 Epoch Time 2.2242074012756348
Epoch  120  loss  2.1537527913744285 correct 48 Epoch Time 1.5838334560394287
Epoch  130  loss  0.4540789692245333 correct 49 Epoch Time 1.5777111053466797
Epoch  140  loss  1.9559619773908197 correct 49 Epoch Time 1.595426082611084
Epoch  150  loss  1.9805788671020497 correct 49 Epoch Time 1.573423147201538
Epoch  160  loss  0.9359641927778374 correct 49 Epoch Time 2.0038065910339355
Epoch  170  loss  0.4646319467108041 correct 49 Epoch Time 1.5814669132232666
Epoch  180  loss  1.9852412370594141 correct 47 Epoch Time 1.5914905071258545
Epoch  190  loss  2.925141799848379 correct 49 Epoch Time 1.7392044067382812
Epoch  200  loss  0.5123720546105347 correct 49 Epoch Time 1.6443696022033691
Epoch  210  loss  2.4706287424534352 correct 47 Epoch Time 1.7417378425598145
Epoch  220  loss  0.4818654822439565 correct 49 Epoch Time 1.6332497596740723
Epoch  230  loss  1.2331026081192555 correct 49 Epoch Time 1.5873947143554688
Epoch  240  loss  1.1858891281549797 correct 49 Epoch Time 2.0385396480560303
Epoch  250  loss  0.2703494503526956 correct 50 Epoch Time 1.6003139019012451
Epoch  260  loss  0.4184062951845051 correct 49 Epoch Time 1.5735833644866943
Epoch  270  loss  0.8963748290888685 correct 49 Epoch Time 1.5719342231750488
Epoch  280  loss  0.3908417371318662 correct 50 Epoch Time 1.5724003314971924
Epoch  290  loss  1.1177495386886398 correct 49 Epoch Time 2.334131956100464
Epoch  300  loss  0.9044334091697345 correct 50 Epoch Time 1.5864372253417969
Epoch  310  loss  1.232470867113834 correct 49 Epoch Time 1.6475234031677246
Epoch  320  loss  0.6464027360780569 correct 49 Epoch Time 1.6016299724578857
Epoch  330  loss  0.052620964921182824 correct 47 Epoch Time 1.6322646141052246
Epoch  340  loss  0.4696880281890975 correct 50 Epoch Time 2.2689368724823
Epoch  350  loss  0.6441826771898082 correct 50 Epoch Time 1.5872595310211182
Epoch  360  loss  0.7049311346986595 correct 49 Epoch Time 1.5622053146362305
Epoch  370  loss  1.5001911719360086 correct 50 Epoch Time 1.6079695224761963
Epoch  380  loss  0.38692788578629755 correct 50 Epoch Time 1.5662951469421387
Epoch  390  loss  0.194555549625001 correct 50 Epoch Time 1.881009817123413
Epoch  400  loss  0.2513279939801301 correct 50 Epoch Time 1.5802202224731445
Epoch  410  loss  0.06458921297607999 correct 50 Epoch Time 1.5690090656280518
Epoch  420  loss  0.3776912726793144 correct 49 Epoch Time 2.010488510131836
Epoch  430  loss  0.14715065822743603 correct 49 Epoch Time 1.597954273223877
Epoch  440  loss  0.05049521930728294 correct 50 Epoch Time 1.6331303119659424
Epoch  450  loss  0.8471173485546362 correct 50 Epoch Time 1.5653419494628906
Epoch  460  loss  0.4430360048694262 correct 50 Epoch Time 1.5688962936401367
Epoch  470  loss  0.11714989220452163 correct 50 Epoch Time 2.29549241065979
Epoch  480  loss  2.164696170307053 correct 50 Epoch Time 1.5898630619049072
Epoch  490  loss  0.02174050243643006 correct 50 Epoch Time 1.5869250297546387
```
</details>


<details>
<summary>XOR (CPU, Small)</summary>
```
Epoch  0  loss  6.890191377647328 correct 34 Epoch Time 5.30669379234314
Epoch  10  loss  5.701134043390645 correct 39 Epoch Time 0.056700944900512695
Epoch  20  loss  4.569230742219733 correct 41 Epoch Time 0.05525493621826172
Epoch  30  loss  4.4021725991856 correct 45 Epoch Time 0.055381059646606445
Epoch  40  loss  4.6446207565726105 correct 46 Epoch Time 0.052346229553222656
Epoch  50  loss  3.672837620893168 correct 44 Epoch Time 0.0608820915222168
Epoch  60  loss  3.182038148285591 correct 44 Epoch Time 0.06096005439758301
Epoch  70  loss  3.359687546096456 correct 47 Epoch Time 0.05308794975280762
Epoch  80  loss  5.5668408352782075 correct 44 Epoch Time 0.05615520477294922
Epoch  90  loss  2.1492180873711675 correct 48 Epoch Time 0.05500006675720215
Epoch  100  loss  2.1329263480229845 correct 48 Epoch Time 0.05860400199890137
Epoch  110  loss  2.842325112937541 correct 50 Epoch Time 0.060353994369506836
Epoch  120  loss  1.5105891684056905 correct 50 Epoch Time 0.05657792091369629
Epoch  130  loss  0.9651247815554514 correct 48 Epoch Time 0.05318403244018555
Epoch  140  loss  1.897051880318213 correct 49 Epoch Time 0.05675792694091797
Epoch  150  loss  1.1177520204407747 correct 49 Epoch Time 0.05655217170715332
Epoch  160  loss  0.6349796820848682 correct 49 Epoch Time 0.05636882781982422
Epoch  170  loss  0.8086714171249456 correct 49 Epoch Time 0.06306195259094238
Epoch  180  loss  1.060950990122216 correct 50 Epoch Time 0.05564284324645996
Epoch  190  loss  0.3867367143255893 correct 49 Epoch Time 0.053913116455078125
Epoch  200  loss  0.532478276441197 correct 49 Epoch Time 0.05765676498413086
Epoch  210  loss  0.7058330652822499 correct 50 Epoch Time 0.05404210090637207
Epoch  220  loss  0.5395241880831046 correct 50 Epoch Time 0.06047201156616211
Epoch  230  loss  0.22081922446739993 correct 50 Epoch Time 0.052778005599975586
Epoch  240  loss  0.8246101283481168 correct 50 Epoch Time 0.05364823341369629
Epoch  250  loss  0.44321177975595116 correct 49 Epoch Time 0.05733990669250488
Epoch  260  loss  0.4703524529717462 correct 50 Epoch Time 0.06213808059692383
Epoch  270  loss  1.1276817811508972 correct 50 Epoch Time 0.05480813980102539
Epoch  280  loss  0.49490934970950623 correct 49 Epoch Time 0.07618188858032227
Epoch  290  loss  0.5191430575282597 correct 50 Epoch Time 0.05671501159667969
Epoch  300  loss  0.8902398948375574 correct 50 Epoch Time 0.06135916709899902
Epoch  310  loss  0.3727247525677529 correct 50 Epoch Time 0.0533299446105957
Epoch  320  loss  0.071357938266591 correct 50 Epoch Time 0.05796980857849121
Epoch  330  loss  0.3211927133679535 correct 50 Epoch Time 0.05738377571105957
Epoch  340  loss  0.39315397376131245 correct 50 Epoch Time 0.05180811882019043
Epoch  350  loss  0.15871091369797705 correct 50 Epoch Time 0.05244803428649902
Epoch  360  loss  0.19055975659087088 correct 50 Epoch Time 0.05461883544921875
Epoch  370  loss  0.07517041135019684 correct 50 Epoch Time 0.05200004577636719
Epoch  380  loss  0.5038656223600794 correct 50 Epoch Time 0.048825979232788086
Epoch  390  loss  0.39898644783275866 correct 50 Epoch Time 0.05550074577331543
Epoch  400  loss  0.5341155943838557 correct 50 Epoch Time 0.056984901428222656
Epoch  410  loss  0.4681849712364154 correct 50 Epoch Time 0.05456209182739258
Epoch  420  loss  0.48037541829249325 correct 50 Epoch Time 0.0564579963684082
Epoch  430  loss  0.08159910394024222 correct 50 Epoch Time 0.052494049072265625
Epoch  440  loss  0.21643190939799958 correct 50 Epoch Time 0.05348610877990723
Epoch  450  loss  0.1257874957208958 correct 50 Epoch Time 0.05264902114868164
Epoch  460  loss  0.3494162004102195 correct 50 Epoch Time 0.0525660514831543
Epoch  470  loss  0.09429108272942767 correct 50 Epoch Time 0.06123685836791992
Epoch  480  loss  0.21137848817628832 correct 50 Epoch Time 0.05778622627258301
Epoch  490  loss  0.07457444093147607 correct 50 Epoch Time 0.05762195587158203
```
</details>


<details>
<summary>XOR (GPU, Small)</summary>
```
Epoch  0  loss  6.322170555929559 correct 42 Epoch Time 3.730015754699707
Epoch  10  loss  4.557191388460735 correct 47 Epoch Time 1.6281797885894775
Epoch  20  loss  2.9749898745617767 correct 46 Epoch Time 1.7016785144805908
Epoch  30  loss  3.858017775637297 correct 48 Epoch Time 1.609928846359253
Epoch  40  loss  3.0264980308864002 correct 46 Epoch Time 1.698287010192871
Epoch  50  loss  2.318672567890715 correct 48 Epoch Time 1.6317760944366455
Epoch  60  loss  1.8497504029235412 correct 48 Epoch Time 2.0551297664642334
Epoch  70  loss  1.5868487497126307 correct 48 Epoch Time 1.609452486038208
Epoch  80  loss  2.9626979182473483 correct 48 Epoch Time 2.1228950023651123
Epoch  90  loss  0.9020482272675272 correct 49 Epoch Time 1.7134969234466553
Epoch  100  loss  2.5059607788081983 correct 49 Epoch Time 1.6127049922943115
Epoch  110  loss  1.7497769029219707 correct 49 Epoch Time 1.676283359527588
Epoch  120  loss  0.4317324876400075 correct 49 Epoch Time 1.6299574375152588
Epoch  130  loss  2.2038780005714056 correct 49 Epoch Time 1.6152336597442627
Epoch  140  loss  1.661398638796375 correct 50 Epoch Time 1.6068649291992188
Epoch  150  loss  1.1238599724499339 correct 49 Epoch Time 2.208219528198242
Epoch  160  loss  0.3987571275544579 correct 49 Epoch Time 1.6157138347625732
Epoch  170  loss  1.1859733244009358 correct 50 Epoch Time 2.021590232849121
Epoch  180  loss  0.9887420819889023 correct 50 Epoch Time 1.6298363208770752
Epoch  190  loss  0.1866178209681641 correct 50 Epoch Time 1.6148946285247803
Epoch  200  loss  0.09653154808391656 correct 50 Epoch Time 1.7023327350616455
Epoch  210  loss  0.31380349832221344 correct 50 Epoch Time 1.6238353252410889
Epoch  220  loss  0.6478009401327153 correct 50 Epoch Time 1.7223460674285889
Epoch  230  loss  0.3723172353419164 correct 50 Epoch Time 1.636932134628296
Epoch  240  loss  0.4617914471915075 correct 50 Epoch Time 2.2045021057128906
Epoch  250  loss  0.6173740871713629 correct 50 Epoch Time 1.6267235279083252
Epoch  260  loss  1.160636404141652 correct 50 Epoch Time 1.939347743988037
Epoch  270  loss  0.7644434060944083 correct 50 Epoch Time 1.6727566719055176
Epoch  280  loss  0.31149492755474295 correct 50 Epoch Time 1.6352958679199219
Epoch  290  loss  0.48700128603918735 correct 50 Epoch Time 1.646261215209961
Epoch  300  loss  0.2500552845136135 correct 50 Epoch Time 1.6220676898956299
Epoch  310  loss  0.25998762114117513 correct 50 Epoch Time 1.699347734451294
Epoch  320  loss  0.3570295857722735 correct 50 Epoch Time 1.658735752105713
Epoch  330  loss  0.28035604586394547 correct 50 Epoch Time 2.2680776119232178
Epoch  340  loss  0.36053001401423257 correct 50 Epoch Time 1.698695182800293
Epoch  350  loss  0.11821718471245377 correct 50 Epoch Time 2.094243288040161
Epoch  360  loss  0.09357594496090682 correct 50 Epoch Time 1.6094999313354492
Epoch  370  loss  0.4018214597679668 correct 50 Epoch Time 1.6290407180786133
Epoch  380  loss  0.08598768980349016 correct 50 Epoch Time 1.6076226234436035
Epoch  390  loss  0.5641823358733038 correct 50 Epoch Time 1.6175458431243896
Epoch  400  loss  0.40420072910822 correct 50 Epoch Time 1.6215050220489502
Epoch  410  loss  0.14040066925875036 correct 50 Epoch Time 1.6098670959472656
Epoch  420  loss  0.0869024835539823 correct 50 Epoch Time 2.344416379928589
Epoch  430  loss  0.30233021649764064 correct 50 Epoch Time 1.6241416931152344
Epoch  440  loss  0.47413257190987623 correct 50 Epoch Time 1.921487808227539
Epoch  450  loss  0.3097656887407788 correct 50 Epoch Time 1.615079402923584
Epoch  460  loss  0.09267180239847754 correct 50 Epoch Time 1.5988364219665527
Epoch  470  loss  0.053368661112664625 correct 50 Epoch Time 1.6015703678131104
Epoch  480  loss  0.33244713053594654 correct 50 Epoch Time 1.6198186874389648
Epoch  490  loss  0.26126361595294106 correct 50 Epoch Time 2.085435390472412
```
</details>


<details>
<summary>XOR (CPU, Big)</summary>
```
Epoch  0  loss  10.552810699733755  correct  29  Epoch Time 30.428833484649658
Epoch  10  loss  4.39979273567821  correct  45  Epoch Time 0.28137683868408203
Epoch  20  loss  6.222284424110658  correct  38  Epoch Time 0.2838103771209717
Epoch  30  loss  1.3602966857970868  correct  45  Epoch Time 0.2936115264892578
Epoch  40  loss  4.404937588941485  correct  41  Epoch Time 0.5097393989562988
Epoch  50  loss  2.0935785044543644  correct  46  Epoch Time 0.2859015464782715
Epoch  60  loss  2.713558016779565  correct  47  Epoch Time 0.2957572937011719
Epoch  70  loss  2.243980852616881  correct  46  Epoch Time 0.3105733394622803
Epoch  80  loss  2.981556672119811  correct  45  Epoch Time 0.5962865352630615
Epoch  90  loss  3.098591626953965  correct  42  Epoch Time 0.28256988525390625
Epoch  100  loss  3.563016283671855  correct  48  Epoch Time 0.2960813045501709
Epoch  110  loss  3.5594755933154496  correct  46  Epoch Time 0.2931368350982666
Epoch  120  loss  3.360322227072642  correct  46  Epoch Time 0.5846505165100098
Epoch  130  loss  1.8709994864481705  correct  49  Epoch Time 0.2897651195526123
Epoch  140  loss  2.7270548558954735  correct  45  Epoch Time 0.29209041595458984
Epoch  150  loss  0.6111139464018301  correct  48  Epoch Time 0.28166866302490234
Epoch  160  loss  3.1426052719766884  correct  48  Epoch Time 0.6003763675689697
Epoch  170  loss  3.518642944108538  correct  47  Epoch Time 0.2947070598602295
Epoch  180  loss  1.5877813919956232  correct  45  Epoch Time 0.2817838191986084
Epoch  190  loss  1.6126850289559953  correct  49  Epoch Time 0.2866184711456299
Epoch  200  loss  0.8284981781485823  correct  49  Epoch Time 0.6332621574401855
Epoch  210  loss  1.9256248387536876  correct  49  Epoch Time 0.2842977046966553
Epoch  220  loss  2.3261569642385127  correct  47  Epoch Time 0.287243127822876
Epoch  230  loss  1.214465931655418  correct  49  Epoch Time 0.28569841384887695
Epoch  240  loss  2.0479188212711796  correct  47  Epoch Time 0.5087502002716064
Epoch  250  loss  1.6742350748799166  correct  49  Epoch Time 0.28215670585632324
Epoch  260  loss  3.093026168544612  correct  49  Epoch Time 0.29148411750793457
Epoch  270  loss  1.3154071027675918  correct  48  Epoch Time 0.30651068687438965
Epoch  280  loss  1.1887306373005722  correct  45  Epoch Time 0.33915185928344727
Epoch  290  loss  0.8298497827930726  correct  49  Epoch Time 0.29044651985168457
Epoch  300  loss  1.8013237388677017  correct  49  Epoch Time 0.30201101303100586
Epoch  310  loss  1.5632507541974325  correct  49  Epoch Time 0.2841799259185791
Epoch  320  loss  0.5016407042809233  correct  49  Epoch Time 0.282745361328125
Epoch  330  loss  0.8845688093479337  correct  49  Epoch Time 0.2829904556274414
Epoch  340  loss  1.6546645138869251  correct  47  Epoch Time 0.29515600204467773
Epoch  350  loss  3.453897562762501  correct  44  Epoch Time 0.28810572624206543
Epoch  360  loss  1.2479795992467657  correct  49  Epoch Time 0.28362417221069336
Epoch  370  loss  1.2503954263111072  correct  49  Epoch Time 0.2925682067871094
Epoch  380  loss  1.1704576553667736  correct  49  Epoch Time 0.28266263008117676
Epoch  390  loss  1.5036608956419164  correct  50  Epoch Time 0.2849771976470947
Epoch  400  loss  1.045680118599766  correct  49  Epoch Time 0.28623127937316895
Epoch  410  loss  1.314809181830843  correct  50  Epoch Time 0.3015775680541992
Epoch  420  loss  2.5440649627385  correct  49  Epoch Time 0.2852456569671631
Epoch  430  loss  0.7035551477916738  correct  50  Epoch Time 0.3101038932800293
Epoch  440  loss  1.4597181046011471  correct  50  Epoch Time 0.28812217712402344
Epoch  450  loss  1.4296089547428592  correct  50  Epoch Time 0.2829439640045166
Epoch  460  loss  0.1787217332141405  correct  50  Epoch Time 0.2831716537475586
Epoch  470  loss  1.1330155001234765  correct  50  Epoch Time 0.2816905975341797
Epoch  480  loss  1.0157262551726822  correct  50  Epoch Time 0.2817997932434082
Epoch  490  loss  0.34438641122444913  correct  50  Epoch Time 0.2868154048919678
```
</details>


<details>
<summary>XOR (GPU, Big)</summary>
```
Epoch  0  loss  18.33445429860606  correct  27  Epoch Time 5.164613485336304
Epoch  10  loss  3.4441247466654406  correct  43  Epoch Time 1.9572865962982178
Epoch  20  loss  3.1765152715430327  correct  31  Epoch Time 1.962019920349121
Epoch  30  loss  2.4954697299081277  correct  47  Epoch Time 2.0607433319091797
Epoch  40  loss  0.9496218542982041  correct  47  Epoch Time 2.4064157009124756
Epoch  50  loss  2.3989433728795633  correct  47  Epoch Time 1.9619405269622803
Epoch  60  loss  3.413719856976062  correct  46  Epoch Time 1.9897680282592773
Epoch  70  loss  2.381565655305448  correct  48  Epoch Time 2.691392183303833
Epoch  80  loss  2.660354968774216  correct  47  Epoch Time 2.043222665786743
Epoch  90  loss  2.4287082535247224  correct  48  Epoch Time 1.977492332458496
Epoch  100  loss  1.0121077795845521  correct  48  Epoch Time 2.16451358795166
Epoch  110  loss  0.8385567485279248  correct  50  Epoch Time 1.9736196994781494
Epoch  120  loss  2.491340434989916  correct  48  Epoch Time 1.9560377597808838
Epoch  130  loss  1.7676071278998091  correct  49  Epoch Time 2.0707015991210938
Epoch  140  loss  0.21214811666757225  correct  49  Epoch Time 2.822059154510498
Epoch  150  loss  1.2074033140027483  correct  49  Epoch Time 2.0021212100982666
Epoch  160  loss  0.8225433914000941  correct  47  Epoch Time 1.9756524562835693
Epoch  170  loss  0.5422393902668969  correct  50  Epoch Time 2.1615867614746094
Epoch  180  loss  0.2476701502925304  correct  49  Epoch Time 2.0442757606506348
Epoch  190  loss  0.5158953781809927  correct  49  Epoch Time 1.963273525238037
Epoch  200  loss  1.0595364727665468  correct  49  Epoch Time 2.007188558578491
Epoch  210  loss  0.7596062241410294  correct  50  Epoch Time 2.4792065620422363
Epoch  220  loss  0.26157958367188533  correct  50  Epoch Time 2.0266504287719727
Epoch  230  loss  0.926446239574852  correct  49  Epoch Time 1.9686014652252197
Epoch  240  loss  0.24371627741346719  correct  50  Epoch Time 2.7017288208007812
Epoch  250  loss  0.9996633362630173  correct  49  Epoch Time 2.004945755004883
Epoch  260  loss  0.1822214403251717  correct  50  Epoch Time 2.0577316284179688
Epoch  270  loss  0.23402621149557884  correct  50  Epoch Time 1.9943809509277344
Epoch  280  loss  0.4968192129063987  correct  50  Epoch Time 2.2846388816833496
Epoch  290  loss  0.4257667575282168  correct  50  Epoch Time 1.9893722534179688
Epoch  300  loss  0.5523081816477287  correct  49  Epoch Time 2.0195658206939697
Epoch  310  loss  0.7213124626726911  correct  50  Epoch Time 2.808286666870117
Epoch  320  loss  0.027469700989154657  correct  50  Epoch Time 1.9713332653045654
Epoch  330  loss  0.027799713234249762  correct  49  Epoch Time 1.9874699115753174
Epoch  340  loss  0.2283277585297141  correct  50  Epoch Time 2.3340649604797363
Epoch  350  loss  0.49345017299338073  correct  49  Epoch Time 2.0463967323303223
Epoch  360  loss  0.039879521121818584  correct  49  Epoch Time 2.0337471961975098
Epoch  370  loss  0.3522202110869229  correct  49  Epoch Time 1.9598186016082764
Epoch  380  loss  0.7693803623031036  correct  50  Epoch Time 2.1389172077178955
Epoch  390  loss  0.3719415569809946  correct  49  Epoch Time 1.9631803035736084
Epoch  400  loss  0.43316672990903665  correct  50  Epoch Time 2.0207927227020264
Epoch  410  loss  0.02685382262694054  correct  50  Epoch Time 2.6072309017181396
Epoch  420  loss  0.15749529240455948  correct  49  Epoch Time 1.976989984512329
Epoch  430  loss  3.3252515809514227  correct  49  Epoch Time 1.9988839626312256
Epoch  440  loss  0.9343970903991833  correct  49  Epoch Time 2.794128656387329
Epoch  450  loss  0.32642695675164324  correct  50  Epoch Time 1.992530107498169
Epoch  460  loss  0.0344128328638602  correct  50  Epoch Time 1.9990363121032715
Epoch  470  loss  0.1486290321245423  correct  49  Epoch Time 2.020720958709717
Epoch  480  loss  0.1446915928443576  correct  50  Epoch Time 1.9579944610595703
Epoch  490  loss  0.32916343654617974  correct  50  Epoch Time 2.619441270828247
```
</details>
