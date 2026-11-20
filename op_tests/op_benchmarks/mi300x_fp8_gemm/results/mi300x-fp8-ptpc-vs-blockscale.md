# MI300X FP8 GEMM comparison

Median TFLOP/s on one 304-CU gfx942 GPU. Each cell includes all row chunks. † marks a chunked result. A dash indicates an unavailable or rejected measurement. The CSV gives the chunk sizes and reasons. [Tuning guide](../../../../docs/mi300x-fp8-gemm-tuning-guide.md) · [Unrounded CSV](mi300x-fp8-ptpc-vs-blockscale.csv) · [Raw timing samples](cross-scaling/)

## Projection 4,096 → 1,536

### Per-token activation and per-channel weight

| Kernel | M=2,048 | M=4,096 | M=8,192 | M=16,384 | M=32,768 | M=65,536 | M=131,072 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| PyTorch, ordinary | 676.5 | 837.8 | 1,021.4 | 931.0 | 1,063.0 | 1,046.4 | 1,036.4 |
| tgemm hipBLASLt, ordinary | 687.7 | 840.2 | 945.6 | 924.3 | 1,060.1 | 1,050.2 | 1,023.6 |
| tgemm hipBLASLt, packed | 759.3 | 904.8 | 1,112.6 | 1,127.5 | 1,144.6 | 1,118.3 | 1,085.9 |
| AITER PTPC operator, packed | 670.8 | 721.7 | 898.6 | 904.6 | 953.8 | 917.5 | 906.4 |

### K128 block scaling

| Kernel | M=2,048 | M=4,096 | M=8,192 | M=16,384 | M=32,768 | M=65,536 | M=131,072 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Triton, ordinary | 169.5 | 210.1 | 230.1 | 244.4 | 248.7 | 249.6 | 248.1 |
| Classic CK, ordinary | 522.1 | 567.7 | 699.3 | 705.7 | 718.1 | 695.8 | 711.4 |
| Classic CK, packed | 558.9 | 697.6 | 762.9 | 807.6 | 834.1 | 802.4 | 809.6 |
| CK Tile, packed | 457.6 | 527.9 | 599.4 | 608.1 | 648.8 | 649.6 | 670.4 |
| ASM, packed | 646.4 | 711.8 | 759.0 | 772.7 | 772.2 | 734.0 | 726.5 |
| OPUS, packed | 502.4 | 460.6† | 456.3† | 478.7† | 491.7† | 462.6† | 458.2† |
| FlyDSL original tuned port, packed | 263.8 | 334.4 | 402.5 | 413.4 | 423.8 | 395.7 | 367.3 |
| FlyDSL two-stage LDS, M-first | 384.8 | 609.3 | 614.8 | 754.6 | 680.7 | 603.9 | 518.4 |
| FlyDSL two-stage LDS, N-first | 353.6 | 506.9 | 566.9 | 685.6 | 687.5 | 694.9 | 713.1 |
| FlyDSL transposed register cache, M-first | 460.6 | 484.0 | 614.5 | 601.1 | 620.4 | 562.7 | 480.6 |

### One activation multiplier and per-channel weight

| Kernel | M=2,048 | M=4,096 | M=8,192 | M=16,384 | M=32,768 | M=65,536 | M=131,072 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| PyTorch, ordinary | 697.6 | 853.6 | 1,002.5 | 979.8 | 1,051.4 | 1,041.9 | 1,032.8 |
| tgemm hipBLASLt, ordinary | 676.9 | 826.6 | 992.3 | 944.0 | 1,069.0 | 1,035.0 | 1,038.7 |
| tgemm hipBLASLt, packed | 758.3 | 942.2 | 1,115.1 | 1,121.1 | 1,124.7 | 1,120.2 | 1,076.9 |
| AITER PTPC operator, packed | 656.8 | 716.1 | 907.5 | 919.0 | 965.9 | 908.9 | 911.4 |

## Projection 1,536 → 32,768

### Per-token activation and per-channel weight

| Kernel | M=2,048 | M=4,096 | M=8,192 | M=16,384 | M=32,768 | M=65,536 | M=131,072 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| PyTorch, ordinary | 835.8 | 872.4 | 891.3 | 887.2 | 901.7 | 896.2 | 888.9 |
| tgemm hipBLASLt, ordinary | 874.3 | 862.5 | 909.4 | 908.8 | 912.5 | 915.4 | 908.5 |
| tgemm hipBLASLt, packed | 980.3 | 957.4 | 935.7 | 953.3 | 960.7 | 978.2 | 958.0 |
| AITER PTPC operator, packed | 857.8 | 863.3 | 843.8 | 835.0 | 831.5† | 827.2† | 818.6† |

### K128 block scaling

| Kernel | M=2,048 | M=4,096 | M=8,192 | M=16,384 | M=32,768 | M=65,536 | M=131,072 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Triton, ordinary | 274.5 | 282.5 | 278.5 | 280.8 | 263.4 | 262.7 | 261.2 |
| Classic CK, ordinary | 703.3 | 700.8 | 672.9 | 689.2 | 690.5 | 688.1† | 686.5† |
| Classic CK, packed | 766.2 | 743.9 | 748.0 | 746.9 | 738.5 | 740.8† | 736.1† |
| CK Tile, packed | 641.3 | 640.3 | 629.0 | 620.5 | 618.7† | 613.2† | 609.4† |
| ASM, packed | 586.9 | 598.1 | 647.3 | 639.7 | 637.1 | 636.5 | 633.4 |
| OPUS, packed | — | — | — | — | — | — | — |
| FlyDSL original tuned port, packed | 439.5 | 467.5 | 472.7 | 472.8 | 445.5 | 437.4 | 425.1 |
| FlyDSL two-stage LDS, M-first | 692.5 | 720.5 | 714.9 | 707.0 | 614.3 | 609.2 | 607.5 |
| FlyDSL two-stage LDS, N-first | 707.2 | 681.8 | 656.0 | 634.2 | 627.0 | 621.8 | 619.1 |
| FlyDSL transposed register cache, M-first | 626.1 | 639.8 | 651.2 | 642.9 | 592.1 | 579.8 | 568.1 |

### One activation multiplier and per-channel weight

| Kernel | M=2,048 | M=4,096 | M=8,192 | M=16,384 | M=32,768 | M=65,536 | M=131,072 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| PyTorch, ordinary | 818.8 | 862.3 | 894.9 | 895.6 | 903.0 | 896.5 | 890.5 |
| tgemm hipBLASLt, ordinary | 817.0 | 849.6 | 895.1 | 912.9 | 912.3 | 914.7 | 910.2 |
| tgemm hipBLASLt, packed | 998.4 | 954.7 | 973.1 | 969.7 | 973.1 | 981.2 | 959.2 |
| AITER PTPC operator, packed | 900.1 | 867.9 | 838.4 | 835.7 | 828.4† | 825.2† | 823.1† |

## Projection 32,768 → 4,096

### Per-token activation and per-channel weight

| Kernel | M=2,048 | M=4,096 | M=8,192 | M=16,384 | M=32,768 | M=65,536 | M=131,072 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| PyTorch, ordinary | 1,050.8 | 1,145.2 | 1,120.0 | 1,081.7 | 1,068.4 | 1,074.2 | 1,045.9 |
| tgemm hipBLASLt, ordinary | 1,057.9 | 1,140.7 | 1,115.6 | 1,091.1 | 1,099.8 | 1,095.6 | 1,103.9 |
| tgemm hipBLASLt, packed | 1,098.7 | 1,173.8 | 1,154.8 | 1,172.4 | 1,180.9 | 1,177.0 | 1,170.5 |
| AITER PTPC operator, packed | 1,003.5 | 978.3 | 978.6 | 970.9 | 971.7 | 965.1† | 964.6† |

### K128 block scaling

| Kernel | M=2,048 | M=4,096 | M=8,192 | M=16,384 | M=32,768 | M=65,536 | M=131,072 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Triton, ordinary | 252.3 | 245.5 | 255.9 | 255.2 | 246.8 | 262.3 | 277.5† |
| Classic CK, ordinary | 594.5 | 609.1 | 602.3 | 580.1 | 569.4 | 563.1 | 594.5† |
| Classic CK, packed | 809.5 | 802.7 | 821.0 | 837.1 | 828.7 | 838.4 | 836.3† |
| CK Tile, packed | 700.5 | 695.7 | 715.7 | 711.9 | 703.2 | 724.2† | 722.6† |
| ASM, packed | 890.4 | 872.9 | 899.9 | 894.4 | 895.9 | 829.4 | 787.9 |
| OPUS, packed | — | — | — | — | — | — | — |
| FlyDSL original tuned port, packed | 414.1 | 415.0 | 448.3 | 408.9 | 363.9 | 366.9 | 348.7 |
| FlyDSL two-stage LDS, M-first | 781.7 | 804.2 | 740.1 | 715.3 | 619.4 | 538.5 | 533.6 |
| FlyDSL two-stage LDS, N-first | 792.2 | 772.4 | 786.5 | 808.7 | 800.2 | 729.2 | 712.4 |
| FlyDSL transposed register cache, M-first | 650.9 | 653.5 | 650.0 | 590.3 | 512.2 | 488.5 | 484.6 |

### One activation multiplier and per-channel weight

| Kernel | M=2,048 | M=4,096 | M=8,192 | M=16,384 | M=32,768 | M=65,536 | M=131,072 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| PyTorch, ordinary | 1,065.7 | 1,134.0 | 1,116.2 | 1,081.4 | 1,064.9 | 1,074.3 | 1,045.1 |
| tgemm hipBLASLt, ordinary | 1,045.0 | 1,127.5 | 1,116.8 | 1,094.0 | 1,102.6 | 1,094.6 | 1,097.7 |
| tgemm hipBLASLt, packed | 1,113.7 | 1,191.3 | 1,162.7 | 1,174.3 | 1,179.8 | 1,177.4 | 1,171.3 |
| AITER PTPC operator, packed | 1,002.3 | 969.0 | 983.5 | 973.6 | 971.4 | 965.6† | 966.5† |
