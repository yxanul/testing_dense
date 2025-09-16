python train_dense_gpu.py --dataset c4 --split train --tokenizer gpt2 --layers 12 --heads 11 --d-model 704 --block-size 1024 --batch 8 --grad-accum 2 --steps 500 --ckpt --compile --amp bf16 --opt-list adamw,muon_adamw,muon_sophia,sophia --num-workers 4 --shuffle-buffer 4000
Model params: 107.5M  (V=50257, L=12, H=11, D=704)
[adamw       ] step  100/500  loss 33.0994  avg 321.91 ms/it
[adamw       ] step  200/500  loss 22.3724  avg 323.90 ms/it
[adamw       ] step  300/500  loss 14.7124  avg 325.57 ms/it
[adamw       ] step  400/500  loss 11.9422  avg 326.74 ms/it
[adamw       ] step  500/500  loss 11.0755  avg 327.55 ms/it

[muon_adamw  ] step  100/500  loss 28.6534  avg 370.00 ms/it
[muon_adamw  ] step  200/500  loss 18.1927  avg 370.80 ms/it
[muon_adamw  ] step  300/500  loss 12.0434  avg 371.09 ms/it
[muon_adamw  ] step  400/500  loss 9.9368  avg 371.20 ms/it
[muon_adamw  ] step  500/500  loss 8.8988  avg 371.29 ms/it

[muon_sophia ] step  100/500  loss 14.8350  avg 572.00 ms/it
[muon_sophia ] step  200/500  loss 8.5316  avg 485.26 ms/it
[muon_sophia ] step  300/500  loss 8.1145  avg 456.58 ms/it
[muon_sophia ] step  400/500  loss 7.6950  avg 442.40 ms/it
[muon_sophia ] step  500/500  loss 7.3435  avg 434.00 ms/it

[sophia      ] step  100/500  loss 19.3555  avg 406.72 ms/it
[sophia      ] step  200/500  loss 10.4711  avg 385.08 ms/it
[sophia      ] step  300/500  loss 9.7164  avg 378.00 ms/it
[sophia      ] step  400/500  loss 9.3258  avg 374.38 ms/it
[sophia      ] step  500/500  loss 8.8878  avg 372.11 ms/it

=== Summary (GPU) ===
adamw         327.55 ms/it       50020 tok/s   loss 11.0755
muon_adamw    371.29 ms/it       44128 tok/s   loss 8.8988
muon_sophia   434.00 ms/it       37752 tok/s   loss 7.3435
sophia        372.11 ms/it       44030 tok/s   loss 8.8878


Seems like the muon sophia outperforms the other optimizers by a lot . More tests are needed



python train_dense_gpu.py --dataset c4 --dataset-config en --split train --tokenizer gpt2 --layers 12 --heads 11 --d-model 704 --block-size 1024 --batch 8 --grad-accum 2 --steps 1500 --ckpt --compile --amp bf16 --opt-list adamw,muon_adamw,muon_sophia,sophia --num-workers 4 --shuffle-buffer 4000 --prefetch-factor 8 --val-split validation --val-every 200 --val-iters 50 --csv runs/c4_gpt2_pad128_s1500.csv

[adamw       ]   (val) loss 21.2550  ppl 1701881090.4
[adamw       ] step  300/1500  loss 14.7494  avg 307.84 ms/it
[adamw       ]   (val) loss 10.8850  ppl 53369.8
[adamw       ]   (val) loss 8.7106  ppl 6067.0
[adamw       ] step  600/1500  loss 9.8962  avg 313.24 ms/it
[adamw       ]   (val) loss 8.2056  ppl 3661.5
[adamw       ] step  900/1500  loss 9.1296  avg 315.93 ms/it
[adamw       ]   (val) loss 7.7794  ppl 2390.8
[adamw       ]   (val) loss 7.6269  ppl 2052.6
[adamw       ] step 1200/1500  loss 8.5705  avg 317.55 ms/it
[adamw       ]   (val) loss 7.4725  ppl 1759.0
[adamw       ] step 1500/1500  loss 8.3177  avg 319.08 ms/it
[muon_adamw  ]   (val) loss 17.7987  ppl 53690680.7
[muon_adamw  ] step  300/1500  loss 11.8170  avg 362.72 ms/it
[muon_adamw  ]   (val) loss 9.8975  ppl 19880.6
[muon_adamw  ]   (val) loss 8.1456  ppl 3448.3
[muon_adamw  ] step  600/1500  loss 8.0024  avg 364.56 ms/it
[muon_adamw  ]   (val) loss 7.5631  ppl 1925.9
[muon_adamw  ] step  900/1500  loss 7.4852  avg 365.51 ms/it
[muon_adamw  ]   (val) loss 7.2488  ppl 1406.3
[muon_adamw  ]   (val) loss 7.0600  ppl 1164.4
[muon_adamw  ] step 1200/1500  loss 7.0269  avg 366.02 ms/it
[muon_adamw  ]   (val) loss 6.9550  ppl 1048.4
[muon_adamw  ] step 1500/1500  loss 6.6880  avg 366.42 ms/it

[muon_sophia ]   (val) loss 9.0075  ppl 8164.1
[muon_sophia ] step  300/1500  loss 8.2064  avg 424.95 ms/it
[muon_sophia ]   (val) loss 7.7506  ppl 2323.0
[muon_sophia ]   (val) loss 7.2625  ppl 1425.8
[muon_sophia ] step  600/1500  loss 6.7424  avg 407.94 ms/it
[muon_sophia ]   (val) loss 6.8850  ppl 977.5
[muon_sophia ] step  900/1500  loss 7.3805  avg 402.50 ms/it
[muon_sophia ]   (val) loss 6.7094  ppl 820.1
[muon_sophia ]   (val) loss 6.4606  ppl 639.5
[muon_sophia ] step 1200/1500  loss 6.2735  avg 399.79 ms/it
[muon_sophia ]   (val) loss 6.2888  ppl 538.5
[muon_sophia ] step 1500/1500  loss 6.4871  avg 397.99 ms/it


[sophia      ]   (val) loss 9.4213  ppl 12348.0
[sophia      ] step  300/1500  loss 9.6969  avg 352.11 ms/it
[sophia      ]   (val) loss 8.5388  ppl 5109.0
[sophia      ]   (val) loss 8.0469  ppl 3124.0
[sophia      ] step  600/1500  loss 8.2328  avg 352.09 ms/it
[sophia      ]   (val) loss 7.8656  ppl 2606.1
[sophia      ] step  900/1500  loss 8.7234  avg 351.61 ms/it
[sophia      ]   (val) loss 7.6556  ppl 2112.5
[sophia      ]   (val) loss 7.3931  ppl 1624.8
[sophia      ] step 1200/1500  loss 7.4909  avg 351.34 ms/it
[sophia      ]   (val) loss 7.2906  ppl 1466.5
[sophia      ] step 1500/1500  loss 7.7590  avg 351.22 ms/it

=== Summary (GPU) ===
adamw         319.08 ms/it       51348 tok/s   loss 8.3177
muon_adamw    366.42 ms/it       44713 tok/s   loss 6.6880
muon_sophia   397.99 ms/it       41167 tok/s   loss 6.4871
sophia        351.22 ms/it       46648 tok/s   loss 7.7590



[adamw       ]   (val) loss 13.6200  ppl 822414.7
[adamw       ]   (val) loss 8.8419  ppl 6918.0
[adamw       ] step  600/3000  loss 10.1039  avg 271.25 ms/it
[adamw       ]   (val) loss 7.9044  ppl 2709.1
[adamw       ]   (val) loss 7.8344  ppl 2526.0
[adamw       ] step 1200/3000  loss 8.5869  avg 277.24 ms/it
[adamw       ]   (val) loss 7.2919  ppl 1468.3
[adamw       ]   (val) loss 7.3450  ppl 1548.4
[adamw       ] step 1800/3000  loss 8.2839  avg 280.25 ms/it
[adamw       ]   (val) loss 7.2069  ppl 1348.7
[adamw       ]   (val) loss 7.1944  ppl 1331.9
[adamw       ] step 2400/3000  loss 8.2561  avg 282.08 ms/it
[adamw       ]   (val) loss 7.1106  ppl 1224.9
[adamw       ]   (val) loss 7.0681  ppl 1173.9
[adamw       ] step 3000/3000  loss 7.3661  avg 283.36 ms/it

[muon_sophia ]   (val) loss 8.2212  ppl 3719.1
[muon_sophia ]   (val) loss 7.2462  ppl 1402.8
[muon_sophia ] step  600/3000  loss 6.7445  avg 362.74 ms/it
[muon_sophia ]   (val) loss 6.7919  ppl 890.6
[muon_sophia ]   (val) loss 6.5181  ppl 677.3
[muon_sophia ] step 1200/3000  loss 6.2802  avg 355.53 ms/it
[muon_sophia ]   (val) loss 6.2687  ppl 527.8
[muon_sophia ]   (val) loss 6.0488  ppl 423.6
[muon_sophia ] step 1800/3000  loss 5.9346  avg 353.20 ms/it
[muon_sophia ]   (val) loss 5.8419  ppl 344.4
[muon_sophia ]   (val) loss 5.5956  ppl 269.2
[muon_sophia ] step 2400/3000  loss 5.7737  avg 352.08 ms/it
[muon_sophia ]   (val) loss 5.4819  ppl 240.3
[muon_sophia ]   (val) loss 5.2950  ppl 199.3
[muon_sophia ] step 3000/3000  loss 5.4582  avg 351.35 ms/it


[muon_adamw  ]   (val) loss 11.9138  ppl 149305.5
[muon_adamw  ]   (val) loss 8.1069  ppl 3317.2
[muon_adamw  ] step  600/3000  loss 8.0479  avg 327.36 ms/it
[muon_adamw  ]   (val) loss 7.4262  ppl 1679.5
[muon_adamw  ]   (val) loss 7.1137  ppl 1228.7
[muon_adamw  ] step 1200/3000  loss 7.0880  avg 327.75 ms/it
[muon_adamw  ]   (val) loss 6.9650  ppl 1058.9
[muon_adamw  ]   (val) loss 6.8269  ppl 922.3
[muon_adamw  ] step 1800/3000  loss 6.8794  avg 327.88 ms/it
[muon_adamw  ]   (val) loss 6.7156  ppl 825.2
[muon_adamw  ]   (val) loss 6.6638  ppl 783.5
[muon_adamw  ] step 2400/3000  loss 6.9889  avg 328.23 ms/it
[muon_adamw  ]   (val) loss 6.5762  ppl 717.8
[muon_adamw  ]   (val) loss 6.4644  ppl 641.9
[muon_adamw  ] step 3000/3000  loss 6.3615  avg 328.31 ms/it

=== Summary (GPU) ===
adamw         283.36 ms/it       57821 tok/s   loss 7.3661
muon_sophia   351.35 ms/it       46632 tok/s   loss 5.4582
muon_adamw    328.31 ms/it       49904 tok/s   loss 6.3615


It seems like muon sophia is the right choice. 