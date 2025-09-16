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
