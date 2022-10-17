# AngleNet_OU

#### OUMVLP
Since the huge differences between OUMVLP and CASIA-B, the network setting on OUMVLP is slightly different.
- The alternated network's code can be found at `./work/OUMVLP_network`. Use them to replace the corresponding files in `./model/network`.
- The checkpoint can be found [here](https://1drv.ms/u/s!AurT2TsSKdxQuWN8drzIv_phTR5m?e=Gfbl3m).
- In `./config.py`, modify `'batch_size': (8, 16)` into `'batch_size': (32,16)`.
- Prepare your OUMVLP dataset according to the instructions in [Dataset & Preparation](#dataset--preparation).
