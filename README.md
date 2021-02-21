# CS5340-project
NUS CS5340 Uncertainty Modelling project


### Abstract
https://docs.google.com/document/d/1YFZ5YdIZEyuBaCcbRfI52tMouADg_OQU0YzcgFwHHLE/edit#


### Conversion of Pipeline to Python:
| Components        | Location in Python-Pipeline   | Done?  |
| ----------------- |:-------------:| -----:|
| NBSR_core/obs_for_SR.m | obs_for_SR.py | [X] |
| NBSR_core/get_img_sz.m | get_img_sz.py  | [X] |
| NBSR_core/my_conv2.m   | my_conv2.py   | [X] |
| NBSR_core/Eval/my_psnr.m   |  eval/my_psnr.py  | [X] |
| NBSR_core/Eval/ssim_index.m   |  eval/ssim_index.py   | [X] |
| NBSR_core/+pml/+image_proc/convmtxn.m  |               | [ ] |
| NBSR_core/+pml/+image_proc/imfiltermtx.m   |               | [ ] |
| NBSR_core/+pml/+image_proc/make_convn_mat.m   |               | [ ] |
| NBSR_core/+pml/+image_proc/make_imfilter_mat.m   |               | [ ] |
| NBSR_core/+pml/+image_proc/psnr.m   |               | [ ] |
| NBSR_core/+pml/+image_proc/ssim_index.m   |               | [ ] |
| NBSR_core/+pml/+distributions/@foe   |               | [ ] |
| NBSR_core/+pml/+distributions/@gsm   |               | [ ] |
| NBSR_core/+pml/+distributions/@gsm_foe   |               | [ ] |
| NBSR_core/+pml/+distributions/@gsm_pairwise_mrf   |               | [ ] |
| NBSR_core/+pml/+distributions/@pairwise_mrf   |               | [ ] |
| NBSR_core/+pml/+distributions/density.m   |               | [ ] |
| NBSR_core/+pml/+distributions/distribution.m   |               | [ ] |
| demo_super_resolution_color.m   |               | [ ] |
| demo_super_resolution.m   |               | [ ] |
