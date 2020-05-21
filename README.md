# Iterative Network for Image Super-Resolution

*Yuqing Liu, Shiqi Wang, Jian Zhang, Shanshe Wang, Siwei Ma and Wen Gao. "Iterative Network for Image Super-Resolution"* [arxiv](https://arxiv.org/abs/2005.09964)

![model](./Images/network.png)

Network structure

![solversr](./Images/solversr.png)

Structure of Solver-SR

In this paper, we propose a substantially different approach relying on the iterative optimization on HR space with an iterative super-resolution network (ISRN). We first analyze the observation model of image SR problem, inspiring a feasible solution by mimicking and fusing each iteration in a more general and efficient manner. Considering the drawbacks of batch normalization, we propose a feature normalization (F-Norm) method to regulate the features in network.  Furthermore, a novel block with F-Norm is developed to improve the network representation, termed as FNB.  Residual-in-residual structure is proposed to form a very deep network, which groups FNBs with a long skip connection for better information delivery and stabling the training phase. Extensive experimental results on testing benchmarks with bicubic (BI) degradation show our ISRN can not only recover more structural information, but also achieve competitive or better PSNR/SSIM results with much fewer parameters. Besides BI, we simulate the real-world degradation with blur-downscale (BD) and downscale-noise (DN). ISRN and its extension ISRN+ both achieve better performance with BD and DN degradation models.

## Code
Comming soon.

## Results
![psnrssim](./Images/psnrssim.png)


## Citation
Please kindly cite our paper when using this project for your research.
```
@misc{liu2020iterative,
    title={Iterative Network for Image Super-Resolution},
    author={Yuqing Liu and Shiqi Wang and Jian Zhang and Shanshe Wang and Siwei Ma and Wen Gao},
    year={2020},
    eprint={2005.09964},
    archivePrefix={arXiv},
    primaryClass={eess.IV}
}
```