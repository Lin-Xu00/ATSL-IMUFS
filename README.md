## Adaptive Topological Similarity Learning for Incomplete Multi-view Unsupervised Feature Selection (ATSL-IMUFS)

This repository contains the MATLAB implementation of the algorithm proposed in the paper **“Adaptive Topological Similarity Learning for Incomplete Multi-view Unsupervised Feature Selection (ATSL-IMUFS)”**.

All experiments were conducted using MATLAB R2022b on a desktop equipped with an Intel Core i9-10900 CPU (2.80 GHz) and 64 GB of RAM.

------

## Implementation

The MATLAB implementation of ATSL-IMUFS is provided in the `ATSL-IMUFS` directory. The main components include:

- **`ATSL_IMUFS.m`**: implements the core algorithm and includes the complete optimization procedure.
- **`ClusteringPerformance.m`**: performs K-means clustering and computes the ACC and NMI metrics.

------

## Demo

An example usage of the proposed algorithm is provided in **`Main.m`**.

------

## Citation

If you find our approach useful in your research, please cite the following reference:

```bibtex
@ARTICLE{xu2025adaptive,
  author={Xu, Lin and Li, Ke and Wang, Dongjie and Zhou, Fanyin and Lv, Fengmao and Li, Tianrui and Huang, Yanyong},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  title={Adaptive Topological Similarity Learning for Incomplete Multi-view Unsupervised Feature Selection},
  year={2025},
  doi={10.1109/TCSVT.2025.3642753}
}
