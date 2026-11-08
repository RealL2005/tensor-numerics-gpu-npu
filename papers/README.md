# 参考文献 PDF

选题说明中列出的 5 篇参考论文，均已下载 arXiv 预印本（开放获取）。正式发表版可能存在排版与数据差异，
引用时应以出版版本为准。

重新抓取：`bash papers/fetch_papers.sh`（或 `--force` 覆盖）。

| # | 文件 | 页 | arXiv | 出处 |
| --- | --- | --- | --- | --- |
| 1 | `01_mind-the-gap_heterogeneous-accelerators_inconsistencies.pdf` | 8 | [2511.11601](https://arxiv.org/abs/2511.11601) | MSN 2025, DOI [10.1109/MSN69125.2025.00054](https://doi.org/10.1109/MSN69125.2025.00054) |
| 2 | `02_taming-bitwise-behavior_gpu-kernels-tensor-core.pdf` | 18 | [2609.11356](https://arxiv.org/abs/2609.11356) | 预印本 |
| 3 | `03_kernel-contracts_ml-kernel-correctness-spec-language.pdf` | 28 | [2604.22032](https://arxiv.org/abs/2604.22032) | 预印本 |
| 4 | `04_testing-gpu-numerics_nvidia-vs-amd.pdf` | 11 | [2410.09172](https://arxiv.org/abs/2410.09172) | SC-W '24, DOI [10.1109/SCW63240.2024.00077](https://doi.org/10.1109/SCW63240.2024.00077) |
| 5 | `05_floating-point-non-associativity_reproducibility.pdf` | 10 | [2408.05148](https://arxiv.org/abs/2408.05148) | SC-W '24, DOI [10.1109/SCW63240.2024.00028](https://doi.org/10.1109/SCW63240.2024.00028) |

---

## BibTeX

```bibtex
@inproceedings{wen2025mindgap,
  title     = {Mind the Gap: Revealing Inconsistencies Across Heterogeneous {AI} Accelerators},
  author    = {Wen, Elliott and Ma, Sean and Tempero, Ewan and Dietrich, Jens and Luo, Daniel and
               Shen, Jiaxing and Zhao, Kaiqi and Sham, Bruce and Song, Yousong and Hua, Jiayi and Hong, Jia},
  booktitle = {2025 International Conference on Mobility, Sensing and Networking (MSN)},
  year      = {2025},
  doi       = {10.1109/MSN69125.2025.00054},
  note      = {arXiv:2511.11601}
}

@misc{yang2026taming,
  title  = {Taming Bitwise Behavior in {GPU} Kernels with Tensor Core: Black-Box Reconstruction,
            Compiler Enforcement, and Static Verification},
  author = {Yang, Ziteng and Riasanovsky, Nicholas J. and Deng, Warren and Sarkar, Vivek},
  year   = {2026},
  note   = {arXiv:2609.11356}
}

@misc{veit2026kernelcontracts,
  title  = {Kernel Contracts: A Specification Language for {ML} Kernel Correctness Across
            Heterogeneous Silicon},
  author = {Veit, Cooper},
  year   = {2026},
  note   = {arXiv:2604.22032}
}

@inproceedings{zahid2024testinggpunumerics,
  title     = {Testing {GPU} Numerics: Finding Numerical Differences Between {NVIDIA} and {AMD} {GPUs}},
  author    = {Zahid, Anwar Hossain and Laguna, Ignacio and Le, Wei},
  booktitle = {SC24 Workshops (SC-W '24)},
  year      = {2024},
  doi       = {10.1109/SCW63240.2024.00077},
  note      = {arXiv:2410.09172}
}

@inproceedings{shanmugavelu2024fpnareproducibility,
  title     = {Impacts of Floating-Point Non-Associativity on Reproducibility for {HPC} and
               Deep Learning Applications},
  author    = {Shanmugavelu, Sanjif and Taillefumier, Mathieu and Culver, Christopher and
               Hernandez, Oscar and Coletti, Mark and Sedova, Ada},
  booktitle = {SC24 Workshops (SC-W '24)},
  year      = {2024},
  doi       = {10.1109/SCW63240.2024.00028},
  note      = {arXiv:2408.05148}
}
```

## 版权说明

本目录仅存放 arXiv 开放获取版本，用于个人学习与研究引用。各论文版权归原作者所有，
请勿再分发；如需正式版请通过出版方获取。
