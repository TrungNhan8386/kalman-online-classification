# Kalman Filter for Online Classification of Non-Stationary Data — Reproduction

Term project môn **Introduction to Machine Learning (HCMUS)**: đọc hiểu, tái lập thí nghiệm, phê bình lý thuyết và chứng minh điểm yếu bằng thực nghiệm cho paper:

> M. K. Titsias, A. Galashov, A. Rannen-Triki, R. Pascanu, Y. W. Teh, J. Bornschein, "Kalman Filter for Online Classification of Non-Stationary Data," in *Proc. International Conference on Learning Representations (ICLR)*, 2024. [Poster](https://iclr.cc/virtual/2024/poster/18380) · [arXiv:2306.08448](https://arxiv.org/abs/2306.08448)

- **Nhóm:** Team 4 — Trần Phước Tài, Nguyễn Đặng Hoàng Khang, Nguyễn Trần Trung Nhân
- **W&B project (public):** https://wandb.ai/nhan-ng-hcmus/kalman-online-classification
- **Scope reproduction:** Core scope (lý do: xem mục [Scope](#scope))

## Cấu trúc repo

```
.
├── configs/            # cấu hình YAML cho từng thí nghiệm
├── notebooks/          # notebook phân tích, vẽ hình
├── scripts/            # entry point chạy thí nghiệm
│   ├── run_synthetic.py
│   ├── extract_features.py
│   └── run_cifar100.py
├── src/kfoc/           # package chính
│   ├── data.py         # sinh dữ liệu synthetic, tải CIFAR-100 từ Hugging Face
│   ├── models.py       # baseline online SGD + Kalman filter (phần đóng góp cốt lõi)
│   ├── metrics.py      # online accuracy, log-loss tích lũy
│   ├── stream.py       # vòng lặp đánh giá prequential (test-then-train)
│   ├── seed.py         # cố định random seed
│   └── tracking.py     # khởi tạo W&B run với tag
└── tests/              # smoke test
```

## Setup

Yêu cầu Python ≥ 3.9.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
wandb login
```

## Chạy thí nghiệm

```bash
# Synthetic stream với trọng số tuyến tính bị drift
python scripts/run_synthetic.py --config configs/synthetic.yaml

# CIFAR-100 non-stationary: trích feature một lần, sau đó chạy online learner trên feature
python scripts/extract_features.py --config configs/cifar100.yaml
python scripts/run_cifar100.py --config configs/cifar100.yaml
```

Thêm `--no-wandb` để chạy thử mà không log lên W&B.

## Scope

**Core scope.** Thí nghiệm CLOC trong paper cần hạ tầng vượt khả năng của nhóm, nên được thay bằng phân tích bổ sung do nhóm tự thiết kế. Chi tiết lý do trình bày trong deck.

| Thí nghiệm | Trong paper | Nhóm thực hiện |
|---|---|---|
| Synthetic non-stationary stream | Có | Có |
| CIFAR-100 non-stationary (feature cố định) | Có | Có |
| CLOC | Có | Không — thay bằng sensitivity sweep |

## Bảng ánh xạ thí nghiệm → code → W&B

| Thí nghiệm | Script / notebook | W&B run | Run cho thấy gì |
|---|---|---|---|
| Synthetic — baseline online SGD | `scripts/run_synthetic.py` | `<link>` | `<1 dòng>` |
| Synthetic — Kalman filter | `scripts/run_synthetic.py` | `<link>` | `<1 dòng>` |
| CIFAR-100 — Kalman filter | `scripts/run_cifar100.py` | `<link>` | `<1 dòng>` |
| Phân tích bổ sung | `<...>` | `<link>` | `<1 dòng>` |

Tag W&B: `reproduction` cho chặng tái lập, `empirics` cho chặng chứng minh điểm yếu.

## Tái lập được (reproducibility)

- Mọi script nhận `seed` từ config; `kfoc.seed.set_seed` cố định seed cho `random`, `numpy`, `torch`.
- Dataset tải programmatic từ Hugging Face (`uoft-cs/cifar100`) trong code, không commit dữ liệu vào repo.
- Cấu hình của mỗi run được log đầy đủ lên W&B.

## Tái sử dụng thư viện và trích dẫn

| Thành phần | Dùng cho | Nguồn |
|---|---|---|
| PyTorch, torchvision | backbone pretrained để trích feature | [1] |
| Hugging Face `datasets` | tải CIFAR-100 | [2] |
| Weights & Biases | theo dõi thí nghiệm | [3] |

Phần đóng góp cốt lõi của paper (Kalman filter trên trọng số tuyến tính, học hệ số forgetting) do nhóm tự cài đặt từ mô tả trong paper, không sao chép code gốc.

[1] A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," in *Proc. NeurIPS*, 2019.
[2] Q. Lhoest et al., "Datasets: A community library for natural language processing," in *Proc. EMNLP: System Demonstrations*, 2021.
[3] L. Biewald, "Experiment tracking with Weights and Biases," 2020. [Online]. Available: https://wandb.ai
