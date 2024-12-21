#pragma once
#include <torch/torch.h>

#include <array>

class InputPadder {
public:
    InputPadder(int h, int w, int div_by = 8)
        : ht_(h),
          wd_(w) {
        int pad_ht = (((ht_ / div_by) + 1) * div_by - ht_) % div_by;
        int pad_wd = (((wd_ / div_by) + 1) * div_by - wd_) % div_by;

        pad_ = {pad_wd / 2, pad_wd - pad_wd / 2,
                pad_ht / 2, pad_ht - pad_ht / 2};
    }

    // Move semantics for pad operation
    torch::Tensor pad(torch::Tensor x) &&;
    torch::Tensor pad(const torch::Tensor& x) &;

    // Move semantics for unpad operation
    [[maybe_unused]] torch::Tensor unpad(torch::Tensor x) &&;
    torch::Tensor unpad(const torch::Tensor& x) &;

    void setPadding(const std::array<int, 4>& pad);
    const std::array<int, 4>& getPadding() const { return pad_; }

private:
    int ht_;
    int wd_;
    std::array<int, 4> pad_;
};