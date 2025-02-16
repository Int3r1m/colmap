#include "colmap/math/math.h"

namespace colmap {

uint64_t NChooseK(uint64_t n, uint64_t k) {
  if (n == 0 || n < k) {
    return 0;
  }

  uint64_t r = 1;
  for (uint64_t d = 1; d <= k; ++d) {
    r *= n--;
    r /= d;
  }
  return r;
}

}  // namespace colmap
