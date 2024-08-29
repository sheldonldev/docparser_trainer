# 半精度介绍

(s: sign, e: exponent, m: mantissa)

- fp32: 32位浮点数，单精度。1s + 8e + 23m, range: [1e-38, 3e38]
- fp16: 16位浮点数，半精度。1s + 5e + 10m, range: [5.96^-8, 65504]
  - 溢出问题
  - 舍入问题
- bf16: 16位浮点数，半精度。1s + 8e + 7m, range: [1e-38, 3e38]
