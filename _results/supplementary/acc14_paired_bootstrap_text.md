# Top-14 Accuracy Paired Bootstrap Text Model Comparison

## GMS8K Istances

| Model A | Model B | Difference (A - B) | 95% CI | p |
|---|---|---:|---:|---:|
| Falcon3-1B-Instruct | Falcon3-3B-Instruct | -0.021 | [-0.025, -0.018] | 0.0002 |
| Falcon3-1B-Instruct | Falcon3-7B-Instruct | -0.026 | [-0.029, -0.023] | 0.0002 |
| Falcon3-1B-Instruct | Falcon3-10B-Instruct | -0.032 | [-0.035, -0.028] | 0.0002 |
| Falcon3-3B-Instruct | Falcon3-7B-Instruct | -0.004 | [-0.006, -0.002] | 0.0002 |
| Falcon3-3B-Instruct | Falcon3-10B-Instruct | -0.010 | [-0.013, -0.008] | 0.0002 |
| Falcon3-7B-Instruct | Falcon3-10B-Instruct | -0.006 | [-0.008, -0.004] | 0.0002 |

| Model A | Model B | Difference (A - B) | 95% CI | p |
|---|---|---:|---:|---:|
| Falcon3-Mamba-7B-Instruct | falcon-7b-instruct | 0.036 | [0.033, 0.040] | 0.0002 |

| Model A | Model B | Difference (A - B) | 95% CI | p |
|---|---|---:|---:|---:|
| Falcon-E-1B-Instruct | Falcon-E-3B-Instruct | -0.008 | [-0.010, -0.006] | 0.0002 |
| Falcon-E-1B-Instruct | Falcon-H1-Tiny-90M-Instruct | 0.045 | [0.040, 0.049] | 0.0002 |
| Falcon-E-3B-Instruct | Falcon-H1-Tiny-90M-Instruct | 0.052 | [0.048, 0.057] | 0.0002 |

| Model A | Model B | Difference (A - B) | 95% CI | p |
|---|---|---:|---:|---:|
| Codestral-22B-v0.1 | Mistral-7B-Instruct-v0.3 | 0.010 | [0.009, 0.012] | 0.0002 |

| Model A | Model B | Difference (A - B) | 95% CI | p |
|---|---|---:|---:|---:|
| Mistral-7B-Instruct-v0.1 | Mistral-7B-Instruct-v0.2 | 0.013 | [0.010, 0.016] | 0.0002 |

| Model A | Model B | Difference (A - B) | 95% CI | p |
|---|---|---:|---:|---:|
| Phi-3-medium-128k-instruct | Phi-3-mini-128k-instruct | 0.003 | [0.002, 0.005] | 0.0002 |
| Phi-3-medium-128k-instruct | Phi-3.5-mini-instruct | 0.005 | [0.003, 0.006] | 0.0002 |
| Phi-3-medium-128k-instruct | Phi-3.5-MoE-instruct | 0.001 | [-0.000, 0.003] | 0.076 |
| Phi-3-mini-128k-instruct | Phi-3.5-mini-instruct | 0.001 | [-0.000, 0.002] | 0.0684 |
| Phi-3-mini-128k-instruct | Phi-3.5-MoE-instruct | -0.002 | [-0.004, -0.001] | 0.0126 |
| Phi-3.5-mini-instruct | Phi-3.5-MoE-instruct | -0.003 | [-0.005, -0.002] | 0.0002 |

| Model A | Model B | Difference (A - B) | 95% CI | p |
|---|---|---:|---:|---:|
| Qwen2-0.5B-Instruct | Qwen2-1.5B-Instruct | -0.032 | [-0.037, -0.029] | 0.0002 |
| Qwen2-0.5B-Instruct | Qwen2-7B-Instruct | -0.052 | [-0.057, -0.048] | 0.0002 |
| Qwen2-0.5B-Instruct | Qwen2.5-0.5B-Instruct | -0.030 | [-0.034, -0.026] | 0.0002 |
| Qwen2-0.5B-Instruct | Qwen2.5-1.5B-Instruct | -0.046 | [-0.051, -0.042] | 0.0002 |
| Qwen2-0.5B-Instruct | Qwen2.5-3B-Instruct | -0.054 | [-0.059, -0.049] | 0.0002 |
| Qwen2-0.5B-Instruct | Qwen2.5-7B-Instruct | -0.051 | [-0.056, -0.046] | 0.0002 |
| Qwen2-0.5B-Instruct | Qwen2.5-14B-Instruct | -0.061 | [-0.067, -0.056] | 0.0002 |
| Qwen2-0.5B-Instruct | Qwen2.5-32B-Instruct | -0.063 | [-0.069, -0.058] | 0.0002 |
| Qwen2-0.5B-Instruct | Qwen3-4B-Instruct-2507 | -0.047 | [-0.052, -0.042] | 0.0002 |
| Qwen2-0.5B-Instruct | Qwen3-30B-A3B-Instruct-2507 | -0.057 | [-0.062, -0.052] | 0.0002 |
| Qwen2-1.5B-Instruct | Qwen2-7B-Instruct | -0.020 | [-0.023, -0.016] | 0.0002 |
| Qwen2-1.5B-Instruct | Qwen2.5-0.5B-Instruct | 0.002 | [-0.001, 0.006] | 0.221 |
| Qwen2-1.5B-Instruct | Qwen2.5-1.5B-Instruct | -0.014 | [-0.018, -0.010] | 0.0002 |
| Qwen2-1.5B-Instruct | Qwen2.5-3B-Instruct | -0.022 | [-0.025, -0.018] | 0.0002 |
| Qwen2-1.5B-Instruct | Qwen2.5-7B-Instruct | -0.019 | [-0.022, -0.015] | 0.0002 |
| Qwen2-1.5B-Instruct | Qwen2.5-14B-Instruct | -0.029 | [-0.033, -0.025] | 0.0002 |
| Qwen2-1.5B-Instruct | Qwen2.5-32B-Instruct | -0.031 | [-0.035, -0.027] | 0.0002 |
| Qwen2-1.5B-Instruct | Qwen3-4B-Instruct-2507 | -0.014 | [-0.018, -0.010] | 0.0002 |
| Qwen2-1.5B-Instruct | Qwen3-30B-A3B-Instruct-2507 | -0.024 | [-0.028, -0.021] | 0.0002 |
| Qwen2-7B-Instruct | Qwen2.5-0.5B-Instruct | 0.022 | [0.018, 0.026] | 0.0002 |
| Qwen2-7B-Instruct | Qwen2.5-1.5B-Instruct | 0.006 | [0.003, 0.008] | 0.0002 |
| Qwen2-7B-Instruct | Qwen2.5-3B-Instruct | -0.002 | [-0.005, 0.001] | 0.148 |
| Qwen2-7B-Instruct | Qwen2.5-7B-Instruct | 0.001 | [-0.001, 0.004] | 0.432 |
| Qwen2-7B-Instruct | Qwen2.5-14B-Instruct | -0.009 | [-0.012, -0.006] | 0.0002 |
| Qwen2-7B-Instruct | Qwen2.5-32B-Instruct | -0.011 | [-0.014, -0.009] | 0.0002 |
| Qwen2-7B-Instruct | Qwen3-4B-Instruct-2507 | 0.005 | [0.003, 0.008] | 0.0004 |
| Qwen2-7B-Instruct | Qwen3-30B-A3B-Instruct-2507 | -0.005 | [-0.008, -0.002] | 0.0022 |
| Qwen2.5-0.5B-Instruct | Qwen2.5-1.5B-Instruct | -0.016 | [-0.020, -0.013] | 0.0002 |
| Qwen2.5-0.5B-Instruct | Qwen2.5-3B-Instruct | -0.024 | [-0.028, -0.020] | 0.0002 |
| Qwen2.5-0.5B-Instruct | Qwen2.5-7B-Instruct | -0.021 | [-0.025, -0.017] | 0.0002 |
| Qwen2.5-0.5B-Instruct | Qwen2.5-14B-Instruct | -0.031 | [-0.035, -0.027] | 0.0002 |
| Qwen2.5-0.5B-Instruct | Qwen2.5-32B-Instruct | -0.033 | [-0.038, -0.029] | 0.0002 |
| Qwen2.5-0.5B-Instruct | Qwen3-4B-Instruct-2507 | -0.017 | [-0.021, -0.013] | 0.0002 |
| Qwen2.5-0.5B-Instruct | Qwen3-30B-A3B-Instruct-2507 | -0.027 | [-0.031, -0.023] | 0.0002 |
| Qwen2.5-1.5B-Instruct | Qwen2.5-3B-Instruct | -0.008 | [-0.011, -0.005] | 0.0002 |
| Qwen2.5-1.5B-Instruct | Qwen2.5-7B-Instruct | -0.005 | [-0.008, -0.002] | 0.0022 |
| Qwen2.5-1.5B-Instruct | Qwen2.5-14B-Instruct | -0.015 | [-0.018, -0.012] | 0.0002 |
| Qwen2.5-1.5B-Instruct | Qwen2.5-32B-Instruct | -0.017 | [-0.021, -0.014] | 0.0002 |
| Qwen2.5-1.5B-Instruct | Qwen3-4B-Instruct-2507 | -0.000 | [-0.004, 0.003] | 0.834 |
| Qwen2.5-1.5B-Instruct | Qwen3-30B-A3B-Instruct-2507 | -0.011 | [-0.014, -0.008] | 0.0002 |
| Qwen2.5-3B-Instruct | Qwen2.5-7B-Instruct | 0.003 | [0.001, 0.005] | 0.012 |
| Qwen2.5-3B-Instruct | Qwen2.5-14B-Instruct | -0.007 | [-0.010, -0.005] | 0.0002 |
| Qwen2.5-3B-Instruct | Qwen2.5-32B-Instruct | -0.009 | [-0.012, -0.007] | 0.0002 |
| Qwen2.5-3B-Instruct | Qwen3-4B-Instruct-2507 | 0.007 | [0.005, 0.010] | 0.0002 |
| Qwen2.5-3B-Instruct | Qwen3-30B-A3B-Instruct-2507 | -0.003 | [-0.005, -0.000] | 0.0308 |
| Qwen2.5-7B-Instruct | Qwen2.5-14B-Instruct | -0.010 | [-0.013, -0.008] | 0.0002 |
| Qwen2.5-7B-Instruct | Qwen2.5-32B-Instruct | -0.012 | [-0.015, -0.010] | 0.0002 |
| Qwen2.5-7B-Instruct | Qwen3-4B-Instruct-2507 | 0.004 | [0.002, 0.007] | 0.0032 |
| Qwen2.5-7B-Instruct | Qwen3-30B-A3B-Instruct-2507 | -0.006 | [-0.008, -0.003] | 0.0002 |
| Qwen2.5-14B-Instruct | Qwen2.5-32B-Instruct | -0.002 | [-0.004, -0.000] | 0.0504 |
| Qwen2.5-14B-Instruct | Qwen3-4B-Instruct-2507 | 0.015 | [0.012, 0.017] | 0.0002 |
| Qwen2.5-14B-Instruct | Qwen3-30B-A3B-Instruct-2507 | 0.004 | [0.002, 0.007] | 0.001 |
| Qwen2.5-32B-Instruct | Qwen3-4B-Instruct-2507 | 0.017 | [0.014, 0.020] | 0.0002 |
| Qwen2.5-32B-Instruct | Qwen3-30B-A3B-Instruct-2507 | 0.007 | [0.004, 0.009] | 0.0002 |
| Qwen3-4B-Instruct-2507 | Qwen3-30B-A3B-Instruct-2507 | -0.010 | [-0.013, -0.007] | 0.0002 |

## GMS8K Ablation Istances

| Model A | Model B | Difference (A - B) | 95% CI | p |
|---|---|---:|---:|---:|
| Falcon3-1B-Instruct | Falcon3-3B-Instruct | -0.024 | [-0.028, -0.021] | 0.0002 |
| Falcon3-1B-Instruct | Falcon3-7B-Instruct | -0.029 | [-0.033, -0.026] | 0.0002 |
| Falcon3-1B-Instruct | Falcon3-10B-Instruct | -0.035 | [-0.039, -0.031] | 0.0002 |
| Falcon3-3B-Instruct | Falcon3-7B-Instruct | -0.005 | [-0.008, -0.002] | 0.0006 |
| Falcon3-3B-Instruct | Falcon3-10B-Instruct | -0.011 | [-0.014, -0.009] | 0.0002 |
| Falcon3-7B-Instruct | Falcon3-10B-Instruct | -0.006 | [-0.008, -0.004] | 0.0002 |

| Model A | Model B | Difference (A - B) | 95% CI | p |
|---|---|---:|---:|---:|
| Falcon3-Mamba-7B-Instruct | falcon-7b-instruct | 0.036 | [0.032, 0.040] | 0.0002 |

| Model A | Model B | Difference (A - B) | 95% CI | p |
|---|---|---:|---:|---:|
| Falcon-E-1B-Instruct | Falcon-E-3B-Instruct | -0.008 | [-0.010, -0.006] | 0.0002 |
| Falcon-E-1B-Instruct | Falcon-H1-Tiny-90M-Instruct | 0.048 | [0.044, 0.052] | 0.0002 |
| Falcon-E-3B-Instruct | Falcon-H1-Tiny-90M-Instruct | 0.056 | [0.052, 0.060] | 0.0002 |

| Model A | Model B | Difference (A - B) | 95% CI | p |
|---|---|---:|---:|---:|
| Codestral-22B-v0.1 | Mistral-7B-Instruct-v0.3 | 0.015 | [0.013, 0.018] | 0.0002 |

| Model A | Model B | Difference (A - B) | 95% CI | p |
|---|---|---:|---:|---:|
| Mistral-7B-Instruct-v0.1 | Mistral-7B-Instruct-v0.2 | 0.015 | [0.012, 0.018] | 0.0002 |

| Model A | Model B | Difference (A - B) | 95% CI | p |
|---|---|---:|---:|---:|
| Phi-3-medium-128k-instruct | Phi-3-mini-128k-instruct | 0.008 | [0.006, 0.010] | 0.0002 |
| Phi-3-medium-128k-instruct | Phi-3.5-mini-instruct | 0.008 | [0.006, 0.011] | 0.0002 |
| Phi-3-medium-128k-instruct | Phi-3.5-MoE-instruct | 0.005 | [0.004, 0.007] | 0.0002 |
| Phi-3-mini-128k-instruct | Phi-3.5-mini-instruct | 0.000 | [-0.001, 0.002] | 0.663 |
| Phi-3-mini-128k-instruct | Phi-3.5-MoE-instruct | -0.003 | [-0.004, -0.001] | 0.0072 |
| Phi-3.5-mini-instruct | Phi-3.5-MoE-instruct | -0.003 | [-0.005, -0.001] | 0.0062 |

| Model A | Model B | Difference (A - B) | 95% CI | p |
|---|---|---:|---:|---:|
| Qwen2-0.5B-Instruct | Qwen2-1.5B-Instruct | -0.039 | [-0.044, -0.035] | 0.0002 |
| Qwen2-0.5B-Instruct | Qwen2-7B-Instruct | -0.058 | [-0.063, -0.053] | 0.0002 |
| Qwen2-0.5B-Instruct | Qwen2.5-0.5B-Instruct | -0.033 | [-0.037, -0.029] | 0.0002 |
| Qwen2-0.5B-Instruct | Qwen2.5-1.5B-Instruct | -0.049 | [-0.053, -0.044] | 0.0002 |
| Qwen2-0.5B-Instruct | Qwen2.5-3B-Instruct | -0.057 | [-0.062, -0.053] | 0.0002 |
| Qwen2-0.5B-Instruct | Qwen2.5-7B-Instruct | -0.060 | [-0.066, -0.055] | 0.0002 |
| Qwen2-0.5B-Instruct | Qwen2.5-14B-Instruct | -0.066 | [-0.072, -0.061] | 0.0002 |
| Qwen2-0.5B-Instruct | Qwen2.5-32B-Instruct | -0.074 | [-0.080, -0.068] | 0.0002 |
| Qwen2-0.5B-Instruct | Qwen3-4B-Instruct-2507 | -0.060 | [-0.066, -0.055] | 0.0002 |
| Qwen2-0.5B-Instruct | Qwen3-30B-A3B-Instruct-2507 | -0.067 | [-0.072, -0.062] | 0.0002 |
| Qwen2-1.5B-Instruct | Qwen2-7B-Instruct | -0.018 | [-0.022, -0.015] | 0.0002 |
| Qwen2-1.5B-Instruct | Qwen2.5-0.5B-Instruct | 0.006 | [0.003, 0.010] | 0.0012 |
| Qwen2-1.5B-Instruct | Qwen2.5-1.5B-Instruct | -0.009 | [-0.013, -0.006] | 0.0002 |
| Qwen2-1.5B-Instruct | Qwen2.5-3B-Instruct | -0.018 | [-0.022, -0.015] | 0.0002 |
| Qwen2-1.5B-Instruct | Qwen2.5-7B-Instruct | -0.021 | [-0.025, -0.017] | 0.0002 |
| Qwen2-1.5B-Instruct | Qwen2.5-14B-Instruct | -0.027 | [-0.031, -0.023] | 0.0002 |
| Qwen2-1.5B-Instruct | Qwen2.5-32B-Instruct | -0.034 | [-0.038, -0.030] | 0.0002 |
| Qwen2-1.5B-Instruct | Qwen3-4B-Instruct-2507 | -0.021 | [-0.024, -0.017] | 0.0002 |
| Qwen2-1.5B-Instruct | Qwen3-30B-A3B-Instruct-2507 | -0.028 | [-0.032, -0.024] | 0.0002 |
| Qwen2-7B-Instruct | Qwen2.5-0.5B-Instruct | 0.024 | [0.020, 0.029] | 0.0002 |
| Qwen2-7B-Instruct | Qwen2.5-1.5B-Instruct | 0.009 | [0.005, 0.012] | 0.0002 |
| Qwen2-7B-Instruct | Qwen2.5-3B-Instruct | 0.000 | [-0.003, 0.003] | 0.91 |
| Qwen2-7B-Instruct | Qwen2.5-7B-Instruct | -0.003 | [-0.006, 0.001] | 0.149 |
| Qwen2-7B-Instruct | Qwen2.5-14B-Instruct | -0.009 | [-0.012, -0.005] | 0.0002 |
| Qwen2-7B-Instruct | Qwen2.5-32B-Instruct | -0.016 | [-0.020, -0.013] | 0.0002 |
| Qwen2-7B-Instruct | Qwen3-4B-Instruct-2507 | -0.002 | [-0.006, 0.001] | 0.167 |
| Qwen2-7B-Instruct | Qwen3-30B-A3B-Instruct-2507 | -0.009 | [-0.013, -0.006] | 0.0002 |
| Qwen2.5-0.5B-Instruct | Qwen2.5-1.5B-Instruct | -0.016 | [-0.019, -0.012] | 0.0002 |
| Qwen2.5-0.5B-Instruct | Qwen2.5-3B-Instruct | -0.024 | [-0.028, -0.020] | 0.0002 |
| Qwen2.5-0.5B-Instruct | Qwen2.5-7B-Instruct | -0.027 | [-0.031, -0.023] | 0.0002 |
| Qwen2.5-0.5B-Instruct | Qwen2.5-14B-Instruct | -0.033 | [-0.038, -0.029] | 0.0002 |
| Qwen2.5-0.5B-Instruct | Qwen2.5-32B-Instruct | -0.041 | [-0.045, -0.037] | 0.0002 |
| Qwen2.5-0.5B-Instruct | Qwen3-4B-Instruct-2507 | -0.027 | [-0.031, -0.023] | 0.0002 |
| Qwen2.5-0.5B-Instruct | Qwen3-30B-A3B-Instruct-2507 | -0.034 | [-0.038, -0.030] | 0.0002 |
| Qwen2.5-1.5B-Instruct | Qwen2.5-3B-Instruct | -0.009 | [-0.011, -0.006] | 0.0002 |
| Qwen2.5-1.5B-Instruct | Qwen2.5-7B-Instruct | -0.011 | [-0.014, -0.008] | 0.0002 |
| Qwen2.5-1.5B-Instruct | Qwen2.5-14B-Instruct | -0.017 | [-0.021, -0.014] | 0.0002 |
| Qwen2.5-1.5B-Instruct | Qwen2.5-32B-Instruct | -0.025 | [-0.028, -0.022] | 0.0002 |
| Qwen2.5-1.5B-Instruct | Qwen3-4B-Instruct-2507 | -0.011 | [-0.015, -0.008] | 0.0002 |
| Qwen2.5-1.5B-Instruct | Qwen3-30B-A3B-Instruct-2507 | -0.018 | [-0.021, -0.015] | 0.0002 |
| Qwen2.5-3B-Instruct | Qwen2.5-7B-Instruct | -0.003 | [-0.006, 0.000] | 0.0878 |
| Qwen2.5-3B-Instruct | Qwen2.5-14B-Instruct | -0.009 | [-0.012, -0.006] | 0.0002 |
| Qwen2.5-3B-Instruct | Qwen2.5-32B-Instruct | -0.016 | [-0.019, -0.013] | 0.0002 |
| Qwen2.5-3B-Instruct | Qwen3-4B-Instruct-2507 | -0.003 | [-0.006, 0.001] | 0.171 |
| Qwen2.5-3B-Instruct | Qwen3-30B-A3B-Instruct-2507 | -0.010 | [-0.013, -0.006] | 0.0002 |
| Qwen2.5-7B-Instruct | Qwen2.5-14B-Instruct | -0.006 | [-0.010, -0.003] | 0.0004 |
| Qwen2.5-7B-Instruct | Qwen2.5-32B-Instruct | -0.014 | [-0.017, -0.011] | 0.0002 |
| Qwen2.5-7B-Instruct | Qwen3-4B-Instruct-2507 | 0.000 | [-0.003, 0.004] | 0.917 |
| Qwen2.5-7B-Instruct | Qwen3-30B-A3B-Instruct-2507 | -0.007 | [-0.010, -0.004] | 0.0004 |
| Qwen2.5-14B-Instruct | Qwen2.5-32B-Instruct | -0.007 | [-0.010, -0.005] | 0.0002 |
| Qwen2.5-14B-Instruct | Qwen3-4B-Instruct-2507 | 0.006 | [0.003, 0.010] | 0.0004 |
| Qwen2.5-14B-Instruct | Qwen3-30B-A3B-Instruct-2507 | -0.001 | [-0.004, 0.003] | 0.745 |
| Qwen2.5-32B-Instruct | Qwen3-4B-Instruct-2507 | 0.014 | [0.011, 0.017] | 0.0002 |
| Qwen2.5-32B-Instruct | Qwen3-30B-A3B-Instruct-2507 | 0.007 | [0.004, 0.009] | 0.0002 |
| Qwen3-4B-Instruct-2507 | Qwen3-30B-A3B-Instruct-2507 | -0.007 | [-0.010, -0.004] | 0.0002 |

