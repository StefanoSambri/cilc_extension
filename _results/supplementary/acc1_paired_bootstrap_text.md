# Top-1 Accuracy Paired Bootstrap Text Model Comparison

## GMS8K Istances

| Model A | Model B | Difference (A - B) | 95% CI | p |
|---|---|---:|---:|---:|
| Falcon3-1B-Instruct | Falcon3-3B-Instruct | -0.044 | [-0.052, -0.036] | 0.0002 |
| Falcon3-1B-Instruct | Falcon3-7B-Instruct | -0.065 | [-0.073, -0.059] | 0.0002 |
| Falcon3-1B-Instruct | Falcon3-10B-Instruct | -0.078 | [-0.086, -0.071] | 0.0002 |
| Falcon3-3B-Instruct | Falcon3-7B-Instruct | -0.022 | [-0.027, -0.017] | 0.0002 |
| Falcon3-3B-Instruct | Falcon3-10B-Instruct | -0.035 | [-0.039, -0.030] | 0.0002 |
| Falcon3-7B-Instruct | Falcon3-10B-Instruct | -0.013 | [-0.017, -0.009] | 0.0002 |

| Model A | Model B | Difference (A - B) | 95% CI | p |
|---|---|---:|---:|---:|
| Falcon3-Mamba-7B-Instruct | falcon-7b-instruct | 0.107 | [0.099, 0.115] | 0.0002 |

| Model A | Model B | Difference (A - B) | 95% CI | p |
|---|---|---:|---:|---:|
| Falcon-E-1B-Instruct | Falcon-E-3B-Instruct | -0.036 | [-0.041, -0.031] | 0.0002 |
| Falcon-E-1B-Instruct | Falcon-H1-Tiny-90M-Instruct | 0.129 | [0.123, 0.136] | 0.0002 |
| Falcon-E-3B-Instruct | Falcon-H1-Tiny-90M-Instruct | 0.165 | [0.156, 0.174] | 0.0002 |

| Model A | Model B | Difference (A - B) | 95% CI | p |
|---|---|---:|---:|---:|
| Codestral-22B-v0.1 | Mistral-7B-Instruct-v0.3 | 0.040 | [0.035, 0.044] | 0.0002 |

| Model A | Model B | Difference (A - B) | 95% CI | p |
|---|---|---:|---:|---:|
| Mistral-7B-Instruct-v0.1 | Mistral-7B-Instruct-v0.2 | 0.009 | [0.004, 0.013] | 0.0004 |

| Model A | Model B | Difference (A - B) | 95% CI | p |
|---|---|---:|---:|---:|
| Phi-3-medium-128k-instruct | Phi-3-mini-128k-instruct | 0.004 | [-0.000, 0.008] | 0.0602 |
| Phi-3-medium-128k-instruct | Phi-3.5-mini-instruct | 0.008 | [0.005, 0.012] | 0.0002 |
| Phi-3-medium-128k-instruct | Phi-3.5-MoE-instruct | -0.001 | [-0.004, 0.003] | 0.74 |
| Phi-3-mini-128k-instruct | Phi-3.5-mini-instruct | 0.005 | [0.002, 0.008] | 0.0022 |
| Phi-3-mini-128k-instruct | Phi-3.5-MoE-instruct | -0.004 | [-0.008, -0.000] | 0.0364 |
| Phi-3.5-mini-instruct | Phi-3.5-MoE-instruct | -0.009 | [-0.013, -0.005] | 0.0002 |

| Model A | Model B | Difference (A - B) | 95% CI | p |
|---|---|---:|---:|---:|
| Qwen2-0.5B-Instruct | Qwen2-1.5B-Instruct | -0.041 | [-0.048, -0.033] | 0.0002 |
| Qwen2-0.5B-Instruct | Qwen2-7B-Instruct | -0.079 | [-0.089, -0.071] | 0.0002 |
| Qwen2-0.5B-Instruct | Qwen2.5-0.5B-Instruct | -0.030 | [-0.038, -0.023] | 0.0002 |
| Qwen2-0.5B-Instruct | Qwen2.5-1.5B-Instruct | -0.078 | [-0.087, -0.070] | 0.0002 |
| Qwen2-0.5B-Instruct | Qwen2.5-3B-Instruct | -0.089 | [-0.098, -0.080] | 0.0002 |
| Qwen2-0.5B-Instruct | Qwen2.5-7B-Instruct | -0.095 | [-0.105, -0.086] | 0.0002 |
| Qwen2-0.5B-Instruct | Qwen2.5-14B-Instruct | -0.085 | [-0.096, -0.075] | 0.0002 |
| Qwen2-0.5B-Instruct | Qwen2.5-32B-Instruct | -0.100 | [-0.110, -0.090] | 0.0002 |
| Qwen2-0.5B-Instruct | Qwen3-4B-Instruct-2507 | -0.080 | [-0.090, -0.071] | 0.0002 |
| Qwen2-0.5B-Instruct | Qwen3-30B-A3B-Instruct-2507 | -0.099 | [-0.110, -0.088] | 0.0002 |
| Qwen2-1.5B-Instruct | Qwen2-7B-Instruct | -0.039 | [-0.045, -0.032] | 0.0002 |
| Qwen2-1.5B-Instruct | Qwen2.5-0.5B-Instruct | 0.010 | [0.004, 0.016] | 0.0012 |
| Qwen2-1.5B-Instruct | Qwen2.5-1.5B-Instruct | -0.037 | [-0.044, -0.031] | 0.0002 |
| Qwen2-1.5B-Instruct | Qwen2.5-3B-Instruct | -0.048 | [-0.055, -0.042] | 0.0002 |
| Qwen2-1.5B-Instruct | Qwen2.5-7B-Instruct | -0.054 | [-0.062, -0.047] | 0.0002 |
| Qwen2-1.5B-Instruct | Qwen2.5-14B-Instruct | -0.044 | [-0.052, -0.037] | 0.0002 |
| Qwen2-1.5B-Instruct | Qwen2.5-32B-Instruct | -0.059 | [-0.067, -0.051] | 0.0002 |
| Qwen2-1.5B-Instruct | Qwen3-4B-Instruct-2507 | -0.040 | [-0.047, -0.032] | 0.0002 |
| Qwen2-1.5B-Instruct | Qwen3-30B-A3B-Instruct-2507 | -0.058 | [-0.066, -0.050] | 0.0002 |
| Qwen2-7B-Instruct | Qwen2.5-0.5B-Instruct | 0.049 | [0.041, 0.057] | 0.0002 |
| Qwen2-7B-Instruct | Qwen2.5-1.5B-Instruct | 0.001 | [-0.004, 0.007] | 0.602 |
| Qwen2-7B-Instruct | Qwen2.5-3B-Instruct | -0.010 | [-0.015, -0.004] | 0.0026 |
| Qwen2-7B-Instruct | Qwen2.5-7B-Instruct | -0.016 | [-0.021, -0.010] | 0.0002 |
| Qwen2-7B-Instruct | Qwen2.5-14B-Instruct | -0.006 | [-0.011, 0.000] | 0.0552 |
| Qwen2-7B-Instruct | Qwen2.5-32B-Instruct | -0.020 | [-0.026, -0.015] | 0.0002 |
| Qwen2-7B-Instruct | Qwen3-4B-Instruct-2507 | -0.001 | [-0.007, 0.005] | 0.838 |
| Qwen2-7B-Instruct | Qwen3-30B-A3B-Instruct-2507 | -0.019 | [-0.025, -0.013] | 0.0002 |
| Qwen2.5-0.5B-Instruct | Qwen2.5-1.5B-Instruct | -0.048 | [-0.055, -0.041] | 0.0002 |
| Qwen2.5-0.5B-Instruct | Qwen2.5-3B-Instruct | -0.059 | [-0.067, -0.051] | 0.0002 |
| Qwen2.5-0.5B-Instruct | Qwen2.5-7B-Instruct | -0.065 | [-0.073, -0.057] | 0.0002 |
| Qwen2.5-0.5B-Instruct | Qwen2.5-14B-Instruct | -0.055 | [-0.064, -0.046] | 0.0002 |
| Qwen2.5-0.5B-Instruct | Qwen2.5-32B-Instruct | -0.070 | [-0.079, -0.061] | 0.0002 |
| Qwen2.5-0.5B-Instruct | Qwen3-4B-Instruct-2507 | -0.050 | [-0.058, -0.042] | 0.0002 |
| Qwen2.5-0.5B-Instruct | Qwen3-30B-A3B-Instruct-2507 | -0.068 | [-0.078, -0.059] | 0.0002 |
| Qwen2.5-1.5B-Instruct | Qwen2.5-3B-Instruct | -0.011 | [-0.016, -0.006] | 0.0002 |
| Qwen2.5-1.5B-Instruct | Qwen2.5-7B-Instruct | -0.017 | [-0.023, -0.012] | 0.0002 |
| Qwen2.5-1.5B-Instruct | Qwen2.5-14B-Instruct | -0.007 | [-0.013, -0.001] | 0.0174 |
| Qwen2.5-1.5B-Instruct | Qwen2.5-32B-Instruct | -0.022 | [-0.028, -0.016] | 0.0002 |
| Qwen2.5-1.5B-Instruct | Qwen3-4B-Instruct-2507 | -0.002 | [-0.008, 0.004] | 0.477 |
| Qwen2.5-1.5B-Instruct | Qwen3-30B-A3B-Instruct-2507 | -0.021 | [-0.027, -0.014] | 0.0002 |
| Qwen2.5-3B-Instruct | Qwen2.5-7B-Instruct | -0.006 | [-0.012, 0.000] | 0.05 |
| Qwen2.5-3B-Instruct | Qwen2.5-14B-Instruct | 0.004 | [-0.002, 0.010] | 0.206 |
| Qwen2.5-3B-Instruct | Qwen2.5-32B-Instruct | -0.011 | [-0.018, -0.004] | 0.0018 |
| Qwen2.5-3B-Instruct | Qwen3-4B-Instruct-2507 | 0.009 | [0.003, 0.015] | 0.0034 |
| Qwen2.5-3B-Instruct | Qwen3-30B-A3B-Instruct-2507 | -0.010 | [-0.016, -0.003] | 0.0048 |
| Qwen2.5-7B-Instruct | Qwen2.5-14B-Instruct | 0.010 | [0.004, 0.016] | 0.0008 |
| Qwen2.5-7B-Instruct | Qwen2.5-32B-Instruct | -0.005 | [-0.010, 0.000] | 0.0818 |
| Qwen2.5-7B-Instruct | Qwen3-4B-Instruct-2507 | 0.015 | [0.009, 0.021] | 0.0002 |
| Qwen2.5-7B-Instruct | Qwen3-30B-A3B-Instruct-2507 | -0.003 | [-0.009, 0.002] | 0.234 |
| Qwen2.5-14B-Instruct | Qwen2.5-32B-Instruct | -0.015 | [-0.020, -0.010] | 0.0002 |
| Qwen2.5-14B-Instruct | Qwen3-4B-Instruct-2507 | 0.005 | [-0.001, 0.011] | 0.112 |
| Qwen2.5-14B-Instruct | Qwen3-30B-A3B-Instruct-2507 | -0.014 | [-0.019, -0.008] | 0.0002 |
| Qwen2.5-32B-Instruct | Qwen3-4B-Instruct-2507 | 0.020 | [0.014, 0.026] | 0.0002 |
| Qwen2.5-32B-Instruct | Qwen3-30B-A3B-Instruct-2507 | 0.001 | [-0.004, 0.006] | 0.616 |
| Qwen3-4B-Instruct-2507 | Qwen3-30B-A3B-Instruct-2507 | -0.018 | [-0.025, -0.012] | 0.0002 |

## GMS8K Ablation Istances

| Model A | Model B | Difference (A - B) | 95% CI | p |
|---|---|---:|---:|---:|
| Falcon3-1B-Instruct | Falcon3-3B-Instruct | -0.051 | [-0.059, -0.044] | 0.0002 |
| Falcon3-1B-Instruct | Falcon3-7B-Instruct | -0.078 | [-0.086, -0.071] | 0.0002 |
| Falcon3-1B-Instruct | Falcon3-10B-Instruct | -0.093 | [-0.102, -0.084] | 0.0002 |
| Falcon3-3B-Instruct | Falcon3-7B-Instruct | -0.027 | [-0.031, -0.022] | 0.0002 |
| Falcon3-3B-Instruct | Falcon3-10B-Instruct | -0.041 | [-0.047, -0.036] | 0.0002 |
| Falcon3-7B-Instruct | Falcon3-10B-Instruct | -0.015 | [-0.019, -0.010] | 0.0002 |

| Model A | Model B | Difference (A - B) | 95% CI | p |
|---|---|---:|---:|---:|
| Falcon3-Mamba-7B-Instruct | falcon-7b-instruct | 0.097 | [0.090, 0.104] | 0.0002 |

| Model A | Model B | Difference (A - B) | 95% CI | p |
|---|---|---:|---:|---:|
| Falcon-E-1B-Instruct | Falcon-E-3B-Instruct | -0.034 | [-0.039, -0.028] | 0.0002 |
| Falcon-E-1B-Instruct | Falcon-H1-Tiny-90M-Instruct | 0.120 | [0.114, 0.126] | 0.0002 |
| Falcon-E-3B-Instruct | Falcon-H1-Tiny-90M-Instruct | 0.154 | [0.146, 0.161] | 0.0002 |

| Model A | Model B | Difference (A - B) | 95% CI | p |
|---|---|---:|---:|---:|
| Codestral-22B-v0.1 | Mistral-7B-Instruct-v0.3 | 0.046 | [0.041, 0.052] | 0.0002 |

| Model A | Model B | Difference (A - B) | 95% CI | p |
|---|---|---:|---:|---:|
| Mistral-7B-Instruct-v0.1 | Mistral-7B-Instruct-v0.2 | 0.011 | [0.007, 0.016] | 0.0002 |

| Model A | Model B | Difference (A - B) | 95% CI | p |
|---|---|---:|---:|---:|
| Phi-3-medium-128k-instruct | Phi-3-mini-128k-instruct | 0.012 | [0.007, 0.017] | 0.0002 |
| Phi-3-medium-128k-instruct | Phi-3.5-mini-instruct | 0.018 | [0.013, 0.023] | 0.0002 |
| Phi-3-medium-128k-instruct | Phi-3.5-MoE-instruct | 0.007 | [0.003, 0.011] | 0.0014 |
| Phi-3-mini-128k-instruct | Phi-3.5-mini-instruct | 0.006 | [0.003, 0.009] | 0.001 |
| Phi-3-mini-128k-instruct | Phi-3.5-MoE-instruct | -0.005 | [-0.010, -0.001] | 0.0212 |
| Phi-3.5-mini-instruct | Phi-3.5-MoE-instruct | -0.011 | [-0.016, -0.007] | 0.0002 |

| Model A | Model B | Difference (A - B) | 95% CI | p |
|---|---|---:|---:|---:|
| Qwen2-0.5B-Instruct | Qwen2-1.5B-Instruct | -0.047 | [-0.054, -0.040] | 0.0002 |
| Qwen2-0.5B-Instruct | Qwen2-7B-Instruct | -0.096 | [-0.105, -0.088] | 0.0002 |
| Qwen2-0.5B-Instruct | Qwen2.5-0.5B-Instruct | -0.032 | [-0.038, -0.025] | 0.0002 |
| Qwen2-0.5B-Instruct | Qwen2.5-1.5B-Instruct | -0.085 | [-0.093, -0.076] | 0.0002 |
| Qwen2-0.5B-Instruct | Qwen2.5-3B-Instruct | -0.092 | [-0.101, -0.083] | 0.0002 |
| Qwen2-0.5B-Instruct | Qwen2.5-7B-Instruct | -0.112 | [-0.121, -0.103] | 0.0002 |
| Qwen2-0.5B-Instruct | Qwen2.5-14B-Instruct | -0.105 | [-0.114, -0.097] | 0.0002 |
| Qwen2-0.5B-Instruct | Qwen2.5-32B-Instruct | -0.133 | [-0.144, -0.124] | 0.0002 |
| Qwen2-0.5B-Instruct | Qwen3-4B-Instruct-2507 | -0.098 | [-0.107, -0.089] | 0.0002 |
| Qwen2-0.5B-Instruct | Qwen3-30B-A3B-Instruct-2507 | -0.123 | [-0.133, -0.113] | 0.0002 |
| Qwen2-1.5B-Instruct | Qwen2-7B-Instruct | -0.049 | [-0.056, -0.042] | 0.0002 |
| Qwen2-1.5B-Instruct | Qwen2.5-0.5B-Instruct | 0.015 | [0.009, 0.022] | 0.0002 |
| Qwen2-1.5B-Instruct | Qwen2.5-1.5B-Instruct | -0.038 | [-0.045, -0.032] | 0.0002 |
| Qwen2-1.5B-Instruct | Qwen2.5-3B-Instruct | -0.045 | [-0.052, -0.039] | 0.0002 |
| Qwen2-1.5B-Instruct | Qwen2.5-7B-Instruct | -0.065 | [-0.073, -0.057] | 0.0002 |
| Qwen2-1.5B-Instruct | Qwen2.5-14B-Instruct | -0.058 | [-0.067, -0.051] | 0.0002 |
| Qwen2-1.5B-Instruct | Qwen2.5-32B-Instruct | -0.087 | [-0.095, -0.078] | 0.0002 |
| Qwen2-1.5B-Instruct | Qwen3-4B-Instruct-2507 | -0.051 | [-0.058, -0.045] | 0.0002 |
| Qwen2-1.5B-Instruct | Qwen3-30B-A3B-Instruct-2507 | -0.076 | [-0.084, -0.068] | 0.0002 |
| Qwen2-7B-Instruct | Qwen2.5-0.5B-Instruct | 0.064 | [0.056, 0.073] | 0.0002 |
| Qwen2-7B-Instruct | Qwen2.5-1.5B-Instruct | 0.011 | [0.004, 0.018] | 0.0038 |
| Qwen2-7B-Instruct | Qwen2.5-3B-Instruct | 0.004 | [-0.002, 0.010] | 0.202 |
| Qwen2-7B-Instruct | Qwen2.5-7B-Instruct | -0.016 | [-0.023, -0.009] | 0.0002 |
| Qwen2-7B-Instruct | Qwen2.5-14B-Instruct | -0.009 | [-0.017, -0.002] | 0.0174 |
| Qwen2-7B-Instruct | Qwen2.5-32B-Instruct | -0.037 | [-0.044, -0.031] | 0.0002 |
| Qwen2-7B-Instruct | Qwen3-4B-Instruct-2507 | -0.002 | [-0.009, 0.005] | 0.507 |
| Qwen2-7B-Instruct | Qwen3-30B-A3B-Instruct-2507 | -0.027 | [-0.034, -0.020] | 0.0002 |
| Qwen2.5-0.5B-Instruct | Qwen2.5-1.5B-Instruct | -0.053 | [-0.060, -0.046] | 0.0002 |
| Qwen2.5-0.5B-Instruct | Qwen2.5-3B-Instruct | -0.060 | [-0.068, -0.052] | 0.0002 |
| Qwen2.5-0.5B-Instruct | Qwen2.5-7B-Instruct | -0.080 | [-0.090, -0.071] | 0.0002 |
| Qwen2.5-0.5B-Instruct | Qwen2.5-14B-Instruct | -0.073 | [-0.082, -0.066] | 0.0002 |
| Qwen2.5-0.5B-Instruct | Qwen2.5-32B-Instruct | -0.102 | [-0.111, -0.093] | 0.0002 |
| Qwen2.5-0.5B-Instruct | Qwen3-4B-Instruct-2507 | -0.067 | [-0.075, -0.058] | 0.0002 |
| Qwen2.5-0.5B-Instruct | Qwen3-30B-A3B-Instruct-2507 | -0.091 | [-0.101, -0.082] | 0.0002 |
| Qwen2.5-1.5B-Instruct | Qwen2.5-3B-Instruct | -0.007 | [-0.013, -0.001] | 0.027 |
| Qwen2.5-1.5B-Instruct | Qwen2.5-7B-Instruct | -0.027 | [-0.034, -0.021] | 0.0002 |
| Qwen2.5-1.5B-Instruct | Qwen2.5-14B-Instruct | -0.020 | [-0.027, -0.014] | 0.0002 |
| Qwen2.5-1.5B-Instruct | Qwen2.5-32B-Instruct | -0.049 | [-0.056, -0.041] | 0.0002 |
| Qwen2.5-1.5B-Instruct | Qwen3-4B-Instruct-2507 | -0.013 | [-0.020, -0.007] | 0.0002 |
| Qwen2.5-1.5B-Instruct | Qwen3-30B-A3B-Instruct-2507 | -0.038 | [-0.045, -0.031] | 0.0002 |
| Qwen2.5-3B-Instruct | Qwen2.5-7B-Instruct | -0.020 | [-0.026, -0.014] | 0.0002 |
| Qwen2.5-3B-Instruct | Qwen2.5-14B-Instruct | -0.013 | [-0.019, -0.007] | 0.0002 |
| Qwen2.5-3B-Instruct | Qwen2.5-32B-Instruct | -0.042 | [-0.048, -0.035] | 0.0002 |
| Qwen2.5-3B-Instruct | Qwen3-4B-Instruct-2507 | -0.006 | [-0.012, -0.000] | 0.0518 |
| Qwen2.5-3B-Instruct | Qwen3-30B-A3B-Instruct-2507 | -0.031 | [-0.037, -0.024] | 0.0002 |
| Qwen2.5-7B-Instruct | Qwen2.5-14B-Instruct | 0.007 | [0.000, 0.013] | 0.0402 |
| Qwen2.5-7B-Instruct | Qwen2.5-32B-Instruct | -0.022 | [-0.027, -0.016] | 0.0002 |
| Qwen2.5-7B-Instruct | Qwen3-4B-Instruct-2507 | 0.014 | [0.008, 0.019] | 0.0002 |
| Qwen2.5-7B-Instruct | Qwen3-30B-A3B-Instruct-2507 | -0.011 | [-0.017, -0.005] | 0.0004 |
| Qwen2.5-14B-Instruct | Qwen2.5-32B-Instruct | -0.028 | [-0.034, -0.023] | 0.0002 |
| Qwen2.5-14B-Instruct | Qwen3-4B-Instruct-2507 | 0.007 | [0.000, 0.014] | 0.0424 |
| Qwen2.5-14B-Instruct | Qwen3-30B-A3B-Instruct-2507 | -0.017 | [-0.024, -0.011] | 0.0002 |
| Qwen2.5-32B-Instruct | Qwen3-4B-Instruct-2507 | 0.035 | [0.029, 0.042] | 0.0002 |
| Qwen2.5-32B-Instruct | Qwen3-30B-A3B-Instruct-2507 | 0.011 | [0.005, 0.016] | 0.0006 |
| Qwen3-4B-Instruct-2507 | Qwen3-30B-A3B-Instruct-2507 | -0.024 | [-0.029, -0.019] | 0.0002 |

