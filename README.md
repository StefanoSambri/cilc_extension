# An Extended Evaluation of Open-Weight Large Language Models for Neuro-Symbolic Integration

Repository to gather the code for paper: An Extended Evaluation of Open-Weight Large Language Models for Neuro-Symbolic Integration.

## Models

Multimodal models scripts are located in `multimodal_models/`.

Multimodal models requirements are located in `multimodal_models/requirements/`.

Text models scripts are located in `text_models/`.

Text models requirements are located in `text_models/requirements/`.

### Multimodal snapshots
```
google/gemma-3-4b-it                            093f9f388b31de276ce2de164bdc2081324b9767
google/gemma-3-12b-it                           96b6f1eccf38110c56df3a15bffe176da04bfd80
google/gemma-3-27b-it                           05ad3404e59d6023443cb575daa05336842228a
zai-org/GLM-4.1V-9B-Thinking	                e9a4c5e94f4a095c353f4152d520a2644a553b2
OpenGVLab/InternVL2-1B	                        0d75ccd166b1d0b79446ae6c5d1a4a667f1e6187
OpenGVLab/InternVL2-2B	                        e4f6747bd20f139e637642c6a058c6bd00b36919
OpenGVLab/InternVL2-4B	                        6eb21ec278fbdb03b751a3d48d8dfc9724e05608
OpenGVLab/InternVL2-8B	                        6fb9ad6924f69424e57fab2ab061d707688f0296
OpenGVLab/InternVL2-26B	                        46f37972e05604cc44a59f72334df220fa197a7b
OpenGVLab/InternVL2_5-1B	                    9d423ea1ae9f893897ee3f7493141073f5afcf22
OpenGVLab/InternVL2_5-2B	                    573169ee54df216786bb9a189e9a32a060a008cf
OpenGVLab/InternVL2_5-4B	                    2cf4a8158bbc40d35015e7c63b527890de4d27b3
OpenGVLab/InternVL2_5-8B	                    e9e4c0dc1db56bfab10458671519b7fa3dd29463
OpenGVLab/InternVL2_5-26B	                    b537a9974b89cb621e9e6b9e7ebe2a904334fff0
OpenGVLab/InternVL2_5-38B	                    32ba5a61b8a870d9ac946b639b134abed529d45c
OpenGVLab/InternVL3-1B-Instruct	                f91c6391476dbc8fd07674d2589e8168d3ca667f
OpenGVLab/InternVL3-2B-Instruct	                f6c7b60375759170fd49f5e9e298e2178485c5ba
OpenGVLab/InternVL3-8B-Instruct	                ddb3a169d5582e5c76e0809a128e55ab63686ada
OpenGVLab/InternVL3-14B-Instruct	            6ffe6f06d88b6e7e3bd6eccdf73028fae7855d38
OpenGVLab/InternVL3-38B-Instruct	            150ad666e0c733b809742decded1f7484995b99c
XiaomiMiMo/MiMo-VL-7B-RL-2508	                4bfb270765825d2fa059011deb4c96fdd579be6f
mistralai/Mistral-Small-3.2-24B-Instruct-2506	95a6d26c4bfb886c58daf9d3f7332c857cb27b43
mistralai/Mistral-Small-3.1-24B-Instruct-2503	68faf511d618ef198fef186659617cfd2eb8e33a
allenai/Molmo-7B-D-0924                         cab33fb7f1a40091911f81165f8481920621948f
allenai/Molmo-7B-O-0924                         7a8c4bf80c839c243a6908c6ebbb0f1ee576d7ca
AIDC-AI/Ovis2.5-2B	                            393c932b2a03e28eb9aaa503e3c4ab3ad384d958
AIDC-AI/Ovis2.5-9B	                            d73b2283ae2a930b7762f8d7b8b8a3f0f3b5c3bd
AIDC-AI/Ovis2-1B	                            642d0807bf09b89ed52ea97230ceab26c9a6ed2d
AIDC-AI/Ovis2-2B	                            63d303670a24612b2d0dd368490fcd7f75989d73
AIDC-AI/Ovis2-4B	                            30d3ed6869e44de8e8d986d79fdc772bfa1c66fc
AIDC-AI/Ovis2-8B	                            c0730f752cf605d44788a08151cfede0caab714d
AIDC-AI/Ovis2-16B	                            5dea2745b747a63fcf41e934b19653b1b8d6b7e6
AIDC-AI/Ovis2-34B	                            53006418a206066ab75217483dc49da0afdfa33a
Qwen/Qwen2-VL-2B-Instruct	                    895c3a49bc3fa70a340399125c650a463535e71c
Qwen/Qwen2-VL-7B-Instruct	                    eed13092ef92e448dd6875b2a00151bd3f7db0ac
Qwen/Qwen2.5-VL-3B-Instruct	                    66285546d2b821cf421d4f5eb2576359d3770cd3
Qwen/Qwen2.5-VL-7B-Instruct	                    cc594898137f460bfe9f0759e9844b3ce807cfb5
Qwen/Qwen2.5-VL-32B-Instruct	                7cfb30d71a1f4f49a57592323337a4a4727301da
Qwen/Qwen3-VL-2B-Instruct	                    89644892e4d85e24eaac8bacfd4f463576704203
Qwen/Qwen3-VL-4B-Instruct	                    ebb281ec70b05090aa6165b016eac8ec08e71b17
Qwen/Qwen3-VL-8B-Instruct	                    0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
Qwen/Qwen3-VL-32B-Instruct	                    0cfaf48183f594c314753d30a4c4974bc75f3ccb
Qwen/Qwen3-VL-30B-A3B-Instruct	                9c4b90e1e4ba969fd3b5378b57d966d725f1b86c
Skywork/Skywork-VL-Reward-7B	                cfc6496e451f4e636d7e78ce6249a4ee0efb89a1
```

### Text snapshots
```
deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B	    ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562
deepseek-ai/DeepSeek-R1-Distill-Qwen-7B	        916b56a44061fd5cd7d6a8fb632557ed4f724f60
deepseek-ai/DeepSeek-R1-Distill-Qwen-14B	    1df8507178afcc1bef68cd8c393f61a886323761
deepseek-ai/DeepSeek-R1-Distill-Qwen-32B	    711ad2ea6aa40cfca18895e8aca02ab92df1a746
tiiuae/falcon-7b-instruct	                    8782b5c5d8c9290412416618f36a133653e85285
tiiuae/Falcon-H1-Tiny-90M-Instruct	            e6389502a0b12cd8da894b395ba5bf7436873b16
tiiuae/Falcon-H1R-7B	                        a6f74bf181389908efd6970878d7ee2b42f5d417
tiiuae/Falcon3-1B-Instruct	                    28ba2251970a01dd1edc7ba7dad2eb71216ccfdf
tiiuae/Falcon3-3B-Instruct	                    411bb94318f94f7a5735b77109f456b1e74b42a1
tiiuae/Falcon3-7B-Instruct	                    1e57a0ecd176c7c139f289c60a74e57f887c3dfb
tiiuae/Falcon3-10B-Instruct	                    8799bc6aec0152757221dc6b272d824642db6202
tiiuae/Falcon3-Mamba-7B-Instruct	            79268d5c8e650ec0ec24aad2729bfc906f569580
tiiuae/Falcon-E-1B-Instruct	                    d20167318084c330b205e221eceed19354197bdc
tiiuae/Falcon-E-3B-Instruct	                    deeb1e557f4eee27f0ebb2389604b8a67c56a74b
openai/gpt-oss-20b	                            6cee5e81ee83917806bbde320786a8fb61efebee
mistralai/Mistral-7B-Instruct-v0.1	            ec5deb64f2c6e6fa90c1abf74a91d5c93a9669ca
mistralai/Mistral-7B-Instruct-v0.2	            63a8b081895390a26e140280378bc85ec8bce07a
mistralai/Mistral-7B-Instruct-v0.3	            c170c708c41dac9275d15a8fff4eca08d52bab71
mistralai/Mistral-Nemo-Instruct-2407	        04d8a90549d23fc6bd7f642064003592df51e9b3
mistralai/Codestral-22B-v0.1	                28b1c1a51dabe9d86ca8c41420ada1984632498f
mistralai/Mistral-Small-24B-Instruct-2501	    9527884be6e5616bdd54de542f9ae13384489724
microsoft/Phi-3-mini-128k-instruct	            f3c06aed622e14ca0abf5115094e4fc9a9948f36
microsoft/Phi-3.5-mini-instruct	                2fe192450127e6a83f7441aef6e3ca586c338b77
microsoft/Phi-3-small-128k-instruct	            e95b185a59a58450c9b1dd7347a9dee78f18113e
microsoft/Phi-3-medium-128k-instruct	        a088b37c71d441ab6d862bb3fcfe6165b3014702
microsoft/Phi-3.5-MoE-instruct	                43688451b462a3351d8580625ebe1931adb3986d
microsoft/Phi-4-mini-instruct	                cfbefacb99257ffa30c83adab238a50856ac3083
microsoft/Phi-4-mini-reasoning	                0e3b1e2d02ee478a3743abe3f629e9c0cb722e0a
Qwen/Qwen2-0.5B-Instruct	                    c540970f9e29518b1d8f06ab8b24cba66ad77b6d
Qwen/Qwen2-1.5B-Instruct	                    ba1cf1846d7df0a0591d6c00649f57e798519da8
Qwen/Qwen2-7B-Instruct	                        f2826a00ceef68f0f2b946d945ecc0477ce4450c
Qwen/Qwen2.5-0.5B-Instruct	                    7ae557604adf67be50417f59c2c2f167def9a775
Qwen/Qwen2.5-1.5B-Instruct	                    989aa7980e4cf806f80c7fef2b1adb7bc71aa306
Qwen/Qwen2.5-3B-Instruct	                    aa8e72537993ba99e69dfaafa59ed015b17504d1
Qwen/Qwen2.5-7B-Instruct	                    a09a35458c702b33eeacc393d103063234e8bc28
Qwen/Qwen2.5-14B-Instruct	                    cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8
Qwen/Qwen2.5-32B-Instruct	                    5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd
Qwen/Qwen3-0.6B	                                c1899de289a04d12100db370d81485cdf75e47ca
Qwen/Qwen3-1.7B	                                70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
Qwen/Qwen3-4B	                                1cfa9a7208912126459214e8b04321603b3df60c
Qwen/Qwen3-8B	                                b968826d9c46dd6066d109eabc6255188de91218
Qwen/Qwen3-14B	                                40c069824f4251a91eefaf281ebe4c544efd3e18
Qwen/Qwen3-32B	                                9216db5781bf21249d130ec9da846c4624c16137
Qwen/Qwen3-4B-Instruct-2507	                    cdbee75f17c01a7cc42f958dc650907174af0554
Qwen/Qwen3-30B-A3B-Instruct-2507	            0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe
Qwen/QwQ-32B	                                976055f8c83f394f35dbd3ab09a285a984907bd0
```
## Dataset

Image dataset is located in `_dataset/dataset_img/`.

Text dataset is located in `_dataset/dataset_txt.json`.

## Prompts

Prompts used are located in `utility/prompts.py`.

- `prompt_ttt` Used for images with classical handwritten symbols.

- `prompt_mnist` Used for images with MNIST digits.

- `prompt` Used for GMS8K istances.

## Results

Multimodal models results are located in `_results/multimodal/`.

Text models results are located in `_results/text/`.