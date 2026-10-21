---
license: mit
language:
- en
base_model:
- ornith-ai/Ornith-1.5-35B-A3B
pipeline_tag: text-generation
---
# Model Card for Model ID

A fine mix-quantified and REAP-50 version for Ornith-1.5 that is specially designed for 12GB cards.

## Model Details

### Model Description

- **Base model: Ornith-1.5-35B-A3B Ttimms/Ornith-1.5-35B-A3B-REAP-50-bf16
- **Quantized by :TSalpha
- **License: mit
- **Model type: MoE

### quantify methods
- **layer0-14: IQ2_XXS
- **layer15-30: IQ3_XXS
- **layer31: IQ3_S
- **layer32-39: IQ4_XS
- **Shared experts(40layers): IQ4_XS                                                                             
- **attn_qkv / attn_gate、attn_q/k/v/o:Q8_0                
-  **ssm_alpha / ssm_beta / ssm_out:Q8_0              
- **token_embd / output                                 IQ4_XS            
- **all norm、ssm_a、ssm_conv1d、ssm_dt.bias、ffn_gate_inp(_shexp)、output_norm F32                  
## Uses

- **The model runs well inside a llama.cpp. And maximum context of 256K can be reached with params below on 12G card. 
- **Suggested params: -c 262144 -ngl 99 -ctk q5_1 -ctv q5_0 -fa on --jinja -np 1 --temp 1 --top-k 20 --top-p 0.95 --min-p 0.10

## Evaluation
-*10 chunk ppl comparison with original f16(lower means better performance)
- **model-----------------size--------ppl
- **f16--------------------- 34.57 GiB 2.3849    
- **IQ4_XS(this model)7.39 GiB 2.4425 
- **Q3_K------------------ 9.35 GiB 2.4958  
- **q2-60pct + imatrix- 8.03 GiB  2.5112     
- **Q4 swich to MXFP4 7.39 GiB  2.5335      
- **24IQ2+16IQ4--------        7.40 GiB  2.5537
- the other 5 files are not here.    
#### Hardware

Test runs on RTX5070ti laptop GPU(12G) with 90token/s of max speed.
Real agent coding have an output approxly 50t/s tested on Kimicode.

## License & Credits

     MIT (inherited from `ornith-ai/Ornith-1.5-35B-A3B`).

     - Base model: [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)
     - Pruned checkpoint:
            [Ttimms/Ornith-1.5-35B-A3B-REAP-50-bf16](https://huggingface.co/Ttimms/Ornith-1.5-35B-A3B-REAP-50-bf16)
       (REAP method: [CerebrasResearch/reap](https://github.com/CerebrasResearch/reap))
     - Quantization & imatrix: TSalpha (local llama.cpp)
