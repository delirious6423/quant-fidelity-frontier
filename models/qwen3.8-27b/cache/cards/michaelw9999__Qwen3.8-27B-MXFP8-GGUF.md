---
base_model:
- Qwen/Qwen3.8-27B
tags:
- MXFP8
- llama.cpp
- Qwen3.8
- Qwen3.8-27B
---
More information and benchmarks will be posted soon.<BR>
This was converted from the original Qwen/Qwen3.8-27B repository.<BR><BR>
It was quantized using my <A HREF="https://github.com/michaelw9999/advanced-gguf-quantizer/">advanced-gguf-quantizer</A> tool.<BR>
<BR>
To use this model, you must build my unofficial MXFP8 fork of llama.cpp at:<BR>
<A HREF="https://github.com/michaelw9999/llama.cpp/tree/mxfp8_cuda">https://github.com/michaelw9999/llama.cpp/tree/mxfp8_cuda</A><BR>
On Blackwell, I am seeing that MXFP8 is faster than Q8_0 and is comparable in quality.
<BR>
I will post some mixed NVFP4/MXFP6/MXFP8 models soon that should provide the fastest possible Blackwell models.
<BR>
Please let me know your feedback and if you have any problems, I will be happy to help!<BR>