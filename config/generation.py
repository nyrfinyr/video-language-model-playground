from transformers import GenerationConfig


PRECISE = GenerationConfig(
    do_sample=False,
    num_beams=1,
    max_new_tokens=256,
    repetition_penalty=1.1,
)

BALANCED = GenerationConfig(
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    max_new_tokens=512,
    repetition_penalty=1.1,
)

CREATIVE = GenerationConfig(
    do_sample=True,
    temperature=1.1,
    top_p=0.95,
    top_k=50,
    max_new_tokens=1024,
    repetition_penalty=1.0,
)