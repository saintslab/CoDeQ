# README #

This is the official Pytorch implementation of 
"[CoDeQ: End-to-End Joint Model Compression with Dead-Zone Quantizer for
High-Sparsity and Low-Precision Networks](https://arxiv.org/abs/2512.12981)", Wenshøj et al. 2025.

![results](utils/fig1.png)

### Quick Start
```python
from src.quantizer import DeadZoneLDZCompander
from src.utils_quantization import attach_weight_quantizers, toggle_quantization

model = ...

# Attach weight quantizers
attach_weight_quantizers(
    model=model,
    exclude_layers=EXCLUDE,
    quantizer=DeadZoneLDZ,
    quantizer_kwargs=QUANT_ARGS,
    enabled=True
)

# Separate params into groups
optimizer = torch.optim.AdamW([
    {'params': base_params, 'lr': lr,          'weight_decay': weight_decay},
    {'params': dz_params,   'lr': lr_dz,       'weight_decay': weight_decay_dz},
    {'params': bit_params,  'lr': lr_bit,      'weight_decay': weight_decay_bit},
], **kwargs)

# Enable QAT and train
toggle_quantization(model, enabled=True)
for epoch in epochs:
    ...
```


### Usage guidelines ###

* Kindly cite our publication if you use any part of the code
```
@article{wenshoj2025codeq,
 	title={CoDeQ: End-to-End Joint Model Compression with Dead-Zone Quantizer for
High-Sparsity and Low-Precision Networks},
	author={Jonathan Wenshøj and Tong Chen and Bob Pepin and Raghavendra Selvan},
	journal={Arxiv},
 	note={arXiv preprint arXiv:2512.12981},
	year={2025}}
```
### Who do I talk to? ###

* jowe@di.ku.dk, raghav@di.ku.dk

