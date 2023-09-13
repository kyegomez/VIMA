[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# VIM
A simple implementation of "VIMA: General Robot Manipulation with Multimodal Prompts"

[Original implementation Link](https://github.com/vimalabs/VIMA)

# Appreciation
* Lucidrains
* Agorians

# Install
`pip install vima`

---

# Usage
```python
import torch
from vima import Vima

# Generate a random input sequence
x = torch.randint(0, 256, (1, 1024)).cuda()

# Initialize VIMA model
model = Vima()

# Pass the input sequence through the model
output = model(x)
```

## MultiModal Iteration
* Pass in text and and image tensors into vima
```python
import torch
from vima.vima import VimaMultiModal

#usage
img = torch.randn(1, 3, 256, 256)
text = torch.randint(0, 20000, (1, 1024))


model = VimaMultiModal()
output = model(text, img)

```

# License
MIT

# Citations
```latex
@inproceedings{jiang2023vima,
  title     = {VIMA: General Robot Manipulation with Multimodal Prompts},
  author    = {Yunfan Jiang and Agrim Gupta and Zichen Zhang and Guanzhi Wang and Yongqiang Dou and Yanjun Chen and Li Fei-Fei and Anima Anandkumar and Yuke Zhu and Linxi Fan},
  booktitle = {Fortieth International Conference on Machine Learning},
  year      = {2023}
}
```