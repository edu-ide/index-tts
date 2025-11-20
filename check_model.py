#!/usr/bin/env python3
import torch

checkpoint = torch.load('/mnt/sda1/models/index-tts-ko/checkpoints/latest.pth', map_location='cpu')
model_state = checkpoint['model']
print('Model state_dict keys with "text" or "embedding":')
for key in sorted(model_state.keys()):
    if 'text' in key.lower() or 'embedding' in key.lower():
        val = model_state[key]
        if hasattr(val, 'shape'):
            print(f'  {key}: {val.shape}')

print('\nNumber of text tokens in trained model:')
if 'text_embedding.weight' in model_state:
    print(f'  text_embedding.weight: {model_state["text_embedding.weight"].shape[0]}')
if 'text_head.weight' in model_state:
    print(f'  text_head.weight: {model_state["text_head.weight"].shape[0]}')
