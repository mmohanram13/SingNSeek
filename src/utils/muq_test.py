import torch, librosa
from muq import MuQ

# Auto-detect device: MPS for Apple Silicon, CUDA for NVIDIA, or CPU as fallback
if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print(f'Using device: {device}')

wav, sr = librosa.load("dataset/Mun Paniya.wav", sr = 24000)
wavs = torch.tensor(wav).unsqueeze(0).to(device) 

# This will automatically fetch the checkpoint from huggingface
muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
muq = muq.to(device).eval()

with torch.no_grad():
    output = muq(wavs, output_hidden_states=True)

print('Total number of layers: ', len(output.hidden_states))
print('Feature shape: ', output.last_hidden_state.shape)

print(output)