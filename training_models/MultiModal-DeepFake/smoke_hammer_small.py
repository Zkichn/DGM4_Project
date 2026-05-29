import argparse
import torch
try:
    import ruamel_yaml as yaml
except ImportError:
    from ruamel import yaml
from transformers import BertTokenizerFast
from dataset import create_dataset, create_loader
from models.HAMMER import HAMMER
from models.vit import interpolate_pos_embed
from train import text_input_adjust

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='configs/hammer_small_train.yaml')
parser.add_argument('--checkpoint', default='local_models/ALBEF_4M/ALBEF_4M.pth')
parser.add_argument('--batch_size', type=int, default=2)
args = parser.parse_args()
config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
config['batch_size_train'] = args.batch_size
config['batch_size_val'] = args.batch_size
config['dataset_division'] = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
assert device.type == 'cuda', 'CUDA is not available'
train_dataset, _ = create_dataset(config)
train_loader, _ = create_loader([train_dataset, train_dataset], [None, None], [args.batch_size, args.batch_size], [0, 0], [True, False], [None, None])
tokenizer = BertTokenizerFast.from_pretrained('local_models/bert-base-uncased-ms')
model = HAMMER(args=argparse.Namespace(token_momentum=True), config=config, text_encoder='local_models/bert-base-uncased-ms', tokenizer=tokenizer, init_deit=True).to(device)
ckpt = torch.load(args.checkpoint, map_location='cpu')
state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
if 'visual_encoder.pos_embed' in state:
    state['visual_encoder.pos_embed'] = interpolate_pos_embed(state['visual_encoder.pos_embed'], model.visual_encoder)
msg = model.load_state_dict(state, strict=False)
print('load_msg', msg)
model.train()
if device.type == 'cuda':
    torch.cuda.reset_peak_memory_stats(device)
image, label, text, fake_image_box, fake_word_pos, W, H = next(iter(train_loader))
image = image.to(device, non_blocking=True)
fake_image_box = fake_image_box.to(device, non_blocking=True)
text_input = tokenizer(text, max_length=128, truncation=True, add_special_tokens=True, return_attention_mask=True, return_token_type_ids=False)
text_input, fake_token_pos = text_input_adjust(text_input, fake_word_pos, device)
losses = model(image, label, text_input, fake_image_box, fake_token_pos, alpha=config['alpha'])
loss = sum(losses)
loss.backward()
peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == 'cuda' else 0
print('smoke_ok', {'batch': len(label), 'losses': [float(x.detach().cpu()) for x in losses], 'total_loss': float(loss.detach().cpu()), 'peak_mem_gb': round(peak_mem_gb, 3)})
