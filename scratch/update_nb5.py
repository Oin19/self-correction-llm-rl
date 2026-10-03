import json

nb_path = 'notebooks/05_ppo_training_corrected.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell1_source = [
    "!pip install -q 'trl<0.12.0' peft transformers datasets bitsandbytes accelerate\n",
    "import os,sys,shutil,torch\n",
    "repo=os.path.abspath(os.getcwd())\n",
    "if os.path.isdir('/kaggle/input') and not os.path.exists(os.path.join(repo,'src')):\n",
    "    for root,dirs,files in os.walk('/kaggle/input'):\n",
    "        if 'src' in dirs and os.path.exists(os.path.join(root,'src','training','ppo.py')):\n",
    "            shutil.copytree(os.path.join(root,'src'),os.path.join('/kaggle/working','src'),dirs_exist_ok=True); repo='/kaggle/working'; break\n",
    "if repo not in sys.path: sys.path.insert(0,repo)\n",
    "for mod in list(sys.modules.keys()):\n",
    "    if mod.startswith('src.'):\n",
    "        del sys.modules[mod]\n",
    "from datasets import load_dataset\n",
    "from transformers import AutoTokenizer\n",
    "from src.training.ppo import run_ppo_training\n",
    "MODEL='deepseek-ai/deepseek-coder-1.3b-instruct'\n",
    "SFT='./checkpoints/sft/final'\n",
    "if not os.path.isfile(os.path.join(SFT,'adapter_config.json')):\n",
    "    raise FileNotFoundError('SFT adapter checkpoint required: '+SFT)\n",
    "if not os.path.isfile(os.path.join(SFT,'adapter_model.safetensors')):\n",
    "    raise FileNotFoundError('SFT adapter weights missing: '+SFT)\n",
    "print('Verified SFT adapter:', SFT)\n",
    "apps=load_dataset('codeparrot/apps',revision='refs/convert/parquet',split='train[:16]')\n",
    "apps=apps.filter(lambda x: bool(x.get('solutions')))\n",
    "tok=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True)\n",
    "print('PPO smoke dataset:',len(apps))"
]

nb['cells'][1]['source'] = cell1_source

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Updated notebooks/05_ppo_training_corrected.ipynb Cell 1 successfully!")
