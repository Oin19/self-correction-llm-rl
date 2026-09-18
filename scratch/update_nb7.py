import json

nb_path = 'notebooks/07_evaluation_and_ablations_corrected.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell1_source = [
    "!pip install -q peft transformers datasets pandas\n",
    "import os,sys,shutil,torch,pandas as pd\n",
    "repo=os.path.abspath(os.getcwd())\n",
    "if os.path.isdir('/kaggle/input') and not os.path.exists(os.path.join(repo,'src')):\n",
    "    for root,dirs,files in os.walk('/kaggle/input'):\n",
    "        if 'src' in dirs and os.path.exists(os.path.join(root,'src','evaluation','metrics.py')):\n",
    "            shutil.copytree(os.path.join(root,'src'),os.path.join('/kaggle/working','src'),dirs_exist_ok=True); repo='/kaggle/working'; break\n",
    "sys.path.insert(0,repo)\n",
    "from datasets import load_dataset\n",
    "from transformers import AutoModelForCausalLM,AutoTokenizer\n",
    "from peft import PeftModel\n",
    "from src.evaluation.metrics import evaluate\n",
    "BASE='deepseek-ai/deepseek-coder-1.3b-instruct'\n",
    "he=load_dataset('openai_humaneval',split='test'); mbpp=load_dataset('mbpp',split='test')\n",
    "print('Benchmarks:',len(he),'HumanEval /',len(mbpp),'MBPP')"
]

nb['cells'][1]['source'] = cell1_source

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Updated notebooks/07_evaluation_and_ablations_corrected.ipynb Cell 1 successfully!")
