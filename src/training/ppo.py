"""Execution-guided PPO training."""
import gc, json, os
import torch
from peft import PeftModel
from src.execution.executor import PythonSandbox, run_code
from src.rewards.execution_reward import compute_partial_reward
from trl import PPOTrainer, PPOConfig
from trl.models.modeling_value_head import AutoModelForCausalLMWithValueHead

def normalize_tests(ex):
    if isinstance(ex.get("test"), str) and ex["test"].strip(): return [ex["test"]]
    if isinstance(ex.get("test_list"), list) and ex["test_list"]: return [{"assertion": x} for x in ex["test_list"] if isinstance(x,str) and x.strip()]
    raw=ex.get("input_output",ex.get("test_cases",[]))
    if isinstance(raw,str):
        try: raw=json.loads(raw)
        except Exception: return []
    if isinstance(raw,list): return raw
    if not isinstance(raw,dict): return []
    ins,outs,fn=raw.get("inputs",[]),raw.get("outputs",[]),raw.get("fn_name")
    cases=[]
    for inp,out in zip(ins,outs):
        if fn:
            args=", ".join(repr(x) for x in inp) if isinstance(inp,list) else repr(inp)
            cases.append({"assertion":f"assert {fn}({args}) == {out!r}"})
        else: cases.append({"input":"\n".join(inp) if isinstance(inp,list) else str(inp),"output":"\n".join(out) if isinstance(out,list) else str(out)})
    return cases

def load_sft_adapter(path,device_map):
    cfg_path=os.path.join(path,"adapter_config.json")
    if not os.path.isfile(cfg_path): raise FileNotFoundError("SFT adapter not found: %s"%path)
    with open(cfg_path,encoding="utf-8") as f: cfg=json.load(f)
    base=cfg.get("base_model_name_or_path")
    if not base: raise ValueError("SFT adapter has no base_model_name_or_path")
    wrapper=AutoModelForCausalLMWithValueHead.from_pretrained(base,torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,device_map=device_map,trust_remote_code=True)
    wrapper.pretrained_model=PeftModel.from_pretrained(wrapper.pretrained_model,path,is_trainable=True)
    return wrapper,base

def run_ppo_training(sft_model_path,tokenizer,dataset,output_dir="./checkpoints/ppo",num_epochs=1,learning_rate=1e-6,batch_size=2,mini_batch_size=1,gradient_accumulation_steps=2,init_kl_coef=0.02,target_kl=6.0,max_steps=10):
    tokenizer.padding_side="left"
    if tokenizer.pad_token is None: tokenizer.pad_token=tokenizer.eos_token
    # Filter dataset to ensure every sample has non-empty executable benchmark tests
    dataset=dataset.filter(lambda ex: len(normalize_tests(ex)) > 0)
    if len(dataset)==0: raise ValueError("No dataset examples contain valid executable benchmark tests")
    all_benchmark_tests=[normalize_tests(ex) for ex in dataset]
    def encode(ex):
        p=ex.get("question",ex.get("prompt",""))
        t=tokenizer("### Problem:\n%s\n\n### Solution:\n```python\n"%p,truncation=True,max_length=256)
        return {"input_ids":t["input_ids"],"benchmark_tests":normalize_tests(ex)}
    dataset=dataset.map(encode)
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    ng=torch.cuda.device_count() if torch.cuda.is_available() else 0
    device_map="auto" if ng>=2 else ({"":0} if ng==1 else None)
    model,base=load_sft_adapter(sft_model_path,device_map)
    if hasattr(model.pretrained_model,"gradient_checkpointing_enable"): model.pretrained_model.gradient_checkpointing_enable()
    for p in model.pretrained_model.parameters(): p.requires_grad=False
    for n,p in model.named_parameters():
        if "lora_" in n or "v_head" in n or "summary" in n: p.requires_grad=True
    trainable=[p for p in model.parameters() if p.requires_grad]
    if not trainable: raise RuntimeError("PPO has no trainable parameters")
    opt=torch.optim.AdamW(trainable,lr=learning_rate)
    cfg=PPOConfig(model_name=base,learning_rate=learning_rate,batch_size=batch_size,mini_batch_size=mini_batch_size,gradient_accumulation_steps=gradient_accumulation_steps,kl_penalty="kl",init_kl_coef=init_kl_coef,target_kl=target_kl)
    collate=lambda rows:{k:[r[k] for r in rows] for k in rows[0]}
    trainer=PPOTrainer(config=cfg,model=model,ref_model=None,tokenizer=tokenizer,dataset=dataset,optimizer=opt,data_collator=collate)
    sandbox=PythonSandbox(default_timeout=5.0,max_memory_mb=1024.0)
    kwargs={"max_new_tokens":64,"do_sample":True,"top_p":0.95,"pad_token_id":tokenizer.pad_token_id,"eos_token_id":tokenizer.eos_token_id}
    for step,batch in enumerate(trainer.dataloader,1):
        queries=[q.squeeze() if isinstance(q,torch.Tensor) and q.dim()>1 else torch.as_tensor(q,dtype=torch.long) for q in batch["input_ids"]]
        with torch.no_grad(): responses=trainer.generate(queries,**kwargs)
        rewards=[]
        batch_tests = batch.get("benchmark_tests")
        if batch_tests is None:
            start_i = (step - 1) * batch_size
            end_i = start_i + len(queries)
            batch_tests = all_benchmark_tests[start_i:end_i]
        for response,tests in zip(responses,batch_tests):
            if not tests: raise ValueError("No executable benchmark tests; refusing process-only reward")
            code=tokenizer.decode(response,skip_special_tokens=True)
            if all(isinstance(t,str) for t in tests):
                raw=run_code(code+"\n\n"+"\n\n".join(tests),timeout=5)
                result={"status":raw["status"],"passed_tests":int(raw["status"]=="AC"),"total_tests":1}
            else: result=sandbox.run_tests(code,tests).to_dict()
            rewards.append(torch.tensor(compute_partial_reward(result),dtype=torch.float32))
        stats=trainer.step(queries,responses,rewards)
        print("PPO step %d reward=%.3f kl=%.3f"%(step,stats.get("ppo/mean_scores",0.0),stats.get("objective/kl",0.0)),flush=True)
        if max_steps and step>=max_steps: break
    os.makedirs(os.path.join(output_dir,"final"),exist_ok=True)
    model.save_pretrained(os.path.join(output_dir,"final")); tokenizer.save_pretrained(os.path.join(output_dir,"final"))
    return trainer
