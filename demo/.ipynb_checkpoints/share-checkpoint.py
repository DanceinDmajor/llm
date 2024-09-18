from modelscope import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


# 加载模型和分词器
model_dir = '/root/autodl-tmp/TongyiFinance/Tongyi-Finance-14B-Chat'
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cuda:0", trust_remote_code=True, bf16=True).eval()
model.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True, temperature = 0.0001, top_p = 1, do_sample = False, seed = 1234)

