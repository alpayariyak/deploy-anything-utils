from hfmodel import HFModel

if __name__ == "__main__":
    model_repo = "mistralai/Mistral-7B-Instruct-v0.2" 
    hf_token = None
    model = HFModel(model_repo, hf_token)
    print(f"Memory sizes in GB: {model.memory_sizes}")
    print(f"Model has chat template: {model.model_has_chat_template}")
    print(f"Model max length: {model.max_length}")