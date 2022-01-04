from transformers import list_adapters

# source can be "ah" (AdapterHub), "hf" (huggingface.co) or None (for both, default)
adapter_infos = list_adapters(source="ah", model_name="bert-base-uncased")

for adapter_info in adapter_infos:
    print("Id:", adapter_info.adapter_id)
    print("Model name:", adapter_info.model_name)
    print("Uploaded by:", adapter_info.username)