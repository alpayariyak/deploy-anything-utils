from utils import get_model, calculate_memory

def get_memory_size(model_repo: str, hf_token: str = None):
    """
    Fetches a model from the Hugging Face Hub and calculates its GB memory size
    in various data types.

    Parameters:
    - model_repo: str - The repository name of the model on Hugging Face Hub.
    - hf_token: Optional[str] - The Hugging Face token for authentication, if needed.

    Returns:
    A dictionary with GB memory sizes in float32, float16/bfloat16, int8, and int4.
    """
    # Fetch the model
    model = get_model(model_repo, library="transformers", access_token=hf_token or None)

    # Calculate memory size
    memory_sizes = calculate_memory(model, ["float32", "float16/bfloat16", "int8", "int4"])

    # Convert list of dicts to a single dict for easier readability
    memory_dict = {dtype_info['dtype']: dtype_info["Inference (GB)"] for dtype_info in memory_sizes}
    
    return memory_dict
