import torch
from accelerate.commands.estimate import check_has_model, create_empty_model
from urllib.parse import urlparse
from accelerate.utils import calculate_maximum_sizes
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

DTYPE_MODIFIER = {"float32": 1, "float16/bfloat16": 2, "int8": 4, "int4": 8}

def translate_llama2(text):
    "Translates llama-2 to its hf counterpart"
    if not text.endswith("-hf"):
        return text + "-hf"
    return text

def get_model(model_name, library, access_token):
    """    
    Args:
        model_name (str): The name or URL of the model to be retrieved.
        library (str): The library to use ('auto' for automatic detection).
        access_token (str): Access token for gated repositories.
        
    Returns:
        Any: An initialized model object.
        
    Raises:
        GatedRepoError: If the model is gated and access token is invalid or not provided.
        RepositoryNotFoundError: If the model cannot be found on the Hub.
        ValueError: For various errors related to model loading and library metadata.
    """
    
    # Adjust model_name if it belongs to a specific category
    if "meta-llama" in model_name:
        model_name = translate_llama2(model_name)
        
    # Auto-detect library if 'auto' is specified
    if library == "auto":
        library = None
    
    # Process model name to extract from URL if necessary
    model_name = extract_from_url(model_name)
    
    try:
        # Attempt to create an empty model instance
        model = create_empty_model(model_name, library_name=library, trust_remote_code=True, access_token=access_token)
    except GatedRepoError:
        raise GatedRepoError(
            "Model `{}` is gated. Please ensure to provide a valid access token. Access tokens can be found at https://huggingface.co/settings/tokens.".format(model_name)
        )
    except RepositoryNotFoundError:
        raise RepositoryNotFoundError("Model `{}` was not found on the Hub. Please try another model name.".format(model_name))
    except ValueError as e:
        # Handle missing library metadata or other value errors
        if "does not have any library metadata" in str(e):
            raise ValueError(
                "Model `{}` does not have library metadata on the Hub. Please manually select a `library_name` to use (e.g., `transformers`).".format(model_name)
            )
        else:
            # Attempt to resolve library issues from exception
            library = check_has_model(e)
            if library != "unknown":
                raise ValueError(
                    "Tried to load `{}` with `{}` but a possible model was not found inside the repo.".format(model_name, library)
                )
            else:
                raise e
    except ImportError:
        # Try loading the model without trusting remote code as a fallback
        try:
            model = create_empty_model(model_name, library_name=library, trust_remote_code=False, access_token=access_token)
        except Exception as e:
            # Handle generic errors last
            raise ValueError(
                "Model `{}` had an error. Please open a discussion on the model's page with the error message: `{}`".format(model_name, e)
            )
    except Exception as e:
        # Catch-all for other exceptions
        raise ValueError(
            "Model `{}` encountered an error. Open a discussion on the model's page with the error message: `{}`".format(model_name, e)
        )
    
    return model


def extract_from_url(name: str):
    "Checks if `name` is a URL, and if so converts it to a model name"
    is_url = False
    try:
        result = urlparse(name)
        is_url = all([result.scheme, result.netloc])
    except Exception:
        is_url = False
    # Pass through if not a URL
    if not is_url:
        return name
    else:
        path = result.path
        return path[1:]

def calculate_memory(model: torch.nn.Module, options: list):
    "Calculates the memory usage for a model init on `meta` device"
    total_size, largest_layer = calculate_maximum_sizes(model)
    data = []
    for dtype in options:
        dtype_total_size = total_size
        dtype_largest_layer = largest_layer[0]

        modifier = DTYPE_MODIFIER[dtype]
        dtype_total_size /= modifier
        dtype_largest_layer /= modifier
        dtype_inference = dtype_total_size * 1.2  / (1024**3)
        data.append(
            {
                "dtype": dtype,
                "Inference (GB)" : dtype_inference,
            }
        )
    return data