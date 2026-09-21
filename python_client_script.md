This sample Python client script will interacts with the LlamaEngine API. This script will load a model, generate a response based on a prompt, and print both the response and any status messages.

### Prerequisites

1. **Install Required Packages**: Ensure you have `ctypes` for interfacing with C libraries.
2. **LlamaEngine Library**: Make sure the LlamaEngine library is available in your system and accessible via a path.

### Python Client Script

Here's a sample Python script that demonstrates how to use the LlamaEngine API:

```python
import ctypes
from ctypes import c_char_p, c_size_t, c_void_p, CFUNCTYPE

# Define types for function pointers
GenerateResponseFunc = CFUNCTYPE(ctypes.c_bool, c_char_p, CFUNCTYPE(None, c_char_p, c_void_p), CFUNCTYPE(None, c_char_p, c_void_p), c_void_p)
ParseGGUFFunc = CFUNCTYPE(c_char_p, c_char_p, CFUNCTYPE(None, c_char_p, ctypes.c_int, c_void_p, c_void_p), CFUNCTYPE(None, c_char_p), c_void_p)

# Define the ModelParameter struct
class ModelParameter(ctypes.Structure):
    _fields_ = [
        ("key", c_char_p),
        ("type", ctypes.c_int),
        ("value", c_void_p)
    ]

# Load the LlamaEngine library
llama_engine = ctypes.CDLL('/path/to/LlamaEngined.dll')  # Update with the actual path to your DLL

# Define function prototypes
loadModel = llama_engine.loadModel
loadModel.argtypes = [c_char_p, ctypes.POINTER(ModelParameter), c_size_t, CFUNCTYPE(None, c_char_p)]
loadModel.restype = ctypes.c_bool

generateResponse = llama_engine.generateResponse
generateResponse.argtypes = [c_char_p, GenerateResponseFunc, GenerateResponseFunc, c_void_p]
generateResponse.restype = ctypes.c_bool

getLastResponse = llama_engine.getLastResponse
getLastResponse.argtypes = []
getLastResponse.restype = c_char_p

# Callback for logging messages
def log_callback(message):
    print(f"Log: {message.decode('utf-8')}")

log_callback_type = CFUNCTYPE(None, c_char_p)
log_callback_func = log_callback_type(log_callback)

# Callback for response tokens (streaming)
def stream_callback(token, user_data):
    print(f"Response Token: {token.decode('utf-8')}")

stream_callback_type = CFUNCTYPE(None, c_char_p, c_void_p)
stream_callback_func = stream_callback_type(stream_callback)

# Callback for final response
def final_callback(complete_response, user_data):
    print(f"Final Response: {complete_response.decode('utf-8')}")

final_callback_type = CFUNCTYPE(None, c_char_p, c_void_p)
final_callback_func = final_callback_type(final_callback)

# Example usage
def main():
    # Define model parameters
    params = [
        ModelParameter(b"temperature", 0, ctypes.cast(ctypes.pointer(ctypes.c_float(0.7)), c_void_p)),
        ModelParameter(b"max_tokens", 1, ctypes.cast(ctypes.pointer(ctypes.c_int(256)), c_void_p))
    ]

    # Convert list to array
    param_array = (ModelParameter * len(params))(*params)

    # Load model
    backend_type = b"CUDA"
    if not loadModel(backend_type, param_array, len(param_array), log_callback_func):
        print("Failed to load model")
        return

    # Generate response
    prompt = b"Once upon a time in a land far, far away,"
    user_data = None  # You can pass any custom data here if needed
    if not generateResponse(prompt, stream_callback_func, final_callback_func, user_data):
        print("Failed to generate response")

if __name__ == "__main__":
    main()
```

### Explanation

1. **Loading the Library**:
    - The script uses `ctypes.CDLL` to load the LlamaEngine library (`LlamaEngined.dll`). Make sure to update the path to match where your DLL is located.

2. **Defining Function Pointers and Callbacks**:
    - The script defines types for function pointers using `CFUNCTYPE`.
    - It also defines callback functions for logging, streaming response tokens, and final response.

3. **Model Parameters**:
    - The `ModelParameter` struct is defined to match the C structure.
    - Example parameters (`temperature` and `max_tokens`) are created and converted into an array suitable for passing to the `loadModel` function.

4. **Loading the Model**:
    - The `loadModel` function is called with the backend type, model parameters, and a logging callback.

5. **Generating a Response**:
    - The `generateResponse` function is called with a prompt and callbacks for streaming and final response handling.

6. **Printing Responses**:
    - The callbacks print messages to the console as they are received.

### Notes

- **Backend Type**: Update the backend type (`b"CUDA"`) if you need to use a different backend like `CPU`.
- **Model Parameters**: Adjust the model parameters according to your needs.
- **DLL Path**: Ensure the path to the DLL is correct.

Use this script to interact with the LlamaEngine API and print both responses and status messages.