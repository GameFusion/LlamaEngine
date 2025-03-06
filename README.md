# LlamaEngine  

**A cross-platform C++ library for dynamically loading and running llama.cpp with CPU, CUDA, Vulkan, and Metal backends.**  

LlamaEngine provides a dynamic way to integrate LLMs into native applications, enabling efficient **LLM interaction, prompt engineering, and natural language processing (NLP)** through backend selection and fine-grained AI context control. It allows **C/C++ clients** to load LLM models, such as **GGUF files** from **Hugging Face**, query metadata, and run inferences on various hardware backends supported by **llama.cpp**.  

This project is a **spinoff from StarGit**, designed to enhance **local AI-driven development and automation pipelines**. It offers **source code analysis, security auditing, and development automation**—all without relying on cloud services, ensuring **security, transparency, and cost-effectiveness** for developers and enterprises.  

## **Key Features**  

- **AI-Powered Automation**: Serves as a backend for **automated AI analysis and process optimization**. 
- **Multi-Backend Support**: Load and run models with **CPU, CUDA, Vulkan, or Metal** backends.  
- **Dynamic Library Loading**: Select and switch backends at runtime without recompiling.  
- **GGUF Metadata Parsing**: Extract model metadata, including parameters and context size.  
- **Prompt-Based Response Generation**: Generate AI responses via the **LlamaModel API**.  
- **Native C/C++ API**: Seamless integration for applications interfacing with LlamaEngine.  
- **Cross-Platform**: Compatible with **Windows, macOS, and Linux**.  
- **Security & Transparency**: Designed for **on-premise AI execution**, ensuring full **source code analysis** and **security auditing**.   

## Build Instructions  

### Using QMake (Recommended)  

You can build the libraries using **qmake** or through **QtCreator**.  

```bash
qmake LlamaEngine.pro
make
```

### Generate Visual Studio Projects (Windows)  

On Windows, you can generate **Visual Studio** projects for different backends:  

```powershell
qmake -tp vc LlamaEngine.pro BACKEND=CUDA
qmake -tp vc LlamaEngine.pro BACKEND=CPU
qmake -tp vc LlamaEngine.pro BACKEND=Vulkan
```

Each generated project must be **compiled separately** using **Visual Studio**.  

### Future CMake Support  

A **CMake build system** is planned for future versions to enhance cross-platform compatibility and integration with modern build environments.  

## Integration with C++ Applications  

A **C++ client**, `LlamaClient`, is available for interfacing with applications. This client allows for:  

- **Loading and querying LLM models** dynamically.  
- **Managing AI context and fine-tuning responses**.  
- **Interacting with different llama.cpp backends** seamlessly. 

## Use Case in StarGit  

LlamaEngine is integrated into **StarGit** to enable **AI-driven Git workflows**, including:  

- **Natural Language Git Commands**: AI-assisted Git interactions.  
- **Development Pipeline Analysis**: Identify potential issues and suggest improvements.  
- **Source Code Security Analysis**: Detect vulnerabilities in commits.  
- **Code Review & Diff Analysis**: AI-assisted PR reviews.  
- **Commit Message Generation**: AI-powered commit messages based on code changes. 

This ensures that **LLM capabilities** are embedded **directly into the development process**, enhancing **security and automation** without external dependencies.  

## Roadmap & Future Enhancements  

- **Session Management**: Support for managing multiple AI contexts simultaneously.  
- **Concurrency & Thread Safety**: Improve parallel processing capabilities.  
- **Fine-Tuned AI Prompt Engineering**: Enhanced control over model responses.  

## Performance & Optimization  

LlamaEngine aims to balance **modern C++20 features** with **low-level C optimizations** for:  

- **Efficient memory management** to handle large AI models.  
- **Optimized computation pipelines** for backend performance.  
- **Minimizing latency in prompt processing**.  
- **Dynamic backend selection** for best performance trade-offs.   

## Why LlamaEngine?  

Unlike cloud-based AI integrations, **LlamaEngine** offers a **local, transparent, and cost-effective** alternative for AI-driven software development. By combining **dynamic backend selection, security, and performance optimizations**, it provides an ideal solution for developers, researchers, and AI enthusiasts looking to integrate **LLMs into their workflows**.  

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
