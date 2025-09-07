<img width="210" height="217" alt="taskx" src="https://github.com/user-attachments/assets/4e16f1b5-9a73-47cd-a019-b4266d58491f" />

# TaskX-Nano

## Overview
TaskX-Nano is an open-source, lightweight tool that maps natural language processing (NLP) prompts to programming functions, identifying the best-matching functions and their optimal parameters. Unlike traditional LLM-based systems, FunctionX-Nano operates without requiring a large language model, making it fast, resource-efficient, and ideal for developers seeking to integrate NLP-driven automation into their workflows.

## Features
- **NLP-to-Function Mapping**: Interprets natural language prompts and matches them to relevant programming functions.
- **Parameter Optimization**: Extracts and optimizes function parameters from the prompt context.
- **Lightweight Design**: Built for speed and efficiency, requiring minimal computational resources (no LLM dependency).
- **Extensible**: Easily adaptable to support custom functions and various use cases.
- **Open Source**: Freely available for community contributions and improvements.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/rodyrahi/functionx-nano.git
   cd functionx-nano
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note*: Dependencies (e.g., `spacy`, `scikit-learn`) will be listed in `requirements.txt`. No large-scale NLP models like BERT or GPT are required.

3. Set up the environment:
   - Ensure Python 3.8+ is installed.
   - Configure any additional settings (e.g., function library, model weights) as specified in `config.yaml` (to be added).

## Usage
1. **Write a Prompt**: Create a natural language prompt describing the desired function, e.g., "Calculate the square of a number."
2. **Run the Model**:
   ```bash
   python main.py --prompt "Calculate the square of a number"
   ```
3. **Output**: The model returns the matched function (e.g., `square(num)`) and optimized parameters (e.g., `num=5`).

### Example
```bash
$ python main.py --prompt "Sort a list of numbers in ascending order"
Output:
Matched Function: sort_list
Parameters: list=[3, 1, 4, 1, 5], order='ascending'
Result: [1, 1, 3, 4, 5]
```

## How It Works
FunctionX-Nano uses lightweight NLP techniques (e.g., rule-based parsing, keyword matching, or small-scale embeddings) to analyze user prompts. It matches prompts to a predefined or user-defined function library using semantic analysis and similarity scoring. Parameters are extracted and optimized based on entities or constraints in the prompt. Unlike MCP systems reliant on LLMs, FunctionX-Nano achieves similar functionality with minimal resource overhead, making it suitable for low-resource environments.

*Note*: Detailed architecture and implementation details will be documented as the project evolves.

## Why No LLM?
FunctionX-Nano is designed to avoid the computational and memory demands of large language models. By leveraging efficient NLP techniques and a focused function-matching approach, it delivers fast, reliable results without the need for heavy model dependencies, making it ideal for edge devices, small-scale applications, or rapid prototyping.

## Contributing
We welcome contributions! To contribute:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your feature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature/your-feature
   ```
5. Open a pull request.

Please follow the project's coding standards and include tests where applicable.

## Roadmap
- Support for multiple programming languages (e.g., Python, JavaScript).
- Integration of lightweight NLP libraries for enhanced prompt understanding.
- Release of a pre-built function library for common tasks.
- Documentation for adding custom functions and extending the model.
- Support for batch processing of prompts.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For questions or feedback, create a GitHub Issue or contact the maintainer at [rajvndrarahi126@gmail.com](mailto:rajvndrarahi126@gmail.com).
