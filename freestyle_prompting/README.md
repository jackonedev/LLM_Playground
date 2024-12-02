## Freestyle Prompting

Each user input passes directly to the model in all turns.

**The interface provides the following settings**:
- **model_name**: List of models available to respond to the user input
- **chain_type**: pre-set configuration values
- **temperature**: the temperature of the model
- **top_p**: the probabilistic sum of tokens that should be considered for each subsequent token
- **write memory**: for reading and writing the current turn into the chat history.
- **read only memory**: for reading only without saving the current turn into the chat history.
- **system_message**: for implementing a system prompt message dynamically.


**How the app manages the memory through the chat history**:
- The app saves all interactions
- When "Write memory" is `True`, the input prompt adjusts with all the previous turns in which the "Write memory" was set to `True`
- When "Read memory" is `True`, the input prompt adjusts with all the previous turns in which the "Write memory" was set to `True` but doesn't save the current turn
- When the chat history is downloaded, all turns will appear with the corresponding metadata


**The current app does not have**:
- Dynamic chat interface
- Pre-built prompts


**Current available models**: gpt-4o, gpt-4o-mini, o1-preview, o1-mini, develop-debugging

- **develop-debugging**: This model would not execute any LLM, instead, it will allow you to enter the models output manually. This is for creating synthetic conversation for prompt design purposes.


**Interface Preview**:

![interface-preview](preview.png)
