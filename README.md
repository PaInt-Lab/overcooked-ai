# Overcooked-AI: LLM Agent Edition

<p align="center">
  <img src="images/layouts.gif" alt="Overcooked Demo" width="600"/>
</p>

## Overview

This repository **extends** [Overcooked-AI](https://github.com/HumanCompatibleAI/overcooked_ai) (Carroll et al., NeurIPS 2019), a cooperative human–AI benchmark from the [Center for Human-Compatible AI](https://humancompatible.ai/). The original environment is based on the video game [_Overcooked_](http://www.ghosttowngames.com/overcooked/) by Ghost Town Games. We keep their environment, layouts, and demo infrastructure, and add an **LLM Agent** stack for coordinated navigation, plan adaptation, and state graph-based decision making.

- **LLM-powered coordination**: Uses OpenAI's GPT-5.4-mini for intelligent action prediction and planning
- **Plan adaptation**: Learns from successful action sequences and adapts future behavior
- **State graph navigation**: Goal-directed navigation through complete state space
- **Real-time coordination**: Dynamic human-robot collaboration with WebSocket communication
- **Advanced workflows**: Supports ingredient washing, chopping, and complex cooking sequences

> **What comes from the original project:** game MDP/engine, layouts, Flask + Phaser demo (`overcooked_demo`), and related tooling from [HumanCompatibleAI/overcooked_ai](https://github.com/HumanCompatibleAI/overcooked_ai).
>
> **What this fork adds:** LLM agents, subtask planning UI, state-graph navigation, plan adaptation, and Docker/Ollama deployment for those components. This is a different coordination approach on their environment (natural language planning rather than RL/BC), not a replacement of the original benchmark.

## System Architecture

The system consists of three main layers:

### **Web Layer** (Flask + WebSocket)

- Real-time communication between humans and AI agents
- Task definition and plan management interface
- Live game visualization with Phaser.js

### **Game Engine** (Overcooked Environment)

- Core game mechanics and state management
- Agent coordination and action processing
- Real-time game loop and scoring

### **AI Layer** (LLM Agents + State Navigation)

- **Coordinated Action Predictor**: Main LLM agent using OpenAI GPT-4o-mini
- **Complete State Graph**: Pre-computed state space for washing+chopping recipes
- **Plan Adaptation**: Learning system that stores and adapts from successful sequences
- **Secondary Action Selector**: Smart robot action coordination

---

## Key Features

### **Intelligent Coordination**

- **Human Action Prediction**: LLM predicts what humans will do next
- **Smart Robot Actions**: Robot selects complementary actions based on predictions
- **Conflict Resolution**: Handles object mismatches and blocking situations
- **State-Aware Decisions**: Considers complete game state for optimal coordination

### **Plan Learning & Adaptation**

- **Success Tracking**: Records successful action sequences automatically
- **Plan Repository**: Stores and retrieves effective coordination patterns
- **Adaptive Behavior**: Uses learned patterns to improve future performance
- **Two-Phase Learning**: State-based and sequential action recording

### **Advanced State Navigation**

- **Complete State Graph**: Pre-computed navigation for washing+chopping workflows
- **Flexible Processing**: Supports any order of ingredient preparation
- **Goal-Directed Navigation**: A\* pathfinding through state space
- **Caching System**: Fast state lookup and action prediction

### **Rich User Experience**

- **Drag & Drop Interface**: Visual subtask creation and reordering
- **LLM Task Generation**: "Ask LLM" button for automatic subtask creation
- **Real-time Visualization**: Live game rendering with special station indicators
- **Plan Confirmation**: Interactive plan review before game execution

---

## Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed
- OpenAI API key (for GPT-4o-mini)

### 1. Environment Setup

#### **Clone the Repository**

```bash
git clone <your-repo-url>
cd Overcooked-AI
```

#### **Setup OpenAI API Key**

Create a `.env` file in `src/overcooked_demo/` with your OpenAI API key:

```bash
cd src/overcooked_demo
echo "OPENAI_API_KEY=your-openai-api-key-here" > .env
```

> **Important:** All setup uses Docker containers. Do not attempt to install Python dependencies locally - the `requirements.txt` file contains legacy dependencies that will conflict with modern Python versions.

### 2. Docker Container Setup

#### **Step 1: Create Docker Network**

```bash
docker network create overcook-net
```

This creates a dedicated network for container communication. Only needs to be executed once during initial setup.

#### **Step 2: Create Ollama Models**

Navigate to the Ollama models directory and create the required models:

```bash
cd ollama_models/task_tagger
ollama create task_tagger -f Modelfile

cd ../subtask_creator
ollama create subtask_creator -f Modelfile
```

#### **Step 3: Build Docker Images**

```bash
# Return to overcooked_demo directory
cd ../..

# Build Ollama Image
docker build --no-cache -t ollama/ollama:latest .

# Build Overcooked Application Image
docker build --no-cache -t overcooked-llm .
```

#### **Step 4: Deploy Containers**

**Run Ollama Container:**

```bash
docker run -d \
  --name ollama-model \
  --network overcook-net \
  --gpus all \
  -p 11434:11434 \
  -v "$HOME/.ollama/models:/root/.ollama/models" \
  ollama/ollama:latest
```

**Run Overcooked Application Container:**

```bash
docker run -d \
  --name overcooked-app \
  --network overcook-net \
  --env-file .env \
  -p 5000:5000 \
  -v "${PWD}:/app" \
  overcooked-llm:latest
```

> **Note for Windows users:** Replace `$HOME` with `$env:USERPROFILE` in the Ollama container command.

#### **Alternative: Docker Compose (Simpler)**

If you prefer a simpler setup, you can use Docker Compose:

```bash
cd src/overcooked_demo
./up.sh  # Development mode
# or
./up.sh production  # Production mode
```

### 3. Play with the LLM Agent

1. Open your browser to [http://localhost](http://localhost) (or [http://localhost:5000](http://localhost:5000) if using manual Docker setup)
2. **Define Your Task**: Enter a task like "Serving Washed and Chopped Onion and Tomato Soup"
3. **Create Subtasks**: Use "Ask LLM" or create subtasks manually with drag & drop
4. **Confirm Sequence**: Review and confirm your subtask sequence
5. **Start Game**: Select "overcooked_llm" as your partner and begin playing!

### 4. Container Management

#### **Access Container Shell**

For debugging or manual command execution:

```bash
docker exec -it overcooked-app bash
```

#### **Monitor Container Logs**

View real-time logs from the application:

```bash
docker logs --since 0s -f overcooked-app
```

#### **Stop Containers**

Using Docker Compose:

```bash
./down.sh
```

Or manually:

```bash
docker stop overcooked-app ollama-model
docker rm overcooked-app ollama-model
```

#### **Rebuild After Changes**

If you modify files outside the `/server` folder, rebuild the image:

```bash
docker build --no-cache -t overcooked-llm .
docker stop overcooked-app
docker rm overcooked-app
# Then run the container again (Step 4)
```

> **Note:** Changes to files inside `/server` are automatically reflected due to volume mounting - no rebuild needed!

---

## How It Works

### **Task Definition & Planning**

1. **User Input**: Define high-level cooking tasks through the web interface
2. **Subtask Generation**: LLM breaks down tasks into atomic subtasks
3. **Task Classification**: Subtasks classified as "primary" (human) or "secondary" (robot)
4. **Event Sequencing**: Tasks grouped into coordinated action sequences

### **Real-Time Coordination**

1. **State Analysis**: Agent analyzes current game state and available actions
2. **Human Prediction**: LLM predicts what the human will do next
3. **Robot Action**: Secondary action selector chooses complementary robot action
4. **Execution**: Both agents execute coordinated actions simultaneously
5. **Learning**: Successful sequences are recorded for future adaptation

### **Plan Adaptation**

1. **Success Detection**: System detects when soup is successfully served
2. **Sequence Recording**: Complete action sequence is stored with metadata
3. **Pattern Learning**: Future predictions can reference successful patterns
4. **Adaptive Behavior**: Agent behavior improves over multiple interactions

---

## System Components (Major)

### **Core LLM Agent** (`coordinated_action_predictor.py`)

- **Model**: OpenAI GPT-4o-mini
- **Features**: State graph navigation, plan adaptation, human coordination
- **Integration**: Connects to state graph, plan repository, and action selector

### **State Navigation** (`complete_state_graph.py`)

- **Purpose**: Complete state space for washing+chopping recipes
- **Features**: Flexible processing order, state validation, action mapping
- **Performance**: Pre-computed and cached for fast lookup

### **Plan Learning** (`action_tracker.py` + `plan_repository.py`)

- **Tracking**: Records successful action sequences with timestamps
- **Storage**: Manages plan repository with metadata and retrieval
- **Adaptation**: Enables learning from past successful interactions

### **Robot Coordination** (`secondary_action_selector.py`)

- **Smart Selection**: Chooses robot actions that complement human actions
- **Object Management**: Handles ingredient fetching, staging, and cleanup
- **Conflict Resolution**: Manages blocking and object mismatch situations

### **Web Interface** (`index.html` + `graphics.js`)

- **Task Management**: Drag & drop subtask creation and reordering
- **LLM Integration**: "Ask LLM" button for automatic task decomposition
- **Game Visualization**: Real-time rendering with Phaser.js
- **Plan Confirmation**: Interactive plan review and confirmation

---

## Project Structure

```
src/overcooked_demo/
├── .env                                # Environment variables (OPENAI_API_KEY)
├── docker-compose.yml                  # Docker orchestration
├── up.sh / down.sh                     # Deployment scripts
├── server/
│   ├── Dockerfile                      # Game server container
│   ├── requirements.txt                # Python dependencies
│   ├── app.py                          # Flask web server & API
│   ├── game.py                         # Core game engine
│   ├── config.json                     # Game configuration
│   ├── plan_session.py                 # Session management
│   ├── complete_state_graph.py         # State space navigation
│   ├── secondary_action_selector.py    # Robot action selection
│   └── llm/
│       ├── agents/
│       │   ├── coordinated_action_predictor.py  # Main LLM agent
│       │   ├── subtask_creator.py              # Task decomposition
│       │   └── subtask_to_event_sequence.py    # Task classification
│       ├── plan_adaptation/
│       │   ├── action_tracker.py               # Plan learning
│       │   └── plan_repository.py             # Plan storage
│       └── ollama/
│           └── ollama_client.py               # LLM communication
├── ollama_models/                      # LLM model configurations
│   ├── Dockerfile                      # Ollama container
│   ├── subtask_creator/Modelfile
│   └── task_tagger/Modelfile
└── static/
    ├── templates/index.html            # Web interface
    └── js/graphics.js                  # Game visualization
```

---

## 🔧 Customization & Extension

### **Modify LLM Behavior**

- **Prompts**: Edit `ollama_models/*/Modelfile` for custom prompts
- **Models**: Switch between different LLM models in `coordinated_action_predictor.py`
- **API**: Change OpenAI model or add other LLM providers

### **Add New Recipes**

- **State Graph**: Extend `complete_state_graph.py` for new ingredient workflows
- **Action Mapping**: Update `secondary_action_selector.py` for new actions
- **UI**: Add new recipe templates in `index.html`

### **Customize Coordination**

- **Learning**: Modify `action_tracker.py` for different learning strategies
- **Planning**: Adjust plan adaptation logic in `plan_repository.py`
- **Selection**: Customize robot action selection in `secondary_action_selector.py`

---

## Research Applications

This system enables research in:

- **Human-AI Coordination**: How can LLMs improve human-robot collaboration?
- **Plan Adaptation**: Can agents learn effective coordination patterns?
- **State Space Navigation**: How does graph-based navigation compare to RL?
- **Natural Language Planning**: Can LLMs understand and execute cooking tasks?
- **Real-time Coordination**: How do agents coordinate in dynamic environments?

---

## Performance & Capabilities

- **Real-time Processing**: 30 FPS game updates with LLM inference
- **State Space**: Complete navigation for washing+chopping workflows
- **Learning**: Automatic adaptation from successful interaction patterns
- **Coordination**: Intelligent human-robot action prediction and selection
- **Scalability**: Modular architecture supports easy extension

---

## Authors & Acknowledgments

- **This fork / LLM Agent edition:** Vito Rizzuto
- **Original Overcooked-AI** ([HumanCompatibleAI/overcooked_ai](https://github.com/HumanCompatibleAI/overcooked_ai)): Micah Carroll, Rohin Shah, Mark K. Ho, Thomas L. Griffiths, Sanjit A. Seshia, Pieter Abbeel, and Anca D. Dragan, Center for Human-Compatible AI. Contact for the original project: Micah Carroll (mdc@berkeley.edu).
- **Overcooked (video game):** Ghost Town Games — the original environment is based on their game.
- **Special thanks:** OpenAI, Ollama, and the open-source LLM communities

If you use this repository, please cite **both** Carroll et al. (NeurIPS 2019) for the Overcooked-AI environment and this fork for the LLM agent work.

---

## License

This project inherits the MIT License from the original Overcooked-AI codebase (Copyright (c) 2019 Center for Human-Compatible AI). See [LICENSE](LICENSE) for details.

---

## References & Further Reading

Please cite the original Overcooked-AI paper when using this environment:

Micah Carroll, Rohin Shah, Mark K. Ho, Thomas L. Griffiths, Sanjit A. Seshia, Pieter Abbeel, and Anca D. Dragan. [On the Utility of Learning about Humans for Human-AI Coordination](https://arxiv.org/abs/1910.05789). NeurIPS 2019.

```bibtex
@inproceedings{carroll2019overcooked,
  author    = {Micah Carroll and Rohin Shah and Mark K. Ho and Tom Griffiths and Sanjit A. Seshia and Pieter Abbeel and Anca D. Dragan},
  title     = {On the Utility of Learning about Humans for Human-{AI} Coordination},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  pages     = {5175--5186},
  year      = {2019}
}
```

- [Overcooked-AI (original codebase)](https://github.com/HumanCompatibleAI/overcooked_ai)
- [BAIR blog post on Overcooked-AI](https://bair.berkeley.edu/blog/2019/10/21/coordination/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Ollama: Run open LLMs locally](https://ollama.com/)
- [Phaser.js Game Framework](https://phaser.io/)

---

## Contributing

This is a research project focused on human-AI coordination. Contributions that advance the state of LLM-based agent coordination are welcome!

For questions or collaboration opportunities, please reach out to the project maintainers.
