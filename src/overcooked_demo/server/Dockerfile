FROM python:3.10-slim

WORKDIR /app

# 1. Install system deps (git, C toolchain, OpenGL, GLib for cv2)
RUN apt-get update && \
    apt-get install -y \
      git \
      libglib2.0-0 \
      libgl1-mesa-dev \
      build-essential && \
    rm -rf /var/lib/apt/lists/*

# 2. Upgrade pip & setuptools
RUN pip install --upgrade pip setuptools

# 3. Copy & sanitize requirements
COPY requirements.txt .
# remove old TF1 x, old greenlet pin, old dnspython pin
RUN sed -i '/^tensorflow==2\.0\.3/d;/^greenlet==0\.4\.16/d;/^dnspython==1\.16\.0/d' requirements.txt

# 4. Install Python deps
RUN pip install --no-cache-dir -r requirements.txt
# add back Python‑3.10‑compatible packages
RUN pip install tensorflow-cpu==2.11.0
RUN pip install "dnspython>=2.0,<3.0"
RUN pip install tf-keras

# 5. Eventlet for production
ARG BUILD_ENV=production
RUN if [ "$BUILD_ENV" = "production" ] ; then pip install eventlet ; fi

# 6. Clone & install Overcooked-AI
ARG OVERCOOKED_BRANCH=master
RUN git clone --recursive \
      https://github.com/HumanCompatibleAI/overcooked_ai.git \
      --branch ${OVERCOOKED_BRANCH} --single-branch /overcooked_ai
# dummy data_dir for human_aware_rl
RUN echo "import os; DATA_DIR=os.path.abspath('.')" \
      >> /overcooked_ai/src/human_aware_rl/data_dir.py
RUN pip install -e /overcooked_ai[harl]

# 7. Copy demo server files (including your graphics .js)
COPY . /app

# 8. Environment defaults
ENV HOST=0.0.0.0 \
    PORT=5000 \
    CONF_PATH=config.json \
    GRAPHICS=overcooked_graphics_v2.2.js \
    EVENTLET_NO_GREENDNS=1

EXPOSE 5000
CMD ["python","-u","app.py"]
