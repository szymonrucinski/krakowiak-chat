FROM continuumio/miniconda3
WORKDIR /usr/
ENV REPO_NAME = github.com/szymonrucinski/finetune-llm

RUN apt update && apt install build-essential -y && apt install manpages-dev && apt update
RUN apt install git-lfs && apt install curl -y
RUN git clone https://${GIT_TOKEN}@{REPO_NAME} ./src/
WORKDIR /usr/src/deploy/
RUN pip install --no-cache-dir --upgrade -r /usr/src/deploy/requirements.txt

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH
RUN git config --global lfs.largefilewarning false


WORKDIR $HOME/app

COPY --chown=user . $HOME/app

COPY . .
USER root
RUN curl -L https://huggingface.co/szymonrucinski/Krakowiak-7B-v2-GGUF/resolve/main/krakowiak-7B-v2-gguf.Q4_0.bin -o krakowiak-7b.gguf.q4_k_m.bin
USER user
WORKDIR $HOME/app

RUN ls
EXPOSE 7860
CMD ["python", "-u","main.py"]
