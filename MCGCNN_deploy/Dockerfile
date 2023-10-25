FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN apt-get update

RUN pip install --upgrade setuptools
RUN pip install tomli numpy pandas 
RUN pip install pymatgen==2023.9.2 

WORKDIR /app

COPY inference_model /app/inference_model
COPY mcgcnn /app/mcgcnn
# COPY main.py /app/main.py
COPY predict.py /app/predict.py
COPY predict_default.toml /app/predict_default.toml

#RUN apt-get -y install sudo
#RUN apt-get -y install vim

#ARG usr=docker
#RUN useradd -m $usr && echo "$usr:passwd" | chpasswd && adduser $usr sudo
#WORKDIR /home/$usr
#USER $usr

CMD /bin/bash
