FROM airlab404/dl:cuda10_pytorch_py36

MAINTAINER Marco Cannici "marco.cannici@polimi.it"

 

# Install extras

COPY requirements.yml /requirements.yml

RUN /bin/bash -c ". activate py36 && conda env update -f=/requirements.yml"

 

# Enable jupyter

RUN mkdir /.local

RUN chmod -R 777 /.local

CMD ["bash"]

 

WORKDIR /exp
