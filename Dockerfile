FROM ubuntu:20.04

RUN sudo apt update
RUN sudo apt install python3-pip
RUN pip install -r requirements.txt

COPY df_train.csv ./train.csv
COPY df_test.csv ./test.csv
COPY my_model.h5 ./my_model.h5

COPY test.py ./test.py

CMD ["python3", "test.py"]